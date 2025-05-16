import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from model_utils import TransformerLayer, get_2d_sincos_pos_embed


class StFTBlcok(nn.Module):
    def __init__(
        self,
        cond_time,
        freq_in_channels,
        in_dim,
        out_dim,
        out_channel,
        num_patches,
        modes,
        lift_channel=32,
        dim=256,
        depth=2,
        num_heads=1,
        mlp_dim=256,
        act="relu",
        grid_size=(4, 4),
        layer_indx=0,
    ):
        super(StFTBlcok, self).__init__()
        self.layer_indx = layer_indx
        self.cond_time = cond_time
        self.freq_in_channels = freq_in_channels
        self.modes = modes
        self.out_channel = out_channel
        self.lift_channel = lift_channel
        self.token_embed = nn.Linear(in_dim, dim)
        self.pos_embed = nn.Parameter(
            torch.randn(1, num_patches, dim), requires_grad=False
        )
        self.pos_embed_fno = nn.Parameter(
            torch.randn(1, num_patches, dim), requires_grad=False
        )
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], grid_size)
        pos_embed_fno = get_2d_sincos_pos_embed(self.pos_embed_fno.shape[-1], grid_size)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        self.pos_embed_fno.data.copy_(
            torch.from_numpy(pos_embed_fno).float().unsqueeze(0)
        )
        self.encoder_layers = nn.ModuleList(
            [TransformerLayer(dim, num_heads, mlp_dim, act) for _ in range(depth)]
        )
        self.encoder_layers_fno = nn.ModuleList(
            [TransformerLayer(dim, num_heads, mlp_dim, act) for _ in range(depth)]
        )
        self.head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, out_dim))
        self.p = nn.Linear(freq_in_channels, lift_channel)
        self.linear = nn.Linear(
            modes[0] * modes[1] * (self.cond_time + self.layer_indx) * lift_channel * 2,
            dim,
        )
        self.q = nn.Linear(dim, modes[0] * modes[1] * 1 * lift_channel * 2)
        self.down = nn.Linear(lift_channel, out_channel)

    def forward(self, x):
        x_copy = x
        n, l, _, ph, pw = x.shape
        x_or = x[:, :, : self.cond_time * self.freq_in_channels]
        x_added = x[:, :, (self.cond_time * self.freq_in_channels) :]
        x_or = rearrange(
            x_or,
            "n l (t v) ph pw -> n l ph pw t v",
            t=self.cond_time,
            v=self.freq_in_channels,
        )
        grid_dup = x_or[:, :, :, :, :1, -2:].repeat(1, 1, 1, 1, self.layer_indx, 1)
        x_added = rearrange(
            x_added,
            "n l (t v) ph pw -> n l ph pw t v",
            t=self.layer_indx,
            v=self.freq_in_channels - 2,
        )
        x_added = torch.cat((x_added, grid_dup), axis=-1)
        x = torch.cat((x_or, x_added), axis=-2)
        x = self.p(x)
        x = rearrange(x, "n l ph pw t v -> (n l) v t ph pw")
        x_ft = torch.fft.rfftn(x, dim=[2, 3, 4])[
            :, :, :, : self.modes[0], : self.modes[1]
        ]
        x_ft_real = (x_ft.real).flatten(1)
        x_ft_imag = (x_ft.imag).flatten(1)
        x_ft_real = rearrange(x_ft_real, "(n l) D -> n l D", n=n, l=l)
        x_ft_imag = rearrange(x_ft_imag, "(n l) D -> n l D", n=n, l=l)
        x_ft_real_imag = torch.cat((x_ft_real, x_ft_imag), axis=-1)
        x = self.linear(x_ft_real_imag)
        x = x + self.pos_embed_fno
        for layer in self.encoder_layers_fno:
            x = layer(x)
        x_real, x_imag = self.q(x).split(
            self.modes[0] * self.modes[1] * self.lift_channel, dim=-1
        )
        x_real = x_real.reshape(n * l, -1, 1, self.modes[0], self.modes[1])
        x_imag = x_imag.reshape(n * l, -1, 1, self.modes[0], self.modes[1])
        x_complex = torch.complex(x_real, x_imag)
        out_ft = torch.zeros(
            n * l,
            self.lift_channel,
            1,
            ph,
            pw // 2 + 1,
            dtype=torch.cfloat,
            device=x.device,
        )
        out_ft[:, :, :, : self.modes[0], : self.modes[1]] = x_complex
        x = torch.fft.irfftn(out_ft, s=(1, ph, pw))
        x = rearrange(x, "(n l) v t ph pw -> (n l) ph pw (v t)", n=n, l=l, t=1)
        x = self.down(x)
        x_fno = rearrange(x, "(n l) ph pw c -> n l c ph pw", n=n, l=l)
        x = x_copy
        _, _, _, ph, pw = x.shape
        x = x.flatten(2)
        x = self.token_embed(x) + self.pos_embed
        for layer in self.encoder_layers:
            x = layer(x)
        x = self.head(x)
        x = rearrange(
            x, "n l (c ph pw) -> n l c ph pw", c=self.out_channel, ph=ph, pw=pw
        )
        x = x + x_fno
        return x


class StFT(nn.Module):
    def __init__(
        self,
        cond_time,
        num_vars,
        patch_sizes,
        overlaps,
        in_channels,
        out_channels,
        modes,
        img_size=(50, 50),
        lift_channel=32,
        dim=128,
        vit_depth=3,
        num_heads=1,
        mlp_dim=128,
        act="relu",
    ):
        super(StFT, self).__init__()

        blocks = []
        self.cond_time = cond_time
        self.num_vars = num_vars
        self.patch_sizes = patch_sizes
        self.overlaps = overlaps
        for depth, (p1, p2) in enumerate(patch_sizes):
            H, W = img_size
            cur_modes = modes[depth]
            cur_depth = vit_depth[depth]
            overlap_h, overlap_w = overlaps[depth]

            step_h = p1 - overlap_h
            step_w = p2 - overlap_w

            pad_h = (step_h - (H - p1) % step_h) % step_h
            pad_w = (step_w - (W - p2) % step_w) % step_w
            H_pad = H + pad_h
            W_pad = W + pad_w

            num_patches_h = (H_pad - p1) // step_h + 1
            num_patches_w = (W_pad - p2) // step_w + 1

            num_patches = num_patches_h * num_patches_w
            if depth == 0:
                blocks.append(
                    StFTBlcok(
                        cond_time,
                        num_vars,
                        p1 * p2 * in_channels,
                        out_channels * p1 * p2,
                        out_channels,
                        num_patches,
                        cur_modes,
                        lift_channel=lift_channel,
                        dim=dim,
                        depth=cur_depth,
                        num_heads=num_heads,
                        mlp_dim=mlp_dim,
                        act=act,
                        grid_size=(num_patches_h, num_patches_w),
                        layer_indx=depth,
                    )
                )
            else:
                blocks.append(
                    StFTBlcok(
                        cond_time,
                        num_vars,
                        p1 * p2 * (in_channels + out_channels),
                        out_channels * p1 * p2,
                        out_channels,
                        num_patches,
                        cur_modes,
                        lift_channel=lift_channel,
                        dim=dim,
                        depth=cur_depth,
                        num_heads=num_heads,
                        mlp_dim=mlp_dim,
                        act=act,
                        grid_size=(num_patches_h, num_patches_w),
                        layer_indx=1,
                    )
                )

        self.blocks = nn.ModuleList(blocks)

    def forward(self, x, grid):
        grid_dup = grid[None, :, :, :].repeat(x.shape[0], x.shape[1], 1, 1, 1)
        x = torch.cat((x, grid_dup), axis=2)
        x = rearrange(x, "B L C H W -> B (L C) H W")
        layer_outputs = []
        patches = x
        restore_params = []
        or_patches = x
        if True:
            for depth in range(len(self.patch_sizes)):
                if True:
                    p1, p2 = self.patch_sizes[depth]
                    overlap_h, overlap_w = self.overlaps[depth]

                    step_h = p1 - overlap_h
                    step_w = p2 - overlap_w

                    pad_h = (step_h - (patches.shape[2] - p1) % step_h) % step_h
                    pad_w = (step_w - (patches.shape[3] - p2) % step_w) % step_w
                    padding = (
                        pad_w // 2,
                        pad_w - pad_w // 2,
                        pad_h // 2,
                        pad_h - pad_h // 2,
                    )

                    patches = F.pad(patches, padding, mode="constant", value=0)
                    _, _, H_pad, W_pad = patches.shape

                    h = (H_pad - p1) // step_h + 1
                    w = (W_pad - p2) // step_w + 1

                    restore_params.append(
                        (p1, p2, step_h, step_w, padding, H_pad, W_pad, h, w)
                    )

                    patches = patches.unfold(2, p1, step_h).unfold(3, p2, step_w)
                    patches = rearrange(patches, "n c h w ph pw -> n (h w) c ph pw")

                    processed_patches = self.blocks[depth](patches)

                    patches = rearrange(
                        processed_patches, "n (h w) c ph pw -> n c h w ph pw", h=h, w=w
                    )

                    output = F.fold(
                        rearrange(patches, "n c h w ph pw -> n (c ph pw) (h w)"),
                        output_size=(H_pad, W_pad),
                        kernel_size=(p1, p2),
                        stride=(step_h, step_w),
                    )

                    overlap_count = F.fold(
                        rearrange(
                            torch.ones_like(patches),
                            "n c h w ph pw -> n (c ph pw) (h w)",
                        ),
                        output_size=(H_pad, W_pad),
                        kernel_size=(p1, p2),
                        stride=(step_h, step_w),
                    )
                    output = output / overlap_count
                    output = output[
                        :,
                        :,
                        padding[2] : H_pad - padding[3],
                        padding[0] : W_pad - padding[1],
                    ]
                    layer_outputs.append(output)
                    added = output
                    patches = torch.cat((or_patches, added.detach().clone()), axis=1)

        return layer_outputs
