# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial

import torch
import torch.nn as nn

from mmcv.cnn.bricks.transformer import PatchEmbed
from mmcls.models.utils import resize_pos_embed

from vrwkv.vrwkv import Block
from util.pos_embed import get_2d_sincos_pos_embed


class MaskedAutoencoderVRWKV(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, 
                 img_size=224, 
                 patch_size=16, 
                 in_channels=3, 
                 depth=24,
                 embed_dim=1024, 
                 shift_mode='single_shift', 
                 channel_gamma=1/4,
                 shift_pixel=1, 
                 hidden_rate=4, 
                 init_mode='fancy', 
                 init_values=None, 
                 post_norm=False, 
                 key_norm=True, 
                 drop_path_rate=0., 
                 decoder_embed_dim=512, 
                 decoder_depth=8, 
                 interpolate_mode='bicubic',
                 norm_pix_loss=False,
                 with_cp=False):
        super().__init__()

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.embed_dims = embed_dim
        self.patch_embed = PatchEmbed(
            in_channels=in_channels,
            input_size=img_size,
            embed_dims=self.embed_dims,
            conv_type='Conv2d',
            kernel_size=patch_size,
            stride=patch_size,
            bias=True)
        self.patch_resolution = self.patch_embed.init_out_size
        self.num_patches = self.patch_resolution[0] * self.patch_resolution[1]
        self.patch_size = patch_size

        self.cls_token = None
        self.interpolate_mode = interpolate_mode
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))

        dpr_encoder = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        self.blocks = nn.ModuleList([Block(
                n_embd=embed_dim, 
                n_layer=depth, 
                layer_id=i, 
                shift_mode=shift_mode,
                channel_gamma=channel_gamma, 
                shift_pixel=shift_pixel, 
                hidden_rate=hidden_rate,
                init_mode=init_mode, 
                init_values=init_values, 
                post_norm=post_norm, 
                key_norm=key_norm,
                drop_path=dpr_encoder[i], 
                with_cp=with_cp)
            for i in range(depth)])

        self.ln1 = nn.LayerNorm(embed_dim)
        # --------------------------------------------------------------------------
        # --------------------------------------------------------------------------
        # MAE decoder specifics

        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, decoder_embed_dim))  # fixed sin-cos embedding
        dpr_decoder = [x.item() for x in torch.linspace(0, drop_path_rate, decoder_depth)]

        self.decoder_blocks = nn.ModuleList([Block(
                n_embd=decoder_embed_dim, 
                n_layer=decoder_depth, 
                layer_id=i, 
                shift_mode=shift_mode,
                channel_gamma=channel_gamma, 
                shift_pixel=shift_pixel, 
                hidden_rate=hidden_rate,
                init_mode=init_mode, 
                init_values=init_values, 
                post_norm=post_norm, 
                key_norm=key_norm,
                drop_path=dpr_decoder[i],
                with_cp=with_cp)
            for i in range(decoder_depth)])

        self.ln2 = nn.LayerNorm(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * in_channels, bias=True) # decoder to patch
        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.num_patches**.5), cls_token=False)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.num_patches**.5), cls_token=False)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.projection.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        # torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed'}

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_size
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_size
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_encoder(self, x, mask_ratio):
        # embed patches
        x, patch_resolution = self.patch_embed(x)
        x = x + resize_pos_embed(
            self.pos_embed,
            self.patch_resolution,
            patch_resolution,
            mode=self.interpolate_mode,
            num_extra_tokens=0)
        # add pos embed w/o cls token
        x = x + self.pos_embed

        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.random_masking(x, mask_ratio)

        # append cls token
        # cls_token = self.cls_token + self.pos_embed[:, :1, :]
        # cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        # x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for i, blk in enumerate(self.blocks):
            x = blk(x)
        x = self.ln1(x)

        return x, mask, ids_restore, patch_resolution

    def forward_decoder(self, x, ids_restore, patch_resolution):
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence

        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] - x.shape[1], 1)
        x_ = torch.cat([x, mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = x_.contiguous()
        
        # add pos embed
        x = x + resize_pos_embed(
            self.decoder_pos_embed,
            self.patch_resolution,
            patch_resolution,
            mode=self.interpolate_mode,
            num_extra_tokens=0)

        # apply Transformer blocks
        for i, blk in enumerate(self.decoder_blocks):
            x = blk(x)
        x = self.ln2(x)

        # predictor projection
        x = self.decoder_pred(x)
        return x

    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove, 
        """
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward(self, imgs, mask_ratio=0.75):
        latent, mask, ids_restore, patch_resolution = self.forward_encoder(imgs, mask_ratio)
        # print(f'rank: {torch.distributed.get_rank()}, fwd encoder')
        pred = self.forward_decoder(latent, ids_restore, patch_resolution)  # [N, L, p*p*3]
        # print(f'rank: {torch.distributed.get_rank()}, fwd decoder')
        loss = self.forward_loss(imgs, pred, mask)
        # print(f'rank: {torch.distributed.get_rank()}, fwd loss')
        return loss, pred, mask

    def forward_test(self, imgs, mask_ratio=0.75):
        with torch.no_grad():
            latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
            pred = self.forward_decoder(latent, ids_restore)  # [N, L, p*p*3]
            pred_imgs = self.unpatchify(pred)
            return pred_imgs, imgs


def mae_vrwkv_base_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderVRWKV(
        patch_size=16,
        in_channels=3,
        depth=12,
        embed_dim=768,
        shift_mode='single_shift', 
        channel_gamma=None,
        shift_pixel=1, 
        hidden_rate=4, 
        init_mode='fancy', 
        init_values=1e-5, 
        post_norm=True, 
        drop_path_rate=0.1,
        decoder_embed_dim=512, 
        decoder_depth=8, 
        interpolate_mode='bicubic', 
        **kwargs)
    return model

mae_vrwkv_base_patch16 = mae_vrwkv_base_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
