"""Patchwise reference-guided transformer blocks.

Unused registered attention projections remain for checkpoint compatibility.
"""

import torch.nn as nn


class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=3, embed_dim=768, patch_size=16):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        """
        x: [B, 3, H, W]
        return: patch_embeddings: [B, N, embed_dim],
                (H_patch, W_patch): patch-grid height and width
        """
        (B, C, H, W) = x.shape
        x = self.proj(x)
        (H_patch, W_patch) = (x.shape[2], x.shape[3])
        x = x.flatten(2).transpose(1, 2)
        return (x, (H_patch, W_patch))


class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim=768, depth=4, num_heads=8, mlp_ratio=4.0):
        super().__init__()
        self.blocks = nn.ModuleList(
            [TransformerBlock(embed_dim, num_heads, mlp_ratio) for _ in range(depth)]
        )

    def forward(self, x, x_nor=None):
        for blk in self.blocks:
            (x, x_nor) = blk(x, x_nor)
        return (x, x_nor)


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.attn2 = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, embed_dim)
        )

    def forward(self, x, x_nor=None, is_double=True):
        if is_double:
            x_norm = self.norm1(x)
            x_nor_norm = self.norm1(x_nor)
            (attn_out, _) = self.attn(x_norm, x_norm, x_norm - x_nor_norm)
            (attn_nor_out, _) = self.attn(x_nor_norm, x_nor_norm, x_nor_norm)
            x = x + attn_out
            x_nor = x_nor + attn_nor_out
            x_norm = self.norm2(x)
            x_nor_norm = self.norm2(x_nor)
            x = x + self.mlp(x_norm)
            x_nor = x_nor + self.mlp(x_nor_norm)
            return (x, x_nor)
        else:
            x_norm = self.norm1(x)
            (attn_out, _) = self.attn(x_norm, x_norm, x_norm)
            x = x + attn_out
            x_norm = self.norm2(x)
            x = x + self.mlp(x_norm)
            return x


class PatchReconstruction(nn.Module):
    def __init__(self, embed_dim=768, patch_size=16, out_channels=3):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.reproj = nn.Conv2d(embed_dim, out_channels, kernel_size=1, stride=1)

    def forward(self, x, H_patch, W_patch):
        """
        x: [B, N, embed_dim],  N = H_patch * W_patch
        return: [B, 3, H_patch*patch_size, W_patch*patch_size]
        """
        (B, N, C) = x.shape
        x = x.transpose(1, 2)
        x = x.view(B, C, H_patch, W_patch)
        x = self.reproj(x)
        out = nn.functional.interpolate(x, scale_factor=self.patch_size, mode="nearest")
        return out


class ViTAE(nn.Module):
    """
    Patch embedding, dual-input transformer, and spatial reconstruction.

    Input/output spatial dimensions agree; no classification token is used.
    """

    def __init__(
        self,
        in_channels=3,
        out_channels=3,
        patch_size=16,
        embed_dim=256,
        depth=4,
        num_heads=8,
        mlp_ratio=4.0,
        attention_state=True,
        is_double=True,
    ):
        super().__init__()
        self.attention_state = attention_state
        self.is_double = is_double
        self.patch_embed = PatchEmbedding(
            in_channels=in_channels, embed_dim=embed_dim, patch_size=patch_size
        )
        self.transformer = TransformerEncoder(
            embed_dim=embed_dim, depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio
        )
        self.cross_attn_ano = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=0.1, batch_first=True
        )
        self.reconstruction = PatchReconstruction(
            embed_dim=embed_dim, patch_size=patch_size, out_channels=out_channels
        )

    def forward(self, x, x_nor=None):
        if not self.attention_state:
            return x
        if self.is_double:
            (patches_nor, (H_patch, W_patch)) = self.patch_embed(x_nor)
            (patches, (H_patch, W_patch)) = self.patch_embed(x)
            (feat1, feat2) = self.transformer(patches, patches_nor)
            out1 = self.reconstruction(feat1, H_patch, W_patch)
            out2 = self.reconstruction(feat2, H_patch, W_patch)
            return (out1, out2)
        else:
            (patches, (H_patch, W_patch)) = self.patch_embed(x)
            feat = self.transformer(patches)
            out = self.reconstruction(feat, H_patch, W_patch)
            return out
