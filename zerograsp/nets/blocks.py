from typing import Optional

import torch as th
from torch import nn
import lightning.pytorch as pl

import xformers.ops as xops
from xformers.components import PreNorm, Residual
from xformers.factory.block_configs import NormalizationType
from xformers.components.feedforward.mlp import MLP

from zerograsp.nets.mha import MHA


def _get_ln_factory(
    d_model: int,
    use_triton: bool = True,
    normalization: NormalizationType = NormalizationType.LayerNorm,
):
    """
    Handle all the supported residual path configurations.

    ..Note: we return the appropriate constructor, not an actual layer
    """

    def ln_factory(sublayer: nn.Module):
        return Residual(
            layer=PreNorm(d_model, sublayer, normalization, use_triton), scale=None
        )

    return ln_factory


class MAEBlock(pl.LightningModule):
    def __init__(self, config, attn_type="self") -> None:
        super(MAEBlock, self).__init__()

        ln_factory = _get_ln_factory(config.dim_mae)
        mha = MHA(config, attn_type)
        feedforward = MLP(
            config.dim_mae,
            config.ff_dropout,
            config.ff_activation,
            config.ff_hidden_layer_multiplier,
        )
        self.wrap_att = ln_factory(mha)
        self.wrap_ff = ln_factory(feedforward)
        self.attn_type = attn_type
        self.pe_type = config.pe_type

        if self.pe_type == "cpe":
            self.cpe = nn.Conv1d(
                config.dim_mae, config.dim_mae, kernel_size=5, padding=2, stride=1
            )
            self.ln = nn.LayerNorm(config.dim_mae)

    def forward(
        self,
        x: th.Tensor,
        x_index: th.Tensor,
        attn_bias: xops.AttentionBias,
        y: Optional[th.Tensor] = None,
        y_index: Optional[th.Tensor] = None,
    ):
        if self.attn_type == "self":
            if self.pe_type == "cpe":
                x = x + self.ln(self.cpe(x.transpose(1, 2)).transpose(1, 2))
            x = self.wrap_att(inputs=[x], x_index=x_index, attn_bias=attn_bias)
        else:
            if self.pe_type == "cpe":
                x = x + self.ln(self.cpe(x.transpose(1, 2)).transpose(1, 2))
                y = y + self.ln(self.cpe(y.transpose(1, 2)).transpose(1, 2))
            x = self.wrap_att(
                inputs=[x, y], x_index=x_index, y_index=y_index, attn_bias=attn_bias
            )
        x = self.wrap_ff(inputs=[x])
        return x
