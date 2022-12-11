from jax import numpy as jnp
from flax import linen as nn


class MaskedConvolution(nn.Module):
    c_out: int
    mask: jnp.ndarray
    dilation: int = 1

    @nn.compact
    def __call__(self, x):
        # Flax's convolution module already supports masking
        # The mask must be the same size as kernel
        # => extend over input and output feature channels
        if len(self.mask.shape) == 2:
            mask_ext = self.mask[..., None, None]
            mask_ext = jnp.tile(mask_ext, (1, 1, x.shape[-1], self.c_out))
        else:
            mask_ext = self.mask
        # Convolution with masking
        x = nn.Conv(
            features=self.c_out,
            kernel_size=self.mask.shape[:2],
            kernel_dilation=self.dilation,
            mask=mask_ext,
        )(x)
        return x
