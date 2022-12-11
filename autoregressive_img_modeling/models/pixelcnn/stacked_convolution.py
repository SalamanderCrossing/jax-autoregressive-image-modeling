from jax import numpy as jnp
from flax import linen as nn
from .masked_convolution import MaskedConvolution


class VerticalStackConvolution(nn.Module):
    c_out: int
    kernel_size: int
    mask_center: bool = False
    dilation: int = 1

    def setup(self):
        # Mask out all pixels below. For efficiency, we could also reduce the kernel
        # size in height, but for simplicity, we stick with masking here.
        mask = jnp.ones((self.kernel_size, self.kernel_size), dtype=jnp.float32)
        mask = mask.at[self.kernel_size // 2 + 1 :, :].set(0)
        # For the very first convolution, we will also mask the center row
        if self.mask_center:
            mask = mask.at[self.kernel_size // 2, :].set(0)
        # Our convolution module
        self.conv = MaskedConvolution(
            c_out=self.c_out, mask=mask, dilation=self.dilation
        )

    def __call__(self, x):
        return self.conv(x)


class HorizontalStackConvolution(nn.Module):
    c_out: int
    kernel_size: int
    mask_center: bool = False
    dilation: int = 1

    def setup(self):
        # Mask out all pixels on the left. Note that our kernel has a size of 1
        # in height because we only look at the pixel in the same row.
        mask = jnp.ones((1, self.kernel_size), dtype=jnp.float32)
        mask = mask.at[0, self.kernel_size // 2 + 1 :].set(0)
        # For the very first convolution, we will also mask the center pixel
        if self.mask_center:
            mask = mask.at[0, self.kernel_size // 2].set(0)
        # Our convolution module
        self.conv = MaskedConvolution(
            c_out=self.c_out, mask=mask, dilation=self.dilation
        )

    def __call__(self, x):
        return self.conv(x)
