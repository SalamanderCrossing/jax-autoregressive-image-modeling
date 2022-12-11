from jax import numpy as jnp
import optax
import jax
from tqdm import tqdm
from jax import random

from flax import linen as nn
from .stacked_convolution import HorizontalStackConvolution, VerticalStackConvolution
from .gated_masked_conv import GatedMaskedConv


class PixelCNN(nn.Module):
    c_in: int
    c_hidden: int

    def setup(self):
        # Initial convolutions skipping the center pixel
        self.conv_vstack = VerticalStackConvolution(
            self.c_hidden, kernel_size=3, mask_center=True
        )
        self.conv_hstack = HorizontalStackConvolution(
            self.c_hidden, kernel_size=3, mask_center=True
        )
        # Convolution block of PixelCNN. We use dilation instead of downscaling
        self.conv_layers = [
            GatedMaskedConv(),
            GatedMaskedConv(dilation=2),
            GatedMaskedConv(),
            GatedMaskedConv(dilation=4),
            GatedMaskedConv(),
            GatedMaskedConv(dilation=2),
            GatedMaskedConv(),
        ]
        # Output classification convolution (1x1)
        self.conv_out = nn.Conv(self.c_in * 256, kernel_size=(1, 1))

    def __call__(self, x):
        # Forward pass with bpd likelihood calculation
        logits = self.pred_logits(x)
        labels = x.astype(jnp.int32)
        nll = optax.softmax_cross_entropy_with_integer_labels(logits, labels)
        bpd = nll.mean() * jnp.log2(jnp.exp(1))
        return bpd

    def pred_logits(self, x):
        """
        Forward image through model and return logits for each pixel.
        Inputs:
            x - Image tensor with integer values between 0 and 255.
        """
        # Scale input from 0 to 255 back to -1 to 1
        x = (x.astype(jnp.float32) / 255.0) * 2 - 1

        # Initial convolutions
        v_stack = self.conv_vstack(x)
        h_stack = self.conv_hstack(x)
        # Gated Convolutions
        for layer in self.conv_layers:
            v_stack, h_stack = layer(v_stack, h_stack)
        # 1x1 classification convolution
        # Apply ELU before 1x1 convolution for non-linearity on residual connection
        out = self.conv_out(nn.elu(h_stack))

        # Output dimensions: [Batch, Height, Width, Channels, Classes]
        out = out.reshape(
            out.shape[0], out.shape[1], out.shape[2], out.shape[3] // 256, 256
        )
        return out

    def sample(self, img_shape, rng, img: None | jnp.ndarray = None):
        """
        Sampling function for the autoregressive model.
        Inputs:
            img_shape - Shape of the image to generate (B,C,H,W)
            img (optional) - If given, this tensor will be used as
                             a starting image. The pixels to fill
                             should be -1 in the input tensor.
        """
        # Create empty image
        if img is None:
            img = jnp.zeros(img_shape, dtype=jnp.int32) - 1
        # We jit a prediction step. One could jit the whole loop, but this
        # is expensive to compile and only worth for a lot of sampling calls.
        get_logits = jax.jit(lambda inp: self.pred_logits(inp))
        # Generation loop
        for h in tqdm(range(img_shape[1]), leave=False):
            for w in range(img_shape[2]):
                for c in range(img_shape[3]):

                    # Skip if not to be filled (-1)
                    if (img[:, h, w, c] != -1).all().item():
                        continue
                    # For efficiency, we only have to input the upper part of the image
                    # as all other parts will be skipped by the masked convolutions anyways
                    logits = get_logits(img)
                    logits = logits[:, h, w, c, :]
                    rng, pix_rng = random.split(rng)
                    img = img.at[:, h, w, c].set(
                        random.categorical(pix_rng, logits, axis=-1)
                    )
        return img
