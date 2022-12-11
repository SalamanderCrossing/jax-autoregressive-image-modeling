from flax import linen as nn
from .stacked_convolution import VerticalStackConvolution
from .stacked_convolution import HorizontalStackConvolution


class GatedMaskedConv(nn.Module):
    dilation: int = 1

    @nn.compact
    def __call__(self, v_stack, h_stack):
        c_in = v_stack.shape[-1]

        # Layers (depend on input shape)
        conv_vert = VerticalStackConvolution(
            c_out=2 * c_in, kernel_size=3, mask_center=False, dilation=self.dilation
        )
        conv_horiz = HorizontalStackConvolution(
            c_out=2 * c_in, kernel_size=3, mask_center=False, dilation=self.dilation
        )
        conv_vert_to_horiz = nn.Conv(2 * c_in, kernel_size=(1, 1))
        conv_horiz_1x1 = nn.Conv(c_in, kernel_size=(1, 1))

        # Vertical stack (left)
        v_stack_feat = conv_vert(v_stack)
        v_val, v_gate = v_stack_feat.split(2, axis=-1)
        v_stack_out = nn.tanh(v_val) * nn.sigmoid(v_gate)

        # Horizontal stack (right)
        h_stack_feat = conv_horiz(h_stack)
        h_stack_feat = h_stack_feat + conv_vert_to_horiz(v_stack_feat)
        h_val, h_gate = h_stack_feat.split(2, axis=-1)
        h_stack_feat = nn.tanh(h_val) * nn.sigmoid(h_gate)
        h_stack_out = conv_horiz_1x1(h_stack_feat)
        h_stack_out = h_stack_out + h_stack

        return v_stack_out, h_stack_out
