import torch
import torch.nn as nn
from typing import Optional
from torch import Tensor


class SwiGLU(nn.Module):
    r"""
    A Feed-Forward Network with a SwiGLU (Swish-Gated Linear Unit) Activation.

    SwiGLU is avariant of GLU that uses Swish activation instead of sigmoid,
    providing smoother gradients and often better performance.

    Calculated as:
    ```
    h = Swish(W_gate · x) ⊙ (W_up · x)
    output = W_down · h
    ```

    where:
    - `⊙` = element-wise multiplication
    - `Swish(z) = z · σ(z)` where σ is sigmoid (smoother than plain sigmoid gating)
    - `W_gate · x` = `gating_pathway` (variable holding the gating values with
      Swish activation applied)
    - `W_up · x` = `signal_pathway` (upward projection, expands dimension of x,
      carries the actual information)
    - `h` = hidden = `Swish(gating_pathway) ⊙ signal_pathway`
    - `output` = projection back to original dimension with `W_down`

    The key difference from GLU: Swish(z) = z · σ(z) vs GLU's σ(z). This allows
    gradients to flow through the gating mechanism more smoothly, as Swish is
    non-monotonic and differentiable everywhere.

    Used in modern architectures like Mixtral, PaLM, and others.
    """

    def __init__(
        self,
        hidden_dim: int,
        ffn_dim: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()

        factory_kwargs = {"device": device, "dtype": dtype}

        self.W_gate = nn.Linear(
            in_features=hidden_dim, out_features=ffn_dim, **factory_kwargs
        )
        self.W_up = nn.Linear(
            in_features=hidden_dim, out_features=ffn_dim, **factory_kwargs
        )

        self.W_down = nn.Linear(
            in_features=ffn_dim, out_features=hidden_dim, **factory_kwargs
        )

    def forward(self, x: Tensor) -> Tensor:
        gating_pathway = self.W_gate(x)
        signal_pathway = self.W_up(x)
        swish = gating_pathway * torch.sigmoid(gating_pathway)
        hidden = swish * signal_pathway
        output = self.W_down(hidden)
        return output


if __name__ == "__main__":
    torch.manual_seed(40)
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    hidden_dim = 8
    ffn_dim = 16
    dtype = torch.float16
    print(f"Using device: {device}\ndata type: {dtype}")
    x = torch.rand(3, 8, device=device, dtype=dtype)
    swiglu = SwiGLU(hidden_dim, ffn_dim, device, dtype)
    x_swiglu = swiglu(x)
    print(f"""Out:
    {x_swiglu.shape}""")