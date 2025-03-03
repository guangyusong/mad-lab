# adapted from:
# https://github.com/BlinkDL/RWKV-LM

import os
import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.cpp_extension import load


def time_mixer_rwkv7_wrapped_bf16(
    dim: int = 128,
    max_length: int = 1_280,
    head_dim: int = 64,  # RWKV-7 uses 64 as default head size
    dim_att: int = 128,
    head_dim_divisor: int = 8,
    n_layer: int = 1,
    layer_id: int = 0,
    use_jit: bool = False,
    chunk_len: int = 16,  # RWKV-7 specific parameter
    *args, **kwargs
):
    """
    Wrapper to ensure that cuda kernel is only loaded when 
    we actually create an instance of the Time Mixer.
    RWKV-7 implements the Wind Backstepping mechanism.
    """

    if not use_jit:
        def __nop(ob):
            return ob
        MyModule = nn.Module
        MyFunction = __nop
    else:
        MyModule = torch.jit.ScriptModule
        MyFunction = torch.jit.script_method

    if 'TUNE_ORIG_WORKING_DIR' in os.environ:
        base_path = os.getenv("TUNE_ORIG_WORKING_DIR")
    else:
        base_path = ''

    # Define flags for CUDA compilation
    flags = [
        "-res-usage", 
        f'-D_C_={head_dim}', 
        f"-D_CHUNK_LEN_={chunk_len}", 
        "--use_fast_math", 
        "-O3", 
        "-Xptxas -O3", 
        "--extra-device-vectorization"
    ]

    wkv7_cuda = load(
        name="wkv7",
        sources=[
            os.path.join(
                base_path,
                "mad",
                "model",
                "layers",
                "rwkv",
                "cuda",
                "wkv7_op.cpp"
            ),
            os.path.join(
                base_path,
                "mad",
                "model",
                "layers",
                "rwkv",
                "cuda",
                "wkv7_cuda.cu"
            ),
        ],
        verbose=True,
        extra_cuda_cflags=flags
    )
        
    class WindBackstepping(torch.autograd.Function):
        @staticmethod
        def forward(ctx, w, q, k, v, z, b):
            B, T, H, C = w.shape 
            assert T % chunk_len == 0, f"Sequence length {T} must be divisible by chunk_len {chunk_len}"
            assert all(i.dtype == torch.bfloat16 for i in [w, q, k, v, z, b])
            assert all(i.is_contiguous() for i in [w, q, k, v, z, b])
            y = torch.empty_like(v)
            s = torch.empty(B, H, T // chunk_len, C, C, dtype=torch.float32, device=w.device)
            sa = torch.empty(B, T, H, C, dtype=torch.float32, device=w.device)
            wkv7_cuda.forward(w, q, k, v, z, b, y, s, sa)
            ctx.save_for_backward(w, q, k, v, z, b, s, sa)
            return y
            
        @staticmethod
        def backward(ctx, dy):
            assert dy.dtype == torch.bfloat16
            assert dy.is_contiguous()
            w, q, k, v, z, b, s, sa = ctx.saved_tensors
            dw, dq, dk, dv, dz, db = [torch.empty_like(x) for x in [w, q, k, v, z, b]]
            wkv7_cuda.backward(w, q, k, v, z, b, dy, s, sa, dw, dq, dk, dv, dz, db)
            return dw, dq, dk, dv, dz, db

    def RUN_CUDA_RWKV7g(q, w, k, v, a, b, orig_shape=None):
        B, T, HC = q.shape
        
        assert T % chunk_len == 0, f"Sequence length {T} must be divisible by chunk_len {chunk_len}"
        
        q, w, k, v, a, b = [i.view(B, T, HC // head_dim, head_dim) for i in [q, w, k, v, a, b]]
        output = WindBackstepping.apply(w, q, k, v, a, b).view(B, T, HC)
            
        return output

    class TimeMixer_RWKV7(MyModule):
        def __init__(self,
            dim: int = 128,
            head_dim: int = 64,
            dim_att: int = 128,
            head_dim_divisor: int = 8,
            n_layer: int = 1,
            layer_id: int = 0,
            *args, **kwargs
        ):
            super().__init__()
            self.layer_id = layer_id
            self.head_size = head_dim
            self.n_head = dim_att // self.head_size
            assert dim_att % self.n_head == 0
            H = self.n_head
            N = self.head_size
            C = dim


            with torch.no_grad():
                ratio_0_to_1 = layer_id / (n_layer - 1) if n_layer > 1 else 0  # 0 to 1
                ratio_1_to_almost0 = 1.0 - (layer_id / n_layer) if n_layer > 0 else 1.0  # 1 to ~0
                ddd = torch.ones(1, 1, C)
                for i in range(C):
                    ddd[0, 0, i] = i / C

                # Fancy time_mix parameters
                self.x_r = nn.Parameter(1.0 - torch.pow(ddd, 0.2 * ratio_1_to_almost0))
                self.x_w = nn.Parameter(1.0 - torch.pow(ddd, 0.9 * ratio_1_to_almost0))
                self.x_k = nn.Parameter(1.0 - (torch.pow(ddd, 0.9 * ratio_1_to_almost0) + 0.4 * ratio_0_to_1))
                self.x_v = nn.Parameter(1.0 - (torch.pow(ddd, 0.4 * ratio_1_to_almost0) + 0.6 * ratio_0_to_1))
                self.x_a = nn.Parameter(1.0 - torch.pow(ddd, 0.9 * ratio_1_to_almost0))
                self.x_g = nn.Parameter(1.0 - torch.pow(ddd, 0.2 * ratio_1_to_almost0))

                # Helper for orthogonal init
                def ortho_init(x, scale):
                    with torch.no_grad():
                        shape = x.shape
                        if len(shape) == 2:
                            gain = math.sqrt(shape[0] / shape[1]) if shape[0] > shape[1] else 1
                            nn.init.orthogonal_(x, gain=gain * scale)
                        elif len(shape) == 3:
                            gain = math.sqrt(shape[1] / shape[2]) if shape[1] > shape[2] else 1
                            for i in range(shape[0]):
                                nn.init.orthogonal_(x[i], gain=gain * scale)
                        else:
                            assert False
                        return x

                # Time decay parameters
                D_DECAY_LORA = max(32, int(round((1.8*(C**0.5))/32)*32))  # suggestion
                self.w1 = nn.Parameter(torch.zeros(C, D_DECAY_LORA))
                self.w2 = nn.Parameter(ortho_init(torch.zeros(D_DECAY_LORA, C), 0.1))
                decay_speed = torch.ones(C)
                for n in range(C):
                    decay_speed[n] = -7 + 5 * (n / (C - 1)) ** (0.85 + 1.0 * ratio_0_to_1 ** 0.5)
                self.w0 = nn.Parameter(decay_speed.reshape(1, 1, C) + 0.5)  # !!! 0.5 comes from F.softplus !!!

                # A parameters (in-context learning rate)
                D_AAA_LORA = max(32, int(round((1.8*(C**0.5))/32)*32))  # suggestion
                self.a1 = nn.Parameter(torch.zeros(C, D_AAA_LORA))
                self.a2 = nn.Parameter(ortho_init(torch.zeros(D_AAA_LORA, C), 0.1))
                self.a0 = nn.Parameter(torch.zeros(1, 1, C))

                # Value mixing parameters
                D_MV_LORA = max(32, int(round((1.3*(C**0.5))/32)*32))  # suggestion
                self.v1 = nn.Parameter(torch.zeros(C, D_MV_LORA))
                self.v2 = nn.Parameter(ortho_init(torch.zeros(D_MV_LORA, C), 0.1))
                self.v0 = nn.Parameter(torch.zeros(1, 1, C) + 1.0)

                # Gate parameters
                D_GATE_LORA = max(32, int(round((0.6*(C**0.8))/32)*32))  # suggestion
                self.g1 = nn.Parameter(torch.zeros(C, D_GATE_LORA))
                self.g2 = nn.Parameter(ortho_init(torch.zeros(D_GATE_LORA, C), 0.1))

                # Other parameters
                self.k_k = nn.Parameter(torch.ones(1, 1, C) * 0.85)
                self.k_a = nn.Parameter(torch.ones(1, 1, C))
                self.r_k = nn.Parameter(torch.zeros(H, N))

            # Layers and other components
            self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
            self.receptance = nn.Linear(C, C, bias=False)
            self.key = nn.Linear(C, C, bias=False)
            self.value = nn.Linear(C, C, bias=False)
            self.output = nn.Linear(C, C, bias=False)
            self.ln_x = nn.GroupNorm(H, C, eps=(1e-5)*(head_dim_divisor**2)).to(torch.bfloat16)  # !!! notice eps value !!!

            # Initialize weights
            self.receptance.weight.data.uniform_(-0.5/(C**0.5), 0.5/(C**0.5))
            self.key.weight.data.uniform_(-0.05/(C**0.5), 0.05/(C**0.5))
            self.value.weight.data.uniform_(-0.5/(C**0.5), 0.5/(C**0.5))
            self.output.weight.data.zero_()

        @MyFunction
        def _process_inputs(self, x):
            """Process inputs to get key tensors"""
            B, T, C = x.size()
            
            # Time shift mechanism - key to RWKV's parallelizable attention
            xx = self.time_shift(x) - x

            # Mix states with time coefficients
            xr = x + xx * self.x_r
            xw = x + xx * self.x_w
            xk = x + xx * self.x_k
            xv = x + xx * self.x_v
            xa = x + xx * self.x_a
            xg = x + xx * self.x_g

            # Calculate receptance, key, value and other parameters
            r = self.receptance(xr)
            w = -F.softplus(-(self.w0 + torch.tanh(xw @ self.w1) @ self.w2)) - 0.5  # soft-clamp to (-inf, -0.5)
            k = self.key(xk)
            v = self.value(xv)
            a = torch.sigmoid(self.a0 + (xa @ self.a1) @ self.a2)  # a is "in-context learning rate"
            g = torch.sigmoid(xg @ self.g1) @ self.g2

            # Key normalization and mixing
            kk = k * self.k_k
            kk = F.normalize(kk.view(B, T, self.n_head, -1), dim=-1, p=2.0).view(B, T, C)
            k = k * (1 + (a-1) * self.k_a)

            return r, w, k, v, kk, a, g, xv

        @MyFunction
        def forward(self, x, v_first=None):
            B, T, C = x.size()
            H = self.n_head
            orig_shape = (B, T, C)
            x = x.contiguous().to(torch.bfloat16)
            
            r, w, k, v, kk, a, g, xv = self._process_inputs(x)
            
            if self.layer_id == 0:
                v_first = v.clone()
            elif v_first is not None:
                v = v + (v_first - v) * torch.sigmoid(self.v0 + (xv @ self.v1) @ self.v2)  # add value residual
            
            # Convert to bfloat16 for CUDA kernel
            r = r.to(torch.bfloat16)
            w = w.to(torch.bfloat16)
            k = k.to(torch.bfloat16)
            v = v.to(torch.bfloat16)
            kk = kk.to(torch.bfloat16)
            a = a.to(torch.bfloat16)

            # Run CUDA kernel for wind backstepping, handling padding if needed
            x = RUN_CUDA_RWKV7g(r, w, k, v, -kk, kk*a, orig_shape)
            
            # Make sure x is contiguous again after padding/unpadding operations
            x = x.contiguous()
            
            # Apply group normalization and product term - using reshape instead of view
            x = self.ln_x(x.reshape(B * T, C).to(torch.bfloat16)).reshape(B, T, C)
            
            # For the product term, also use reshape for safety
            r_view = r.reshape(B, T, H, -1)
            k_view = k.reshape(B, T, H, -1)
            v_view = v.reshape(B, T, H, -1)
            prod_term = ((r_view * k_view * self.r_k.to(torch.bfloat16)).sum(dim=-1, keepdim=True) * v_view).reshape(B, T, C)
            x = x + prod_term
            
            # Final output
            x = self.output(x * g)
            return x, v_first

    class ChannelMixer_RWKV7(MyModule):
        def __init__(self,
            dim: int = 128,
            dim_inner: int = 512,
            n_layer: int = 1,
            layer_id: int = 0,
            *args, **kwargs
        ):
            super().__init__()
            self.layer_id = layer_id
            self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
            
            with torch.no_grad():
                ratio_1_to_almost0 = 1.0 - (layer_id / n_layer) if n_layer > 0 else 1.0  # 1 to ~0
                ddd = torch.ones(1, 1, dim)
                for i in range(dim):
                    ddd[0, 0, i] = i / dim
                self.x_k = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0**4))
                
            self.key = nn.Linear(dim, dim_inner, bias=False)
            self.value = nn.Linear(dim_inner, dim, bias=False)
            
            self.key.weight.data.uniform_(-0.5/(dim**0.5), 0.5/(dim**0.5))
            self.value.weight.data.zero_()
    
        @MyFunction
        def forward(self, x, v_first=None):
            xx = self.time_shift(x) - x
            
            k = x + xx * self.x_k
            k = torch.relu(self.key(k)) ** 2
            
            return self.value(k), v_first

    # Finally return the time mixer
    return TimeMixer_RWKV7(
        dim=dim,
        max_length=max_length,
        head_dim=head_dim,
        dim_att=dim_att,
        head_dim_divisor=head_dim_divisor,
        n_layer=n_layer,
        layer_id=layer_id,
        *args, **kwargs
    ).to(torch.bfloat16)


def channel_mixer_rwkv7_wrapped(
    dim: int = 128,
    dim_inner: int = 512,
    n_layer: int = 1,
    layer_id: int = 0,
    use_jit: bool = False,
    *args, **kwargs
):
    """
    Wrapper to ensure that cuda kernel is only loaded when
    we actually create an instance of the Channel Mixer.
    """

    if not use_jit:
        def __nop(ob):
            return ob
        MyModule = nn.Module
        MyFunction = __nop
    else:
        MyModule = torch.jit.ScriptModule
        MyFunction = torch.jit.script_method

    class ChannelMixer_RWKV7(MyModule):
        def __init__(self,
            dim: int = 128,
            dim_inner: int = 512,
            n_layer: int = 1,
            layer_id: int = 0,
            *args, **kwargs
        ):
            super().__init__()
            self.layer_id = layer_id
            self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
            
            with torch.no_grad():
                ratio_1_to_almost0 = 1.0 - (layer_id / n_layer) if n_layer > 0 else 1.0  # 1 to ~0
                ddd = torch.ones(1, 1, dim)
                for i in range(dim):
                    ddd[0, 0, i] = i / dim
                self.x_k = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0**4))
                
            self.key = nn.Linear(dim, dim_inner, bias=False)
            self.value = nn.Linear(dim_inner, dim, bias=False)
            
            # Initialize weights
            self.key.weight.data.uniform_(-0.5/(dim**0.5), 0.5/(dim**0.5))
            self.value.weight.data.zero_()

        @MyFunction
        def forward(self, x, v_first=None):
            xx = self.time_shift(x) - x
            
            k = x + xx * self.x_k
            k = torch.relu(self.key(k)) ** 2
            
            return self.value(k), v_first

    return ChannelMixer_RWKV7(
        dim=dim,
        dim_inner=dim_inner,
        n_layer=n_layer,
        layer_id=layer_id,
        *args, **kwargs
    )


if __name__ == '__main__':
    x = torch.randn(2, 128, 128).cuda().to(torch.bfloat16)
    cmixer = channel_mixer_rwkv7_wrapped().cuda().to(torch.bfloat16)
    tmixer = time_mixer_rwkv7_wrapped_bf16().cuda().to(torch.bfloat16)
    y1, v_first = tmixer(x)
    y2, _ = cmixer(y1, v_first)