import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple, Union, List
from transformers import Qwen2ForCausalLM, AutoTokenizer
from safetensors.torch import load_file
import os
from flexllmgen.timer import timers
import numpy as np
import dataclasses
import argparse

from flexllmgen.pytorch_backend import (TorchDevice, TorchDisk, TorchLink, TorchTensor, 
    TorchMixedDevice, DeviceType, general_copy, fix_recursive_import)
from flexllmgen.utils import (Task, ExecutionEnv, GB, T, ValueHolder,
    array_1d, array_2d, array_3d, str2bool, project_decode_latency,
    torch_mem_stats, torch_dtype_to_np_dtype, write_benchmark_log,
    read_benchmark_log)
from flexllmgen.qwen2_config import QwenConfig, get_qwen_config, convert_qwen_weights
from flexllmgen.compression import CompressionConfig


from flex_opt import init_weight_list, get_test_inputs, get_file_inputs 

DUMMY_WEIGHT = "_DUMMY_"  # Use dummy weights for benchmark purposes

@dataclasses.dataclass(frozen=True)
class Policy:
    gpu_batch_size: int
    num_gpu_batches: int

    # percent = a means a%
    w_gpu_percent: float
    w_cpu_percent: float
    cache_gpu_percent: float
    cache_cpu_percent: float
    act_gpu_percent: float
    act_cpu_percent: float

    # Whether to overlap the I/O and compute
    overlap: bool

    # Whether to separate attention and mlp as two layers
    sep_layer: bool

    # Whether to use pinned memory for weights on CPU
    pin_weight: bool

    # Whether to compute attention on CPU
    cpu_cache_compute: bool

    # Sparsity of attention weights
    attn_sparsity: float

    # Compress weights with group-wise quantization
    compress_weight: bool
    comp_weight_config: CompressionConfig

    # Compress KV cache with group-wise quantization
    compress_cache: bool
    comp_cache_config: CompressionConfig

    @property
    def w_disk_percent(self):
        return 100 - self.w_gpu_percent - self.w_cpu_percent

    @property
    def cache_disk_percent(self):
        return 100 - self.cache_gpu_percent - self.cache_cpu_percent

    @property
    def act_disk_percent(self):
        return 100 - self.act_gpu_percent - self.act_cpu_percent


def get_choice(cur_percent, percents, choices):
    percents = np.cumsum(percents)
    assert np.abs(percents[-1] - 100) < 1e-5

    for i in range(len(percents)):
        if cur_percent < percents[i]:
            return choices[i]
    return choices[-1]


def init_weight_list(weight_specs, policy, env):
    dev_percents = [policy.w_disk_percent, policy.w_cpu_percent, policy.w_gpu_percent]
    dev_choices = [env.disk, env.cpu, env.gpu]

    sizes = [np.prod(spec[0]) for spec in weight_specs]
    sizes_cumsum = np.cumsum(sizes)
    ret = []
    for i in range(len(weight_specs)):
        mid_percent = (sizes_cumsum[i] - sizes[i] / 2) / sizes_cumsum[-1]
        home = get_choice(mid_percent * 100, dev_percents, dev_choices)
        shape, dtype, filename = weight_specs[i]

        if len(shape) < 2:
            pin_memory = True
            compress = False
        else:
            pin_memory = policy.pin_weight
            compress = policy.compress_weight

        if not compress:
            weight = home.allocate(shape, dtype, pin_memory=pin_memory)

            if DUMMY_WEIGHT not in filename:
                weight.load_from_np_file(weight_specs[i][2])
            else:
                weight.load_from_np(np.ones(shape, dtype))
                #weight.load_from_np(np.random.rand(*shape).astype(dtype))
        else:
            weight = home.compressed_device.allocate(
                shape, dtype, policy.comp_weight_config, pin_memory=pin_memory)

            if DUMMY_WEIGHT not in filename:
                weight.load_from_np_file(weight_specs[i][2])
            else:
                for i in range(2):
                    x = weight.data[i]
                    x.load_from_np(np.ones(x.shape, torch_dtype_to_np_dtype[x.dtype]))

        ret.append(weight)
    return ret


# ------------------------------
# 核心组件（与模型结构相关）
# ------------------------------
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, device = "cuda"):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, device=device))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(x **2, dim=-1, keepdim=True) + self.eps)
        return self.weight * (x / rms)


def precompute_rope_freqs(dim: int, max_seq_len: int, theta: float, device) :
    
    inv_freq = 1.0 / (theta** (torch.arange(0, dim, 2, device=device) / dim))
    seq = torch.arange(max_seq_len, device=device, dtype=inv_freq.dtype)
    
    concat_inv_freq = torch.concat([inv_freq, inv_freq], dim=-1)
    freqs = torch.outer(seq, concat_inv_freq)
    return torch.cos(freqs), torch.sin(freqs)
    


def eager_attention_core(q, k, v , seq_len, head_dim, device):
    casual_mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
    casual_mask = casual_mask.unsqueeze(0).unsqueeze(0)

    attn_scores :torch.Tensor= (q @ k.transpose(-2, -1)) / (head_dim **0.5)
    attn_scores = attn_scores.masked_fill(casual_mask, -torch.inf)
    attn_weights = F.softmax(attn_scores, dim=-1)
    attn_output = attn_weights @ v

    return attn_output

class SelfAttention:
    def __init__(self, config, env, policy, layer_id):
        self.config = config
        self.env = env
        self.layer_id = layer_id
        self.policy = policy
        self.compute = self.env.gpu
        self.weight_load_dst = (self.compute.compressed_device if policy.compress_weight
            else self.compute)
        self.attention_compute = (self.env.cpu if self.policy.cpu_cache_compute
            else self.env.gpu)

        self.task = None

    def set_task(self, task):
        self.task = task

    def init_weight(self, weight_home, path):
        h, dtype = (self.config.hidden_size, self.config.dtype)
        head_dim = h // self.config.n_qhead
        kv_dim = head_dim * self.config.n_kvhead
        layer_path = os.path.join(path, f"layers.{self.layer_id}")
        attn_path = layer_path + ".self_attn"

        weight_specs = [
            # w_q
            ((h, h), dtype, attn_path + ".q_proj.weight"),
            # b_q
            ((h,), dtype, attn_path + ".q_proj.bias"),
            # w_k
            ((kv_dim, h), dtype, attn_path + ".k_proj.weight"),
            # b_k
            ((kv_dim,), dtype, attn_path + ".k_proj.bias"),
            # w_v
            ((kv_dim, h), dtype, attn_path + ".v_proj.weight"),
            # b_v
            ((kv_dim,), dtype, attn_path + ".v_proj.bias"),
            # w_out
            ((h, h), dtype, attn_path + ".o_proj.weight"),
            # b_out, o_proj have no bias in qwen2
            # ((h,), dtype, path + ".out_proj.bias"),
            # w_ln
            ((h,), dtype, layer_path + ".input_layernorm.weight"),
            # # b_ln, rms has no bias
            # ((h,), dtype, path + "_layer_norm.bias"),
        ]
        weights = init_weight_list(weight_specs, self.policy, self.env)
        weight_home.store(weights)

    def load_weight(self, weight_home, weight_read_buf, k):
        w_q, b_q, w_k, b_k, w_v, b_v, w_out, w_ln = weight_home.val
        if k == 0:
            dst1 = self.weight_load_dst
            dst2 = self.compute
            weight_read_buf.store((
                w_q.smart_copy(dst1), b_q.smart_copy(dst2),
                w_k.smart_copy(dst1), b_k.smart_copy(dst2),
                w_v.smart_copy(dst1), b_v.smart_copy(dst2),
                w_out.smart_copy(dst1), 
                w_ln.smart_copy(dst2)))
            # weight_read_buf.store((
            #     w_q.smart_copy(dst1), b_q.smart_copy(dst2),
            #     w_k.smart_copy(dst1), b_k.smart_copy(dst2),
            #     w_v.smart_copy(dst1), b_v.smart_copy(dst2),
            #     w_out.smart_copy(dst1), b_out.smart_copy(dst2),
            #     w_ln.smart_copy(dst2), b_ln.smart_copy(dst2)))

    def init_cache_one_gpu_batch(self, cache_home):
        if self.policy.cache_gpu_percent == 100:
            device = self.env.gpu
        elif self.policy.cache_cpu_percent == 100:
            device = self.env.cpu
        elif self.policy.cache_disk_percent == 100:
            device = self.env.disk
        else:
            device = self.env.mixed

        if self.policy.compress_cache:
            assert device.device_type != DeviceType.MIXED
            device = device.compressed_device

        cache = device.init_cache_one_gpu_batch(self.config, self.task, self.policy)
        cache_home.store(cache)

    def load_cache(self, cache_home, cache_read_buf, i):
        if i == 0:  # prefill, no cache
            return

        k_home, v_home = cache_home.val

        # Pick code path
        if self.policy.compress_cache:
            path = 0
            dst = self.attention_compute.compressed_device
        else:
            if self.policy.cpu_cache_compute:
                if (k_home.device.device_type == DeviceType.MIXED and
                    k_home.data[0][0] is not None):
                    path = 2
                else:
                    path = 1
            else:
                path = 0
            dst = self.attention_compute

        if path == 0:  # Direct copy
            # shape: (s, b * n_head, head_dim)
            indices = (slice(0, self.task.prompt_len + i),
                       slice(0, k_home.shape[1]))

            if self.policy.attn_sparsity >= 1.0:
                cache_read_buf.store((
                    k_home.smart_copy(dst, indices),
                    v_home.smart_copy(dst, indices),
                ))
            else:
                cache_read_buf.store((
                    k_home.smart_copy(dst, indices),
                    (v_home, False),
                ))
        elif path == 1:  # Copy to CPU temporary workspace
            # shape: (s, b * n_head, head_dim)
            k_buf, v_buf = dst.next_attention_compute_workspace()
            indices = (slice(0, self.task.prompt_len + i - 1),
                       slice(0, k_home.shape[1]))
            general_copy(k_buf, indices, k_home, indices)

            if self.policy.attn_sparsity >= 1.0:
                general_copy(v_buf, indices, v_home, indices)
                cache_read_buf.store(((k_buf, False), (v_buf, False)))
            else:
                cache_read_buf.store(((k_buf, False), ((v_home, v_buf), False)))
        elif path == 2:  # Copy to both GPU and CPU
            # The caches are stored on both GPU and other devices.
            # Compute attention on gpu for caches stored on gpu.
            # Compute attention on cpu for caches stored on cpu/disk.
            gpu_k_buf = k_home.data[0][0]
            gpu_v_buf = v_home.data[0][0]

            # shape: (s, b * n_head, head_dim)
            k_buf, v_buf = dst.next_attention_compute_workspace()
            indices = (slice(0, self.task.prompt_len + i - 1),
                       slice(gpu_k_buf.shape[1], k_home.shape[1]))
            general_copy(k_buf, indices, k_home, indices)
            general_copy(v_buf, indices, v_home, indices)
            cache_read_buf.store((((gpu_k_buf, k_buf,), False),
                                  ((gpu_v_buf, v_buf,), False)))
            assert self.policy.attn_sparsity >= 1.0
        else:
            raise ValueError(f"Invalid path: {path}")

    def store_cache(self, cache_home, cache_write_buf, i):
        # shape: (s, b * n_head, head_dim)
        k_home, v_home = cache_home.val
        k_new, v_new = cache_write_buf.pop()

        if i == self.task.gen_len - 1:  # last token, no need to store cache
            return

        if i == 0:  # prefill
            indices = (slice(0, k_new.shape[0]),
                       slice(0, k_new.shape[1]))
        else:  # decoding
            pos = self.task.prompt_len + i
            indices = (slice(pos - k_new.shape[0], pos),
                       slice(0, k_new.shape[1]))

        general_copy(k_home, indices, k_new, None)
        general_copy(v_home, indices, v_new, None)

    def input_act_shape_and_dtype(self, batch_size, seq_len):
        return (batch_size, seq_len, self.config.hidden_size), self.config.dtype

    def forward(self, hidden, cache_read_buf, weight_read_buf, attention_mask,
                position_embeddings, cache_write_buf, i, k):
        # n_head = self.config.n_head
        n_qhead = self.config.n_qhead
        n_kvhead = self.config.n_kvhead


        donate = [False] * 14   # keep 14, but idx for b_out will not be used
        # 0:hidden, 1:mask, 2-11:weight, 12:kcache, 13:vcache
        # my: 0: hidden, 1:mask, 2:embedding, 345678 qkv w&b, 9:out, 10:ln, 11,12 kvcache
        
        h, donate[0] = hidden.val, True

        if k == self.policy.num_gpu_batches - 1:
            # Clear the weight_read_buf if it is the last gpu batch
            # ((w_q, donate[2]), (b_q, donate[3]), (w_k, donate[4]), (b_k, donate[5]),
            #  (w_v, donate[6]), (b_v, donate[7]), (w_out, donate[8]), (b_out, donate[9]),
            #  (w_ln, donate[10]), (b_ln, donate[11])) = weight_read_buf.pop()
            ((w_q, donate[3]), (b_q, donate[4]), (w_k, donate[5]), (b_k, donate[6]),
             (w_v, donate[7]), (b_v, donate[8]), (w_out, donate[9]),
             (w_ln, donate[10])) = weight_read_buf.pop()
        else:
            # ((w_q, _), (b_q, _), (w_k, _), (b_k, _),
            #  (w_v, _), (b_v, _), (w_out, _), (b_out, _),
            #  (w_ln, _), (b_ln, _)) = weight_read_buf.val
            ((w_q, _), (b_q, _), (w_k, _), (b_k, _),
             (w_v, _), (b_v, _), (w_out, _), 
             (w_ln, _)) = weight_read_buf.val

        if i == 0:  # prefill
            mask, donate[1] = attention_mask.val.smart_copy(self.compute)
            # cos, sin = position_embeddings.val
            # cos, donate_cos = cos.smart_copy(self.compute)
            # sin, donate_sin = sin.smart_copy(self.compute)
            position_embed, donate[2] = position_embeddings.val.smart_copy(self.compute)
            # h, new_k_cache, new_v_cache = self.compute.mha(h, mask, w_q, b_q,
            #     w_k, b_k, w_v, b_v, w_out, b_out, w_ln, b_ln, n_head, donate,
            #     self.policy.compress_cache, self.policy.comp_cache_config)
            h, new_k_cache, new_v_cache = self.compute.gqa(h, mask, position_embed, w_q, b_q,
                w_k, b_k, w_v, b_v, w_out, w_ln, (n_qhead, n_kvhead), donate,
                self.policy.compress_cache, self.policy.comp_cache_config)
            cache_write_buf.store((new_k_cache, new_v_cache))
        else:  # decoding
            mask, donate[1] = attention_mask.val.smart_copy(self.attention_compute)
            position_embed, donate[2] = position_embeddings.val.smart_copy(self.compute)
            (k_cache, donate[12]), (v_cache, donate[13]) = cache_read_buf.pop()
            # h, new_k_cache, new_v_cache = self.compute.mha_gen(h, mask, w_q,
            #     b_q, w_k, b_k, w_v, b_v, w_out, b_out, w_ln, b_ln, n_head,
            #     k_cache, v_cache, donate, self.policy.attn_sparsity,
            #     self.policy.compress_cache, self.policy.comp_cache_config)
            h, new_k_cache, new_v_cache = self.compute.gqa_gen(h, mask, position_embed, w_q,
                b_q, w_k, b_k, w_v, b_v, w_out, w_ln, (n_qhead, n_kvhead),
                k_cache, v_cache, donate, self.policy.attn_sparsity,
                self.policy.compress_cache, self.policy.comp_cache_config)
            cache_write_buf.store((new_k_cache, new_v_cache))

        hidden.val = h

class Attention(nn.Module):
    def __init__(self, hidden_size: int, num_attention_heads: int, num_key_value_heads: int, device: torch.device, layer_idx:int):
        super().__init__()
        
        self.layer_idx = layer_idx
        self.device = device
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = hidden_size // num_attention_heads
        self.num_key_value_groups = num_attention_heads // num_key_value_heads

        self.q_proj = nn.Linear(hidden_size, num_attention_heads * self.head_dim, device=device)
        self.k_proj = nn.Linear(hidden_size, num_key_value_heads * self.head_dim, device=device)
        self.v_proj = nn.Linear(hidden_size, num_key_value_heads * self.head_dim, device=device)
        self.o_proj = nn.Linear(num_attention_heads * self.head_dim, hidden_size, bias=False, device=device)

    def forward(self, hidden_states: torch.Tensor, position_embedding: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape

        # Q/K/V投影与多头拆分
        q = self.q_proj(hidden_states).view(batch_size, seq_len, self.num_attention_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(batch_size, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(batch_size, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
                
        if self.layer_idx == 0:
            torch.save(q, f"my/layer{self.layer_idx}.query")
            torch.save(k, f"my/layer{self.layer_idx}.key")
            torch.save(v, f"my/layer{self.layer_idx}.value")
            print(f" >>> k in shape: {k.shape}")
            print(f" >>> v in shape: {v.shape}")
            print(f" >>> q in shape: {q.shape}")

        # 应用RoPE
        q = apply_rope(q, position_embedding, unsqueeze_dim=0)
        k = apply_rope(k, position_embedding, unsqueeze_dim=0)

        if self.layer_idx == 0:
            torch.save(q, f"my/layer{self.layer_idx}.query_after_rope")
            torch.save(k, f"my/layer{self.layer_idx}.key_after_rope")
        
        # GQA扩展K/V头
        # k = k.repeat_interleave(self.num_key_value_groups, dim=1)
        # v = v.repeat_interleave(self.num_key_value_groups, dim=1)

        # 注意力计算
        # attn_output = eager_attention_core(q, k, v, seq_len, self.head_dim, device=self.device)
        attn_output = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)

        if self.layer_idx == 0:
            torch.save(attn_output, "my/layer0.attn_before_o_proj")
            # torch.save(attn_weights, "my/layer0.attn_weights")
        # 合并多头并投影
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)

        attn_output =  self.o_proj(attn_output)

        if self.layer_idx == 0:
            torch.save(attn_output, "my/layer0.attn_output")
        

        return attn_output



class FeedForward(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, device: torch.device):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False, device=device)
        self.up_proj = nn.Linear(hidden_size, intermediate_size,bias=False, device=device)
        self.down_proj = nn.Linear(intermediate_size, hidden_size,bias=False, device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

class MLP:
    def __init__(self, config, env, policy, layer_id):
        self.config = config
        self.env = env
        self.layer_id = layer_id
        self.policy = policy
        self.compute = self.env.gpu
        self.weight_load_dst = (self.compute.compressed_device if policy.compress_weight
            else self.compute)

        self.task = None

    def set_task(self, task):
        self.task = task

    def init_weight(self, weight_home, path):
        h, dtype = (self.config.hidden_size, self.config.dtype)
        layer_path = os.path.join(os.path.join(path, f"layers.{self.layer_id}"))
        mlp_path = layer_path + ".mlp"
        
        
        intermediated_size = self.config.intermediate_size
        hidden_size = self.config.hidden_size
        dtype = self.config.dtype
        
        weight_specs = [
            # wg
            ((intermediated_size, hidden_size), dtype, mlp_path + ".gate_proj.weight"),
            # wup
            ((intermediated_size, hidden_size), dtype, mlp_path + ".up_proj.weight"),
            # wdown
            ((hidden_size, intermediated_size), dtype, mlp_path + ".down_proj.weight"),
            # w_ln
            ((hidden_size, ), dtype, layer_path + ".post_attention_layernorm.weight")
        ]
        weights = init_weight_list(weight_specs, self.policy, self.env)
        weight_home.store(weights)

    def load_weight(self, weight_home, weight_read_buf, k):
        w_gate, w_up, w_down , w_ln = weight_home.val
        if k == 0:
            dst1 = self.weight_load_dst
            dst2 = self.compute
            # weight_read_buf.store((
                # wi.smart_copy(dst1), bi.smart_copy(dst2),
                # wo.smart_copy(dst1), bo.smart_copy(dst2),
           #     w_ln.smart_copy(dst2), b_ln.smart_copy(dst2)))
            weight_read_buf.store(
                ( w_gate.smart_copy(dst1),
                w_up.smart_copy(dst1),
                w_down.smart_copy(dst2), 
                w_ln.smart_copy(dst2) )
            )

    def init_cache_one_gpu_batch(self, cache_home):
        pass  # do nothing

    def load_cache(self, cache_home, cache_read_buf, i):
        pass  # do nothing

    def store_cache(self, cache_home, cache_write_buf, i):
        pass  # do nothing

    def input_act_shape_and_dtype(self, batch_size, seq_len):
        return (batch_size, seq_len, self.config.hidden_size), self.config.dtype

    def forward(self, hidden, cache_read_buf, weight_read_buf, attention_mask,
                cache_write_buf, i, k):
        donate = [False] * 7
        h, donate[0] = hidden.val, True

        if k == self.policy.num_gpu_batches - 1:
            # Clear the weight_read_buf if it is the last gpu batch
            (
                (w_gate,    donate[1]), 
                (w_up,      donate[2]), 
                (w_down,    donate[3]), 
                (w_ln,      donate[4])
            ) = weight_read_buf.pop()
        else:
            (
                (w_gate,    _), 
                (w_up,      _), 
                (w_down,    _), 
                (w_ln,      _)
            ) = weight_read_buf.val

        h = self.compute.qwen2mlp(h, w_gate, w_up, w_down, w_ln, donate)
        hidden.val = h

class Qwen2Block(nn.Module):
    def __init__(self, hidden_size: int, num_attention_heads: int, num_key_value_heads: int, 
                 intermediate_size: int, rms_norm_eps: float, device: torch.device, layer_idx:int):
        super().__init__()
        self.input_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.attention = Attention(hidden_size, num_attention_heads, num_key_value_heads, device, layer_idx=layer_idx)
        self.post_attention_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.feed_forward = FeedForward(hidden_size, intermediate_size, device)

    def forward(self, hidden_states: torch.Tensor, position_embedding: torch.Tensor) -> torch.Tensor:
        # 注意力残差
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.attention(hidden_states, position_embedding)
        hidden_states = residual + hidden_states

        # 前馈网络残差
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.feed_forward(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states

class TransformerLayer:
    def __init__(self, config, env, policy, i):
        self.attention = SelfAttention(config, env, policy, i)
        self.mlp = MLP(config, env, policy, i)
        self.policy = policy
        self.compute = self.attention.compute

    def set_task(self, task):
        self.attention.set_task(task)
        self.mlp.set_task(task)

    def init_weight(self, weight_home, path):
        home1, home2 = ValueHolder(), ValueHolder()
        self.attention.init_weight(home1, path)
        self.mlp.init_weight(home2, path)
        weight_home.store((home1, home2))

    def load_weight(self, weight_home, weight_read_buf, k):
        read_buf1, read_buf2 = ValueHolder(), ValueHolder()
        home1, home2 = weight_home.val
        self.attention.load_weight(home1, read_buf1, k)
        self.mlp.load_weight(home2, read_buf2, k)
        if k == 0:
            weight_read_buf.store((read_buf1, read_buf2))

    def init_cache_one_gpu_batch(self, cache_home):
        self.attention.init_cache_one_gpu_batch(cache_home)

    def load_cache(self, cache_home, cache_read_buf, i):
        self.attention.load_cache(cache_home, cache_read_buf, i)

    def store_cache(self, cache_home, cache_write_buf, i):
        self.attention.store_cache(cache_home, cache_write_buf, i)

    def forward(self, 
                hidden, 
                cache_read_buf, weight_read_buf, 
                attention_mask, position_embedding, 
                cache_write_buf, i, k):
        # i : id of layer 
        # k : id of gpu batch
        if k == self.policy.num_gpu_batches - 1:
            read_buf1, read_buf2 = weight_read_buf.pop()
        else:
            read_buf1, read_buf2 = weight_read_buf.val

        print(f" >>> pre layer-{i} attention")
        self.attention.forward(hidden, cache_read_buf, read_buf1, attention_mask, position_embedding, 
                               cache_write_buf, i, k)
        print(f" >>> post layer-{i} attention")
        self.mlp.forward(hidden, None, read_buf2, attention_mask, None, i, k)
        print(f" >>> post layer-{i} mlp")

class InputEmbed:
    def __init__(self, config, env, policy):
        self.config = config
        self.env = env
        self.policy = policy
        self.compute = self.env.gpu
        self.weight_load_dst = (self.compute.compressed_device if policy.compress_weight
            else self.compute)

        self.task = None

    def set_task(self, task):
        self.task = task

    def init_weight(self, weight_home, path):
        v, h, dtype = (self.config.vocab_size, self.config.hidden_size,
            self.config.dtype)
        path = os.path.join(path, "")
        weight_specs = [
            # w_token
            ((v, h), dtype, path + "embed_tokens.weight"), 
        ]
        weights = init_weight_list(weight_specs, self.policy, self.env)

        weight_home.store(weights)

    def load_weight(self, weight_home, weight_read_buf, k):
        w_token, = weight_home.val
        if k == 0:
            dst = self.weight_load_dst
            weight_read_buf.store((w_token.smart_copy(dst), ))

    def init_cache_one_gpu_batch(self, cache_home):
        pass  # do nothing

    def load_cache(self, cache_home, cache_read_buf, i):
        pass  # do nothing

    def store_cache(self, cache_home, cache_write_buf, i):
        pass  # do nothing

    def input_act_shape_and_dtype(self, batch_size, seq_len):
        return (batch_size, seq_len), np.int64

    def forward(self, hidden, cache_read_buf, weight_read_buf, attention_mask, position_embedding, 
                cache_write_buf, i, k):
        # Compute input embedding
        donate = [False] * 4
        h, donate[0] = hidden.val, True
        mask, donate[1] = attention_mask.val.smart_copy(self.compute)

        if k == self.policy.num_gpu_batches - 1:
            # Clear the weight_read_buf if it is the last gpu batch
            (w_token, donate[2]),  = weight_read_buf.pop()
        else:
            (w_token, _),  = weight_read_buf.val

        print(f" >>> hidden in dtype: {h.data.dtype}")
        h = self.compute.qwen_input_embed(h, w_token, self.config.pad_token_id, donate)
        hidden.val = h


class OutputEmbed:
    def __init__(self, config, env, policy):
        self.config = config
        self.env = env
        self.policy = policy
        self.compute = self.env.gpu
        self.weight_load_dst = (self.compute.compressed_device if policy.compress_weight
            else self.compute)

        self.task = None

    def set_task(self, task):
        self.task = task

    def init_weight(self, weight_home, path):
        v, h, dtype = (self.config.vocab_size, self.config.hidden_size,
            self.config.dtype)
        path = os.path.join(path, "")
        weight_specs = [
            # w_ln
            ((h,), dtype, path + "norm.weight"),
            # w_token
            ((v, h), dtype, path + "embed_tokens.weight"),
        ]
        weights = init_weight_list(weight_specs, self.policy, self.env)

        weight_home.store(weights)

    def load_weight(self, weight_home, weight_read_buf, k):
        w_ln, b_ln, w_token = weight_home.val
        if k == 0:
            dst1 = self.weight_load_dst
            dst2 = self.compute
            weight_read_buf.store((w_ln.smart_copy(dst2), b_ln.smart_copy(dst2),
                w_token.smart_copy(dst1)))

    def init_cache_one_gpu_batch(self, cache_home):
        pass  # do nothing

    def load_cache(self, cache_home, cache_read_buf, i):
        pass  # do nothing

    def store_cache(self, cache_home, cache_write_buf, i):
        pass  # do nothing

    def input_act_shape_and_dtype(self, batch_size, seq_len):
        return (batch_size, seq_len, self.config.hidden_size), self.config.dtype

    def forward(self, 
                hidden, 
                cache_read_buf, weight_read_buf, 
                attention_mask, position_embedding, 
                cache_write_buf, i, k):
        donate = [False] * 3
        h, donate[0] = hidden.val, True

        if k == self.policy.num_gpu_batches - 1:
            # Clear the weight_read_buf if it is the last gpu batch
            (w_ln, donate[1]), (w_token, donate[2]) = weight_read_buf.pop()
        else:
            (w_ln, _), (w_token, _) = weight_read_buf.val

        h = self.compute.qwen_output_embed(h, w_ln, w_token, donate,
            self.task.do_sample, self.task.temperature)
        hidden.val = h

class QwenLM:
    def __init__(self,
                 config: Union[str, QwenConfig],
                 env: ExecutionEnv,
                 path: str,
                 policy: Policy):

        # path is the current
        if isinstance(config, str):
            config = get_qwen_config(config)
        self.config = config
        self.env = env
        self.path = path
        self.policy = policy
        self.num_gpu_batches = policy.num_gpu_batches

        layers = []
        layers.append(InputEmbed(self.config, self.env, self.policy))
        for j in range(self.config.num_hidden_layers):
            if policy.sep_layer:
                layers.append(SelfAttention(self.config, self.env, self.policy, j))
                layers.append(MLP(self.config, self.env, self.policy, j))
            else:
                layers.append(TransformerLayer(self.config, self.env, self.policy, j))
        layers.append(OutputEmbed(self.config, self.env, self.policy))
        self.layers = layers
        self.num_layers = len(layers)

        if self.policy.act_gpu_percent == 100:
            self.act_home = self.env.gpu
        elif self.policy.act_cpu_percent == 100:
            self.act_home = self.env.cpu
        elif self.policy.act_disk_percent == 100:
            self.act_home = self.env.disk
        else:
            raise NotImplementedError()

        # CUDA streams
        self.load_weight_stream = torch.cuda.Stream()
        self.load_cache_stream = torch.cuda.Stream()
        self.store_cache_stream = torch.cuda.Stream()

        # Intermediate tensors
        # The following buffers store values used
        # for the i-th token, j-th layer, k-th gpu batch.
        num_layers, num_gpu_batches = self.num_layers, self.policy.num_gpu_batches

        # cache[j][k]
        self.cache_home = array_2d(num_layers, num_gpu_batches, ValueHolder)
        self.cache_read_buf = array_2d(num_layers, num_gpu_batches, ValueHolder)
        self.cache_write_buf = array_2d(num_layers, num_gpu_batches, ValueHolder)
        # weight[j]
        self.weight_read_buf = array_1d(num_layers, ValueHolder)
        # attention_mask[k]
        self.attention_mask = array_1d(num_gpu_batches, ValueHolder)
        # position_embedding[k]
        self.position_embedding = array_1d(num_gpu_batches, ValueHolder)

        self.task = None
        self.init_all_weights()

    def set_task(self, task):
        self.task = task
        for l in self.layers:
            l.set_task(task)

    def init_weight(self, j):
        expanded_path = os.path.abspath(os.path.expanduser(
            os.path.join(self.path, f"{self.config.name}-np")))
        check_path = os.path.join(expanded_path, "embed_tokens.weight")
        if not os.path.exists(check_path) and DUMMY_WEIGHT not in check_path:
            convert_qwen_weights(self.config.name, self.path)

        self.layers[j].init_weight(self.weight_home[j], expanded_path)

    def load_weight(self, i, j, k, overlap=True):
        # Handle corner cases
        if j == self.num_layers:
            j = 0
            i += 1
            if i == self.execute_gen_len:
                return

        # Load from weight_home to weight_read_buf
        if overlap:
            with torch.cuda.stream(self.load_weight_stream):
                self.layers[j].load_weight(self.weight_home[j], self.weight_read_buf[j], k)
        else:
            self.layers[j].load_weight(self.weight_home[j], self.weight_read_buf[j], k)

    def delete_weight(self, j, k):
        if k == 0:
            for x in self.weight_home[j].pop():
                if isinstance(x, ValueHolder):
                    for y in x.pop():
                        y.delete()
                else:
                    x.delete()

    def init_cache(self, j, k):
        self.layers[j].init_cache_one_gpu_batch(self.cache_home[j][k])

    def load_cache(self, i, j, k, overlap=True):
        # Handle corner cases
        if i == 0:  # prefill, no cache
            return
        if k == self.num_gpu_batches:
            k = 0
            j += 1
        if j == self.num_layers:
            j = 0
            i += 1
            if i == self.execute_gen_len:
                return

        # Load from cache_home to cache_read_buf
        if overlap:
            with torch.cuda.stream(self.load_cache_stream):
                self.layers[j].load_cache(self.cache_home[j][k], self.cache_read_buf[j][k], i)
        else:
            self.layers[j].load_cache(self.cache_home[j][k], self.cache_read_buf[j][k], i)

    def store_cache(self, i, j, k, overlap=True):
        # Handle corner cases
        if k == -1:
            k = self.num_gpu_batches - 1
            j -= 1
        if j == -1:
            j = self.num_layers - 1
            i -= 1
            if i == -1:
                return
        if i == self.task.gen_len - 1:  # last token, no need to store cache
            self.cache_write_buf[j][k].pop()
            return

        # Store cache_write_buf to cache_home
        # Delete cache_write_buf
        if overlap:
            with torch.cuda.stream(self.store_cache_stream):
                self.layers[j].store_cache(self.cache_home[j][k], self.cache_write_buf[j][k], i)
        else:
            self.layers[j].store_cache(self.cache_home[j][k], self.cache_write_buf[j][k], i)

    def delete_cache(self, j, k):
        v = self.cache_home[j][k].pop()
        if v:
            for x in v:
                x.delete()

    def load_hidden(self, i, j, k):
        # Handle corner cases
        if k == self.num_gpu_batches:
            k = 0
            j += 1
        if j == self.num_layers:
            j = 0
            i += 1
            if i == self.execute_gen_len:
                return

        # Load to hidden states buffers
        dst = self.layers[j].compute
        if j == 0:
            gpu_batch_size = self.policy.gpu_batch_size
            left, right = k * gpu_batch_size, (k + 1) * gpu_batch_size
            if i == 0:  # load from the input ids
                val = dst.allocate((gpu_batch_size, self.task.prompt_len), np.int32)
                val.load_from_np(self.output_ids[left:right, :self.task.prompt_len])
            else:  # load from the last generated token
                pos = self.task.prompt_len + i
                val = dst.allocate((gpu_batch_size, 1), np.int32)
                val.load_from_np(self.output_ids[left:right, pos-1:pos])
        else:  # load from the last layer
            val = self.hidden[i][j-1][k].pop().move(dst)
        self.hidden[i][j][k].store(val)

    def store_hidden(self, i, j, k):
        # Handle corner cases
        if k == -1:
            k = self.num_gpu_batches - 1
            j -= 1
        if j == -1:
            j = self.num_layers - 1
            i -= 1
            if i == -1:
                return

        # Store to hidden states buffers
        if j == self.num_layers - 1:  # store to output
            gpu_batch_size = self.policy.gpu_batch_size
            left, right = k * gpu_batch_size, (k + 1) * gpu_batch_size
            ids = self.hidden[i][j][k].pop().data.detach().cpu().numpy()
            pos = self.task.prompt_len + i
            if self.task.stop:
                stopped = self.stopped[left:right]
                self.output_ids[left:right, pos:pos+1] = np.where(
                    stopped, self.config.pad_token_id, ids)
                stopped[:] = np.logical_or(stopped, ids == self.task.stop)
            else:
                self.output_ids[left:right, pos:pos+1] = ids
        else:  # move to home
            x = self.hidden[i][j][k]
            if x.val:  # x may already be moved due to overlapping
                x.val = x.val.move(self.act_home)

    def compute_layer(self, i, j, k):
        # Update the hidden in place
        # Clear the weight_read_buf if it is the last gpu batch
        # Clear the cache_read_buf
        # Run layer computation
        self.layers[j].forward(self.hidden[i][j][k], self.cache_read_buf[j][k],
            self.weight_read_buf[j], self.attention_mask[k], self.position_embedding[k], 
            self.cache_write_buf[j][k], i, k)

    def sync(self):
        self.env.disk.synchronize()
        torch.cuda.synchronize()

    def init_all_weights(self):
        self.weight_home = array_1d(self.num_layers, ValueHolder)
        for j in range(self.num_layers):
            self.init_weight(j)

    def delete_all_weights(self):
        for j in range(self.num_layers):
            self.delete_weight(j, 0)

    def update_attention_mask(self, i, k):
        # init attn mask
        if i > 0:
            mask = self.attention_mask[k]
            assert mask.val is not None
            mask.val = mask.val.device.extend_attention_mask(mask.val, [True])
            return

        # init 
        gpu_batch_size = self.policy.gpu_batch_size
        left = k * gpu_batch_size
        right = left + gpu_batch_size
        input_ids = self.output_ids[left:right, :self.task.prompt_len]

        attention_compute = (self.env.cpu if self.policy.cpu_cache_compute
            else self.env.gpu)
        val = attention_compute.allocate(
            (self.policy.gpu_batch_size, self.task.prompt_len), bool)
        val.load_from_np((input_ids != self.config.pad_token_id))
        self.attention_mask[k].store(val)

    def update_position_embedding(self, i, k):
        # init position_embed
        if i == 0:
            # positional_embedding used in attn
            attention_compute = (self.env.cpu if self.policy.cpu_cache_compute
                else self.env.gpu)

            gpu_bs = self.policy.gpu_batch_size
            head_dim = self.config.hidden_size // self.config.n_qhead
            max_seq_len = self.task.prompt_len + self.task.gen_len
            alloc_shape = (2, gpu_bs, max_seq_len, head_dim) # [ cos, sin ]
            # cos_sin = attention_compute.allocate( alloc_shape , float)

            cos, sin = precompute_rope_freqs(head_dim, max_seq_len, self.config.rope_theta, device=attention_compute.dev)
            cos_sin  = torch.stack([cos, sin])

            cos_sin = TorchTensor.create_from_torch(cos_sin, device=attention_compute)
            
            self.position_embedding[k].store(cos_sin) 
        # if i > 0:
        #     position_embed = self.position_embedding[k]
        #     assert position_embed.val is not None
        #     position_embed.val = position_embed.val.device.extend_position_embedding(position_embed, )

    def generate(self,
                 inputs: Union[np.array, List[List[int]]],
                 max_new_tokens: int = 32,
                 do_sample: bool = False,
                 temperature: float = 1.0,
                 stop: Optional[int] = None,
                 debug_mode: Optional[str] = None,
                 cut_gen_len: Optional[int] = None,
                 verbose: int = 0):
        task = Task(
            inputs=inputs,
            prompt_len=len(inputs[0]),
            gen_len=max_new_tokens,
            cut_gen_len=cut_gen_len,
            do_sample=do_sample,
            temperature=temperature,
            stop=stop,
        )
        num_layers = self.num_layers
        num_gpu_batches = self.num_gpu_batches
        gpu_batch_size = self.policy.gpu_batch_size
        overlap = self.policy.overlap
        prompt_len, gen_len = task.prompt_len, task.gen_len
        self.execute_gen_len = task.cut_gen_len if task.cut_gen_len else task.gen_len

        # Output token ids
        self.output_ids = np.full((len(task.inputs), prompt_len + gen_len),
            self.config.pad_token_id, dtype=np.int32)
        self.stopped = np.zeros((len(task.inputs), 1), dtype=bool)
        self.output_ids[:, :prompt_len] = np.asarray(task.inputs)
        assert gpu_batch_size * num_gpu_batches == len(task.inputs)

        # Intermediate tensors
        # The following buffers store values used
        # for the i-th token, j-th layer, k-th gpu batch.
        num_layers, num_gpu_batches = self.num_layers, self.policy.num_gpu_batches
        for j in range(num_layers):
            for k in range(num_gpu_batches):
                self.cache_home[j][k].clear()
                self.cache_read_buf[j][k].clear()
                self.cache_write_buf[j][k].clear()
        for j in range(num_layers):
            self.weight_read_buf[j].clear()
        for k in range(num_gpu_batches):
            self.attention_mask[k].clear()
        self.hidden = array_3d(gen_len, num_layers, num_gpu_batches, ValueHolder)

        # Init cache
        self.set_task(task)
        for j in range(num_layers):
            for k in range(num_gpu_batches):
                self.init_cache(j, k)
        if self.policy.cpu_cache_compute:
            self.env.cpu.init_attention_compute_workspace(self.config, self.task, self.policy)

        # Generate
        if debug_mode is None:
            if not overlap:
                # No overlap, easy to understand, suitable for debugging
                self.generation_loop_normal()
            else:
                # Overlap I/O and compute
                if num_gpu_batches == 1:
                    self.generation_loop_overlap_single_batch()
                else:
                    self.generation_loop_overlap_multi_batch()
        elif debug_mode == "fewer_batch":
            # Run fewer layeres and batches for debugging
            if num_gpu_batches == 1:
                self.generation_loop_debug_single_batch()
            else:
                self.generation_loop_debug_multi_batch()
        elif debug_mode == "breakdown":
            # No overlap, fewer batches, execution time breakdown
            self.generation_loop_debug_normal()
        else:
            raise ValueError("Invalid debug mode: {debug_mode}")

        # Delete cache
        for j in range(num_layers):
            for k in range(num_gpu_batches):
                self.delete_cache(j, k)
        if self.policy.cpu_cache_compute:
            self.env.cpu.del_attention_compute_workspace()

        return self.output_ids

    def generation_loop_normal(self):
        for i in range(self.execute_gen_len):
            timers("generate").start()
            for k in range(self.num_gpu_batches):
                self.update_attention_mask(i, k)
                self.update_position_embedding(i, k)
            for j in range(self.num_layers):
                for k in range(self.num_gpu_batches):
                    self.load_weight(i, j, k, overlap=False)

                for k in range(self.num_gpu_batches):
                    self.load_cache(i, j, k, overlap=False)
                    self.load_hidden(i, j, k)
                    self.compute_layer(i, j, k)
                    self.store_hidden(i, j, k)
                    self.store_cache(i, j, k, overlap=False)
            timers("generate").stop()

    def generation_loop_debug_normal(self):
        execute_num_batches = 20
        batch_ct = 0
        pbar = tqdm(total=execute_num_batches)
        timers("prefill_total").reset()
        timers("decoding_gpu_batch").reset()

        timers("load_weight").reset()
        timers("load_cache_prefill").reset()
        timers("load_cache_decoding").reset()
        timers("store_cache_prefill").reset()
        timers("store_cache_decoding").reset()
        timers("compute_layer_prefill").reset()
        timers("compute_layer_decoding").reset()
        load_weight_timer = timers("load_weight")

        for i in range(self.execute_gen_len):
            if i == 0:
                timers("prefill_total").start()
                load_cache_timer = timers("load_cache_prefill")
                store_cache_timer = timers("store_cache_prefill")
                compute_layer_timer = timers("compute_layer_prefill")
            else:
                load_cache_timer = timers("load_cache_decoding")
                store_cache_timer = timers("store_cache_decoding")
                compute_layer_timer = timers("compute_layer_decoding")

            for k in range(self.num_gpu_batches):
                self.update_attention_mask(i, k)
                self.update_position_embedding(i, k)

            for j in range(self.num_layers):
                if i > 0: timers("decoding_gpu_batch").start()

                load_weight_timer.start(self.sync)
                for k in range(self.num_gpu_batches):
                    self.load_weight(i, j, k)
                load_weight_timer.stop(self.sync)

                for k in range(self.num_gpu_batches):
                    load_cache_timer.start(self.sync)
                    self.load_cache(i, j, k)
                    load_cache_timer.stop(self.sync)
                    self.load_hidden(i, j, k)
                    compute_layer_timer.start(self.sync)
                    self.compute_layer(i, j, k)
                    compute_layer_timer.stop(self.sync)
                    self.store_hidden(i, j, k)
                    store_cache_timer.start(self.sync)
                    self.store_cache(i, j, k)
                    store_cache_timer.stop(self.sync)

                if i > 0:
                    timers("decoding_gpu_batch").stop()
                    pbar.update(1)
                    batch_ct += 1
                if batch_ct >= execute_num_batches: break
            if batch_ct >= execute_num_batches: break
            if i == 0: timers("prefill_total").stop(self.sync)

        # Convert "decoding_gpu_batch" timer to "generate" timer
        batch_cost = np.mean(timers("decoding_gpu_batch").costs[10:])
        for i in range(self.execute_gen_len):
            if i == 0:
                timers("generate").costs.append(timers("prefill_total").costs[0])
            else:
                timers("generate").costs.append(self.num_layers * batch_cost)

        # Debug the costs of individual functions
        print(f"#layers: {self.num_layers}")

        print(f"#batches prefill:  "
              f"{self.num_layers * self.num_gpu_batches}")
        print(f"#batches decoding: "
              f"{(self.task.gen_len - 1) * self.num_layers * self.num_gpu_batches}")
        print(f"load_weight            (per-layer)"
              f": {np.mean(timers('load_weight').costs):.6f} s")
        for stage in ["prefill", "decoding"]:
            for func in ["load_cache", "store_cache", "compute_layer"]:
                name = func + "_" + stage
                costs = timers(name).costs
                print(f"{name:22s} (per-batch): {np.mean(costs):.6f} s")

    def generation_loop_overlap_single_batch(self):
        # Prologue
        for k in range(self.num_gpu_batches):
            self.load_weight(0, 0, k)
        self.sync()

        # Generate
        for i in range(self.execute_gen_len):
            timers("generate").start()
            self.update_attention_mask(i, 0)
            self.update_position_embedding(i, 0)

            for j in range(self.num_layers):
                self.load_weight(i, j+1, 0)
                self.load_cache(i, j+1, 0)
                self.load_hidden(i, j, 0)
                self.compute_layer(i, j, 0)
                self.store_cache(i, j-1, 0)
                self.store_hidden(i, j, 0)
                self.sync()
            timers("generate").stop()

            if self.task.stop and np.all(self.stopped):
                break

    def generation_loop_overlap_multi_batch(self):
        # Prologue
        for k in range(self.num_gpu_batches):
            self.load_weight(0, 0, k)
        self.load_hidden(0, 0, 0)
        self.sync()

        # Generate
        for i in range(self.execute_gen_len):
            timers("generate").start()
            for k in range(self.num_gpu_batches):
                self.update_attention_mask(i, k)
                self.update_position_embedding(i, k)
            for j in range(self.num_layers):
                for k in range(self.num_gpu_batches):
                    self.load_weight(i, j+1, k)
                    self.load_cache(i, j, k+1)
                    self.store_hidden(i, j, k-1)
                    self.load_hidden(i, j, k+1)
                    self.compute_layer(i, j, k)
                    self.store_cache(i, j, k-1)
                    self.sync()
            timers("generate").stop()

        # Epilogue
        self.store_hidden(
            self.execute_gen_len-1, self.num_layers-1, self.num_gpu_batches-1)

    def generation_loop_debug_single_batch(self):
        execute_num_batches = 20
        batch_ct = 0
        pbar = tqdm(total=execute_num_batches)
        timers("prefill").reset()
        timers("decoding_gpu_batch").reset()

        # Prologue
        for k in range(self.num_gpu_batches):
            self.load_weight(0, 0, k)
        self.sync()

        # Generate
        for i in range(self.execute_gen_len):
            if i == 0: timers("prefill").start()
            self.update_attention_mask(i, 0)
            self.update_position_embedding(i, k)
            for j in range(self.num_layers):
                if i > 0: timers("decoding_gpu_batch").start()
                self.load_weight(i, j+1, 0)
                self.load_cache(i, j+1, 0)
                self.load_hidden(i, j, 0)
                self.compute_layer(i, j, 0)
                self.store_cache(i, j-1, 0)
                self.store_hidden(i, j, 0)
                self.sync()

                if i > 0:
                    timers("decoding_gpu_batch").stop()
                    pbar.update(1)
                    batch_ct += 1
                if batch_ct >= execute_num_batches: break
            if batch_ct >= execute_num_batches: break
            if i == 0: timers("prefill").stop()

        # Convert "decoding_gpu_batch" timer to "generate" timer
        batch_cost = np.mean(timers("decoding_gpu_batch").costs[10:])
        for i in range(self.execute_gen_len):
            if i == 0:
                timers("generate").costs.append(timers("prefill").costs[0])
            else:
                timers("generate").costs.append(self.num_layers * batch_cost)

    def generation_loop_debug_multi_batch(self):
        execute_num_batches = 20
        batch_ct = 0
        pbar = tqdm(total=execute_num_batches)
        timers("prefill").reset()
        timers("decoding_gpu_batch").reset()

        # Prologue
        for k in range(self.num_gpu_batches):
            self.load_weight(0, 0, k)
        self.load_hidden(0, 0, 0)
        self.sync()

        # Generate
        for i in range(self.execute_gen_len):
            if i == 0: timers("prefill").start()
            for k in range(self.num_gpu_batches):
                self.update_attention_mask(i, k)
                self.update_position_embedding(i, k)
            for j in range(self.num_layers):
                if i > 0: timers("decoding_gpu_batch").start()
                for k in range(self.num_gpu_batches):
                    self.load_weight(i, j+1, k)
                    self.load_cache(i, j, k+1)
                    self.store_hidden(i, j, k-1)
                    self.load_hidden(i, j, k+1)
                    self.compute_layer(i, j, k)
                    self.store_cache(i, j, k-1)
                    self.sync()

                if i > 0:
                    timers("decoding_gpu_batch").stop()
                    pbar.update(1)
                    batch_ct += 1
                if batch_ct >= execute_num_batches: break
            if batch_ct >= execute_num_batches: break
            if i == 0: timers("prefill").stop()

        # Convert "decoding_gpu_batch" timer to "generate" timer
        batch_cost = np.mean(timers("decoding_gpu_batch").costs[10:])
        for i in range(self.execute_gen_len):
            if i == 0:
                timers("generate").costs.append(timers("prefill").costs[0])
            else:
                timers("generate").costs.append(self.num_layers * batch_cost)

    def __del__(self):
        self.delete_all_weights()


# ------------------------------
# 推理专用模型类
# ------------------------------
class Qwen2InferenceModel(nn.Module):
    def __init__(self, config: dict, device: torch.device):
        super().__init__()
        self.config = config
        self.vocab_size = config["vocab_size"]
        self.hidden_size = config["hidden_size"]
        self.max_seq_len = config["max_position_embeddings"]
        self.bos_token_id = config["bos_token_id"]
        self.eos_token_id = config["eos_token_id"]

        # 模型组件（仅推理必需）
        self.embed_tokens = nn.Embedding(self.vocab_size, self.hidden_size, device=device)
        self.layers = nn.ModuleList([
            Qwen2Block(
                hidden_size=config["hidden_size"],
                num_attention_heads=config["num_attention_heads"],
                num_key_value_heads=config["num_key_value_heads"],
                intermediate_size=config["intermediate_size"],
                rms_norm_eps=config["rms_norm_eps"],
                device=device,
                layer_idx=i
            ) for i in range(config["num_hidden_layers"])
        ])
        self.norm = RMSNorm(self.hidden_size, eps=config["rms_norm_eps"])
        self.lm_head = nn.Linear(self.hidden_size, self.vocab_size, device=device, bias=False)
        self.lm_head.weight = self.embed_tokens.weight  # 共享词嵌入权重

        # 预计算RoPE频率（推理时直接复用）
        # self.register_buffer(
        #     "freqs_complex",
        #     precompute_rope_freqs(
        #         dim=self.hidden_size // config["num_attention_heads"],
        #         max_seq_len=self.max_seq_len,
        #         theta=config["rope_theta"],
        #         device=device
        #     ),
        #     persistent=False
        # )

        self.position_embedding = precompute_rope_freqs(
            dim=self.hidden_size // config["num_attention_heads"],
            max_seq_len=self.max_seq_len,
            theta=config["rope_theta"],
            device=device
        )

        # 推理模式：关闭dropout等训练相关操作
        self.eval()

    def forward(self, input_ids: torch.Tensor, output_hiddens=False):
        """单次前向传播，返回logits"""
        batch_size, seq_len = input_ids.shape
        if seq_len > self.max_seq_len:
            raise ValueError(f"输入长度({seq_len})超过最大序列长度({self.max_seq_len})")

        if output_hiddens:
            output_hidden_buffer = { "layers":[]}
        # 词嵌入
        hidden_states = self.embed_tokens(input_ids)

        if output_hiddens:
            output_hidden_buffer["embed_tokens"] = hidden_states

        # 应用所有Transformer层
        for layer in self.layers:
            cos, sin = self.position_embedding
            hidden_states = layer(hidden_states, (cos[:seq_len], sin[:seq_len]))

            if output_hiddens:
                output_hidden_buffer["layers"].append(hidden_states)
            

        # 输出logits
        hidden_states = self.norm(hidden_states)

        if output_hiddens:
            output_hidden_buffer["logits"] = hidden_states

        if output_hiddens:
            return self.lm_head(hidden_states), output_hidden_buffer
 
        return self.lm_head(hidden_states)

    @torch.no_grad()  # 推理时禁用梯度计算
    def generate(self, prompt: str, tokenizer, max_new_tokens: int = 100, temperature: float = 1.0) -> str:
        """
        文本生成函数（贪心解码）
        :param prompt: 输入文本
        :param tokenizer: 分词器（需与模型匹配，如Qwen2Tokenizer）
        :param max_new_tokens: 最大生成token数
        :param temperature: 温度参数（控制随机性，0=贪心）
        :return: 生成的文本
        """
        # 分词并添加BOS
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.device)  # 转移到模型设备
        batch_size, current_len = input_ids.shape

        for _ in range(max_new_tokens):
            # 前向传播获取logits（仅需最后一个token的logits）
            logits = self.forward(input_ids)[:, -1, :]  # [batch, vocab_size]

            # 温度调整
            if temperature > 0:
                logits = logits / temperature
                probs = F.softmax(logits, dim=-1)
                next_token_id = torch.multinomial(probs, num_samples=1)  # 采样
            else:
                next_token_id = torch.argmax(logits, dim=-1, keepdim=True)  # 贪心

            # 拼接新token
            input_ids = torch.cat([input_ids, next_token_id], dim=-1)
            current_len += 1

            # 检查是否生成EOS
            if next_token_id.item() == self.eos_token_id:
                break

            # 防止超出最大长度
            if current_len >= self.max_seq_len:
                break

        # 解码为文本
        return tokenizer.decode(input_ids[0], skip_special_tokens=True)

    @property
    def device(self) -> torch.device:
        return self.embed_tokens.weight.device

    def load_from_safetensors(self, weight_path = "/home/llmserver/.cache/huggingface/hub/models--Qwen--Qwen2-0.5B/snapshots/91d2aff3f957f99e4c74c962f2f408dcc88a18d8/model.safetensors"):
        model_weightnames = self.state_dict().keys()
        loaded_weights = load_file(weight_path, device="cuda")
        converted_weights = {}
        for name, weight in loaded_weights.items():
            name = name.replace("model.", "")
            name = name.replace("self_attn", "attention")
            name = name.replace("mlp", "feed_forward")
            
            assert name in model_weightnames, f" weight not in model: {name}"

            converted_weights[name] = weight

        converted_weights["lm_head.weight"] = converted_weights["embed_tokens.weight"]
        self.load_state_dict(converted_weights)


# ------------------------------
# 推理示例
# ------------------------------
# def generate(inputs: str|list[str]):
def generate():
    # 2, device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    model = Qwen2InferenceModel(qwen2_config, device)

    # model 
    model_path = "/home/llmserver/.cache/huggingface/hub/models--Qwen--Qwen2-0.5B/snapshots/91d2aff3f957f99e4c74c962f2f408dcc88a18d8/model.safetensors"
    model.load_from_safetensors(model_path)

    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B")
    # 确保分词器的特殊token与模型一致
    tokenizer.bos_token_id = qwen2_config["bos_token_id"]
    tokenizer.eos_token_id = qwen2_config["eos_token_id"]

    # casual inputs
    prompt = "Paris is the capital city of"
    generated_text = model.generate(
        prompt=prompt,
        tokenizer=tokenizer,
        max_new_tokens=50,
        temperature=0.7
    )
    print(f" >>> prompt: {prompt}")
    print(f" >>> generated: {generated_text}")




def test_my_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # casual input_ids
    input_ids = torch.tensor([[100, 200, 300]], device="cuda")

    # my model
    model = Qwen2InferenceModel(qwen2_config, device)
    model.load_from_safetensors()
    outputs, hiddens = model(input_ids, output_hiddens=True)
    
    
    ref_model = Qwen2ForCausalLM.from_pretrained("Qwen/Qwen2-0.5B", device_map="auto")
    
    with torch.no_grad():
        outputs = ref_model(input_ids, output_hidden_states=True)
    
    hidden_embed = outputs.hidden_states[0]
    hidden_layer0 = outputs.hidden_states[1]

    my_embed = hiddens["embed_tokens"]
    my_layer0 = hiddens["layers"][0]

    print(hidden_layer0)
    
    print("Embedding误差:", torch.norm(hidden_embed - my_embed).item())
    print("layer0误差:", torch.norm(hidden_layer0 - my_layer0).item())

    # TODO
    # test_rope(my_model=model, ref_model=ref_model)


def run_ref_model():
    model_name = "Qwen/Qwen2-0.5B"
    model = Qwen2ForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    prompt = "Paris is the capital city of"

    input_ids = tokenizer(prompt).input_ids
    input_ids = torch.tensor([input_ids, ])
    output_ids = model(input_ids)
    output_token = output_ids[:, -1:].argmax()

    # output_seq = output_ids
    output_seq = tokenizer.batch_decode(output_token)

    print(output_seq)

def run_flexllmgen(args):
    print(f"<run_flexllmgen>: args.model: {args.model}")
    if args.model == "facebook/galactica-30b":
        tokenizer = AutoTokenizer.from_pretrained("facebook/galactica-30b", padding_side="left")
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side="left")
        # print(f" >>> tokenizer use model: {args.model}")
    prompt_len, gen_len, cut_gen_len = args.prompt_len, args.gen_len, args.cut_gen_len
    num_prompts = args.num_gpu_batches * args.gpu_batch_size

    # Task and policy
    if args.input_file is not None:  
        inputs = get_file_inputs(args.input_file)
        input_in_tokens = tokenizer(inputs, padding="longest").input_ids
    else:
        input_in_tokens = get_test_inputs(args.prompt_len, num_prompts, tokenizer)

    warmup_inputs = get_test_inputs(32, num_prompts, tokenizer)
    
    # inputs = get_test_inputs(prompt_len, num_prompts, tokenizer)

    gpu = TorchDevice("cuda:0")
    cpu = TorchDevice("cpu")
    disk = TorchDisk(args.offload_dir)
    env = ExecutionEnv(gpu=gpu, cpu=cpu, disk=disk, mixed=TorchMixedDevice([gpu, cpu, disk]))

    policy = Policy(args.gpu_batch_size, args.num_gpu_batches,
                    args.percent[0], args.percent[1],
                    args.percent[2], args.percent[3],
                    args.percent[4], args.percent[5],
                    args.overlap, args.sep_layer, args.pin_weight,
                    args.cpu_cache_compute, args.attn_sparsity,
                    args.compress_weight,
                    CompressionConfig(num_bits=4, group_size=64,
                                      group_dim=0, symmetric=False),
                    args.compress_cache,
                    CompressionConfig(num_bits=4, group_size=64,
                                      group_dim=2, symmetric=False))
    assert not (args.compress_cache and args.attn_sparsity < 1.0), "Not implemented"

    qwen_config = get_qwen_config(args.model)
    cache_size = qwen_config.cache_bytes(num_prompts, prompt_len + gen_len)
    hidden_size = qwen_config.hidden_bytes(num_prompts, prompt_len + gen_len)
    print(f"model size: {qwen_config.model_bytes()/GB:.3f} GB, "
          f"cache size: {cache_size/GB:.3f} GB, "
          f"hidden size (prefill): {hidden_size/GB:.3f} GB")

    print("init weight...")
    model = QwenLM(qwen_config, env, args.path, policy)

    try:
        print("warmup - generate")
        print(f" >>> warmup inputs in type: {type(warmup_inputs)}")
        output_ids = model.generate(
            warmup_inputs, max_new_tokens=1, verbose=args.verbose
        )
        print("benchmark - generate")
        timers("generate").reset()
        output_ids = model.generate(
            input_in_tokens, max_new_tokens=args.gen_len,
            debug_mode=args.debug_mode, cut_gen_len=cut_gen_len, verbose=args.verbose)
        costs = timers("generate").costs
    finally:
        env.close_copy_threads()

    # Log output
    prefill_latency = costs[0]
    prefill_throughput = num_prompts * prompt_len / prefill_latency
    if cut_gen_len:  # project latency of cut_gen_len to gen_len
        decode_latency = project_decode_latency(costs, prompt_len, gen_len)
    else:
        decode_latency = sum(costs[1:])
    decode_throughput = num_prompts * (gen_len - 1) / max(decode_latency, 1e-10)
    num_generated_tokens = num_prompts * gen_len
    total_latency = prefill_latency + decode_latency
    total_throughput = num_generated_tokens / total_latency
    _, gpu_peak_mem = gpu.mem_stats()
    _, cpu_peak_mem = cpu.mem_stats()

    if DUMMY_WEIGHT not in args.path:
        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        show_str = "Outputs:\n" + 70 * '-' + "\n"
        for i in [0, len(outputs)-1]:
            show_str += f"{i}: {outputs[i]}\n"
            show_str += "-" * 70 + "\n"
        if args.verbose >= 2:
            print(show_str)

    gpu.print_stats()
    cpu.print_stats()
    projected = bool(args.debug_mode or cut_gen_len)

    if args.log_file == "auto":
        filename = get_filename(args) + ".log"
    else:
        filename = args.log_file

    log_str = write_benchmark_log(filename,
        qwen_config.model_bytes(), cache_size, hidden_size,
        gpu_peak_mem, projected, prefill_latency, prefill_throughput,
        decode_latency, decode_throughput, total_latency, total_throughput)
    if args.verbose >= 1:
        print(log_str)



def add_parser_arguments(parser):
    parser.add_argument("--model", type=str, default="Qwen/Qwen2-0.5B",
        help="The model name.")
    parser.add_argument("--path", type=str, default="~/qwen_weights",
        help="The path to the model weights. If there are no cached weights, "
             "FlexLLMGen will automatically download them from HuggingFace.")
    parser.add_argument("--offload-dir", type=str, default="flexllmgen_offload_dir",
        help="The directory to offload tensors. ")
    parser.add_argument("--prompt-len", type=int, default=512)
    parser.add_argument("--gen-len", type=int, default=32)
    parser.add_argument("--cut-gen-len", type=int,
        help="Cut generation length for fast debugging.")
    parser.add_argument("--debug-mode", type=str,
        choices=["fewer_batch", "breakdown"])
    parser.add_argument("--gpu-batch-size", type=int, default=4)
    parser.add_argument("--num-gpu-batches", type=int, default=1)
    parser.add_argument("--percent", nargs="+", type=int,
        default=[100, 0, 100, 0, 100, 0],
        help="Six numbers. They are "
         "the percentage of weight on GPU, "
         "the percentage of weight on CPU, "
         "the percentage of attention cache on GPU, "
         "the percentage of attention cache on CPU, "
         "the percentage of activations on GPU, "
         "the percentage of activations on CPU")
    parser.add_argument("--sep-layer", type=str2bool, nargs='?',
        const=True, default=True)
    parser.add_argument("--pin-weight", type=str2bool, nargs="?",
        const=True, default=True)
    parser.add_argument("--cpu-cache-compute", action="store_true")
    parser.add_argument("--attn-sparsity", type=float, default=1.0)
    parser.add_argument("--compress-weight", action="store_true",
        help="Whether to compress weight.")
    parser.add_argument("--compress-cache", action="store_true",
        help="Whether to compress cache.")


    parser.add_argument("--log-file", type=str, default="auto")
    parser.add_argument("--no-log", action="store_true")
    parser.add_argument("--verbose", type=int, default=2)

    parser.add_argument("--overlap", type=str2bool, nargs='?',
        const=True, default=True)

    # hacked 
    parser.add_argument("--input-file", type=str, 
        help="input file containing prompts, "
        ".txt contains 1 prompt, "
        ".json for many prompts")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_parser_arguments(parser)
    args = parser.parse_args()

    assert len(args.percent) == 6

    run_flexllmgen(args)
