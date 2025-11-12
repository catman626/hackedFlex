import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple
from transformers import Qwen2ForCausalLM, AutoTokenizer
from safetensors.torch import load_file
import os

from flexllmgen.pytorch_backend import (TorchDevice, TorchDisk, TorchLink,
    TorchMixedDevice, DeviceType, general_copy, fix_recursive_import)
from flexllmgen.utils import (Task, ExecutionEnv, GB, T, ValueHolder,
    array_1d, array_2d, array_3d, str2bool, project_decode_latency,
    torch_mem_stats, torch_dtype_to_np_dtype, write_benchmark_log,
    read_benchmark_log)

from flex_opt import init_weight_list

# 1. 定义模型配置（与Qwen2-0.5B完全匹配）
qwen2_config = {
    "vocab_size": 151936,
    "hidden_size": 896,
    "num_hidden_layers": 24,
    "num_attention_heads": 14,
    "num_key_value_heads": 2,
    "intermediate_size": 4864,
    "max_position_embeddings": 131072,
    "rms_norm_eps": 1e-6,
    "rope_theta": 1000000.0,
    "bos_token_id": 151643,
    "eos_token_id": 151643,
}


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


def precompute_rope_freqs(dim: int, max_seq_len: int, theta: float, device: torch.device) :
    
    inv_freq = 1.0 / (theta** (torch.arange(0, dim, 2, device=device) / dim))
    seq = torch.arange(max_seq_len, device=device, dtype=inv_freq.dtype)
    
    concat_inv_freq = torch.concat([inv_freq, inv_freq], dim=-1)
    freqs = torch.outer(seq, concat_inv_freq)
    return torch.cos(freqs), torch.sin(freqs)
    

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rope(x: torch.Tensor, position_embedding: tuple[torch.Tensor, torch.Tensor], unsqueeze_dim = 1 ) -> torch.Tensor:
    # sin, cos in shape (sed_len , hidden_dim)
    # x might in shape (bs, seq_len, head, hidden_dim) : unsqueeze_dim = 1
    #               or (bs, head, seq, hidden_dim):      unsqueeze_dim = 0

    cos, sin = position_embedding
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    x_embed  = (x * cos) + (rotate_half(x) * sin)
    return x_embed

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
        h, dtype = (self.config.input_dim, self.config.dtype)
        path = os.path.join(os.path.join(path, f"decoder.layers.{self.layer_id}.self_attn"))
        weight_specs = [
            # w_q
            ((h, h), dtype, path + ".q_proj.weight"),
            # b_q
            ((h,), dtype, path + ".q_proj.bias"),
            # w_k
            ((h, h), dtype, path + ".k_proj.weight"),
            # b_k
            ((h,), dtype, path + ".k_proj.bias"),
            # w_v
            ((h, h), dtype, path + ".v_proj.weight"),
            # b_v
            ((h,), dtype, path + ".v_proj.bias"),
            # w_out
            ((h, h), dtype, path + ".out_proj.weight"),
            # b_out, o_proj have no bias in qwen2
            # ((h,), dtype, path + ".out_proj.bias"),
            # w_ln
            ((h,), dtype, path + "input_layernorm.weight"),
            # # b_ln, rms has no bias
            # ((h,), dtype, path + "_layer_norm.bias"),
        ]
        weights = init_weight_list(weight_specs, self.policy, self.env)
        weight_home.store(weights)

    def load_weight(self, weight_home, weight_read_buf, k):
        w_q, b_q, w_k, b_k, w_v, b_v, w_out, b_out, w_ln = weight_home.val
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
        return (batch_size, seq_len, self.config.input_dim), self.config.dtype

    def forward(self, hidden, cache_read_buf, weight_read_buf, attention_mask,
                cache_write_buf, i, k):
        n_head = self.config.n_head

        donate = [False] * 14   # keep 14, but idx for b_out will not be used
        # 0:hidden, 1:mask, 2-11:weight, 12:kcache, 13:vcache
        
        h, donate[0] = hidden.val, True

        if k == self.policy.num_gpu_batches - 1:
            # Clear the weight_read_buf if it is the last gpu batch
            # ((w_q, donate[2]), (b_q, donate[3]), (w_k, donate[4]), (b_k, donate[5]),
            #  (w_v, donate[6]), (b_v, donate[7]), (w_out, donate[8]), (b_out, donate[9]),
            #  (w_ln, donate[10]), (b_ln, donate[11])) = weight_read_buf.pop()
            ((w_q, donate[2]), (b_q, donate[3]), (w_k, donate[4]), (b_k, donate[5]),
             (w_v, donate[6]), (b_v, donate[7]), (w_out, donate[8]),
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
            # h, new_k_cache, new_v_cache = self.compute.mha(h, mask, w_q, b_q,
            #     w_k, b_k, w_v, b_v, w_out, b_out, w_ln, b_ln, n_head, donate,
            #     self.policy.compress_cache, self.policy.comp_cache_config)
            h, new_k_cache, new_v_cache = self.compute.gqa(h, mask, w_q, b_q,
                w_k, b_k, w_v, b_v, w_out, w_ln, n_head, donate,
                self.policy.compress_cache, self.policy.comp_cache_config)
            cache_write_buf.store((new_k_cache, new_v_cache))
        else:  # decoding
            mask, donate[1] = attention_mask.val.smart_copy(self.attention_compute)
            (k_cache, donate[12]), (v_cache, donate[13]) = cache_read_buf.pop()
            # h, new_k_cache, new_v_cache = self.compute.mha_gen(h, mask, w_q,
            #     b_q, w_k, b_k, w_v, b_v, w_out, b_out, w_ln, b_ln, n_head,
            #     k_cache, v_cache, donate, self.policy.attn_sparsity,
            #     self.policy.compress_cache, self.policy.comp_cache_config)
            h, new_k_cache, new_v_cache = self.compute.gqa_gen(h, mask, w_q,
                b_q, w_k, b_k, w_v, b_v, w_out, w_ln, n_head,
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
        h, dtype = (self.config.input_dim, self.config.dtype)
        path = os.path.join(os.path.join(path, f"decoder.layers.{self.layer_id}."))
        
        intermediated_size = self.config.intermediate_size
        hidden_size = self.config.hidden_size
        dtype = self.config.dtype
        
        weight_specs = [
            # wg
            ((intermediated_size, hidden_size), dtype, path + "gate_proj.weight"),
            # wup
            ((intermediated_size, hidden_size), dtype, path + "up_proj.weight"),
            # wdown
            ((hidden_size, intermediated_size), dtype, path + "down_proj.weight"),
            # w_ln
            ((hidden_size, ), dtype, path + "post_attention_layernorm")
        ]
        weights = init_weight_list(weight_specs, self.policy, self.env)
        weight_home.store(weights)

    def load_weight(self, weight_home, weight_read_buf, k):
        w_gate, w_up, w_down = weight_home.val
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
                w_down.smart_copy(dst2) )
            )

    def init_cache_one_gpu_batch(self, cache_home):
        pass  # do nothing

    def load_cache(self, cache_home, cache_read_buf, i):
        pass  # do nothing

    def store_cache(self, cache_home, cache_write_buf, i):
        pass  # do nothing

    def input_act_shape_and_dtype(self, batch_size, seq_len):
        return (batch_size, seq_len, self.config.input_dim), self.config.dtype

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

if __name__ == "__main__":
    test_my_model()
    
    # run_ref_model()

    # generate()