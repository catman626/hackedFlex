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
import tqdm

from flexllmgen.pytorch_backend import (TorchDevice, TorchDisk, TorchLink, TorchTensor, dump_hidden, 
    TorchMixedDevice, DeviceType, general_copy, fix_recursive_import)
from flexllmgen.utils import (Task, ExecutionEnv, GB, T, ValueHolder,
    array_1d, array_2d, array_3d, str2bool, project_decode_latency,
    torch_mem_stats, torch_dtype_to_np_dtype, write_benchmark_log,
    read_benchmark_log, expand_block_idx, num_block, tail_length)
from flexllmgen.qwen2_config import QwenConfig, get_qwen_config, convert_qwen_weights
from flexllmgen.compression import CompressionConfig
from flexllmgen.sparse import SparseConfig

from flexllmgen.flex_opt import init_weight_list, get_compact_test_inputs, get_file_inputs, get_test_inputs, get_filename

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
    sparse_config: SparseConfig

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

class TransformerComponent:
    def __init__(self, config, env, policy, layer_id) -> None:
        self.config = config
        self.env = env
        self.policy = policy
        self.layer_id = layer_id
        self.task:Task|None = None
    
    def set_task(self, task)->None:
        self.task = task

    def _context_len(self, i, after_proj=True) -> int:
        return self.task.prompt_len + i

    def _sparse_stage(self, i):
        return self._context_len(i) >= 2048+self.policy.sparse_config.block_size

    def store_cache(self, cache_home, cache_write_buf, i):
        """ this part only use cache, no store """
        pass 

    def delete_cache(self, cache_home):
        pass

    def init_cache_one_gpu_batch(self, cache_home):
        pass  # do nothing

    def load_cache(self, cache_home, cache_read_buf, i):
        pass  # do nothing


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

        self.task:Task|None = None

    def set_task(self, task)->None:
        self.task = task

    def init_weight(self, weight_home, path):
        h, dtype = (self.config.hidden_size, self.config.dtype)
        head_dim = h // self.config.num_attention_heads
        kv_dim = head_dim * self.config.num_key_value_heads
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
            # w_ln
            ((h,), dtype, layer_path + ".input_layernorm.weight"),
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

    def calculate_selected_token_ids(self, block_idx):
        """ return (b, qhead, topk) """
        block_idx = block_idx.val   # (b, head, topk)
        block_size = self.policy.sparse_config.block_size
        b, n_qhead, topk = block_idx.shape
        src_s = self.task.prompt_len + i
        tail_len = src_s % block_size
        if tail_len > 0:
            st = (src_s // block_size) * block_size
            tails = torch.arange(st, st+tail_len)   # (ntail,)
            
        expanded_block = expand_block_idx(block_idx, self.policy.sparse_config.block_size)
        selected_idx = torch.concat(
                [expanded_block, tails.expand(b, n_qhead, tail_len)],  
                dim=-1 )

        return selected_idx

    def load_summary(self, cache_home, cache_read_buf, i):
        k_cache, v_cache, block_cache, idx_cache = cache_home.val
        cache_read_buf.store((
            block_cache.smart_copy(dst),
        )) 

    def load_cache(self, cache_home, cache_read_buf, i):
        # self-load-cache
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
            # indices = (slice(0, self.task.prompt_len + i),
            #            slice(0, k_home.shape[1]))
            indices = (slice(None), slice(None), slice(0, self.task.prompt_len + i))

            if self.policy.sparse_config.mode == "dense":
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

            if self.policy.sparse_config.mode == "dense" :
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
            assert self.policy.sparse_config.mode == "dense"
        else:
            raise ValueError(f"Invalid path: {path}")

    def store_cache(self, cache_home, cache_write_buf, i):
        # self-store-cache
        # shape: (s, b * n_head, head_dim)
        k_home, v_home = cache_home.val
        k_new, v_new = cache_write_buf.pop()

        if i == self.task.gen_len - 1:  # last token, no need to store cache
            return

        if i == 0:  # prefill
            indices = (slice(None), slice(None), slice(0, k_new.shape[2]))
            # indices = (slice(0, k_new.shape[0]),
            #            slice(0, k_new.shape[1]))
        else:  # decoding
            pos = self.task.prompt_len + i
            # indices = (slice(pos - k_new.shape[0], pos),
            #            slice(0, k_new.shape[1]))
            indices = (slice(None), slice(None), slice(pos - k_new.shape[2], pos))
        

        general_copy(k_home, indices, k_new, None)
        general_copy(v_home, indices, v_new, None)

    def input_act_shape_and_dtype(self, batch_size, seq_len):
        return (batch_size, seq_len, self.config.hidden_size), self.config.dtype
    
    def forward_qkvproj(self, hidden, cache_read_buf, weight_read_buf, attention_mask,
                position_embeddings, cache_write_buf, i, k):
        donate = [False] * (1+1+1+7)
        # 0:hidden, 1:mask, 2:pos_embd, 345678 qkv_wb, 9 w_ln
        
        h, donate[0] = hidden.val, True
        
        n_qhead = self.config.num_attention_heads
        n_kvhead = self.config.num_key_value_heads

        if k == self.policy.num_gpu_batches - 1:
            (( w_q,  donate[3]), (b_q, donate[4]), 
            (  w_k, donate[5]), (b_k, donate[6]),
            (  w_v, donate[7]), (b_v, donate[8]),
            (  w_ln, donate[9])) = weight_read_buf.val.pop()
        else:
            # (w, _),  = weight_read_buf.val
            ( (w_q, _), (b_q, _), (w_k, _), (b_k, _), (w_v, _), (b_v, _), (w_ln, _) ) = weight_read_buf.val

        position_embed, donate[2] = position_embeddings.val.smart_copy(self.compute)
        q, k, v= self.compute.qkv_proj(h, w_q, b_q, w_k, b_k, w_v, b_v, position_embeddings)
        K_summary, v_summary = self.compute.kv_summary(k, v)
        cache_write_buf.store((k, v, k_summary, v_summary))

        hidden.val = q


    def forward(self, hidden, cache_read_buf, weight_read_buf, attention_mask,
                position_embeddings, cache_write_buf, i, k):
        # forward-attn
        n_qhead = self.config.num_attention_heads
        n_kvhead = self.config.num_key_value_heads

        donate = [False] * 14   # keep 14, but idx for b_out will not be used
        # 0:hidden, 1:mask, 2-11:weight, 12:kcache, 13:vcache
        # my: 0: hidden, 1:mask, 2:embedding, 345678 qkv w&b, 9:out, 10:ln, 11,12 kvcache
        
        h, donate[0] = hidden.val, True

        if k == self.policy.num_gpu_batches - 1:
            ((w_q, donate[3]), (b_q, donate[4]), (w_k, donate[5]), (b_k, donate[6]),
             (w_v, donate[7]), (b_v, donate[8]), (w_out, donate[9]),
             (w_ln, donate[10])) = weight_read_buf.pop()
        else:
            ((w_q, _), (b_q, _), (w_k, _), (b_k, _),
             (w_v, _), (b_v, _), (w_out, _), 
             (w_ln, _)) = weight_read_buf.val

        if i == 0:  # prefill
            mask, donate[1] = attention_mask.val.smart_copy(self.compute)
            position_embed, donate[2] = position_embeddings.val.smart_copy(self.compute)
            h, new_k_cache, new_v_cache = self.compute.gqa(h, mask, position_embed, w_q, b_q,
                w_k, b_k, w_v, b_v, w_out, w_ln, (n_qhead, n_kvhead), donate,
                self.policy.compress_cache, self.policy.comp_cache_config, self.layer_id)
            cache_write_buf.store((new_k_cache, new_v_cache))

            dump_hidden(h.data,  "attn-output", self.layer_id)

        else:  # decoding
            mask, donate[1] = attention_mask.val.smart_copy(self.attention_compute)
            position_embed, donate[2] = position_embeddings.val.smart_copy(self.compute)
            (k_cache, donate[12]), (v_cache, donate[13]) = cache_read_buf.pop()
            h, new_k_cache, new_v_cache = self.compute.gqa_gen(
                h, 
                mask, position_embed, 
                w_q, b_q, w_k, b_k, w_v, b_v, w_out, w_ln, 
                (n_qhead, n_kvhead),
                k_cache, v_cache, 
                donate, 
                self.policy.sparse_config, self.policy.compress_cache, self.policy.comp_cache_config,
                self.layer_id)
            cache_write_buf.store((new_k_cache, new_v_cache))

        hidden.val = h

class QKVProj(TransformerComponent):
    def __init__(self, config, env, policy, layer_id):
        self.config : QwenConfig = config
        self.env = env
        self.layer_id = layer_id
        self.policy = policy
        self.compute:TorchDevice= self.env.gpu
        self.attention_compute = (self.env.cpu if self.policy.cpu_cache_compute
            else self.env.gpu)
        self.weight_load_dst = (self.compute.compressed_device if policy.compress_weight
            else self.compute)
        self.task:Task|None= None

    def init_weight(self, weight_home, path):
        h, dtype = (self.config.hidden_size, self.config.dtype)
        head_dim = h // self.config.num_attention_heads
        kv_dim = head_dim * self.config.num_key_value_heads
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
            # w_ln
            ((h,), dtype, layer_path + ".input_layernorm.weight"),
        ]
        weights = init_weight_list(weight_specs, self.policy, self.env)
        weight_home.store(weights)


    def load_weight(self, weight_home, weight_read_buf, k):
        if k == 0:
            w_q, b_q, w_k, b_k, w_v, b_v, w_ln = weight_home.val
            dst1 = self.weight_load_dst
            dst2 = self.compute
            weight_read_buf.store(
                ( w_q.smart_copy(dst1), b_q.smart_copy(dst2),
                w_k.smart_copy(dst1), b_k.smart_copy(dst2),
                w_v.smart_copy(dst1), b_v.smart_copy(dst2),
                w_ln.smart_copy(dst2) )
            )

    def delete_cache(self, cache_home):
        k_home, v_home, k_summary_home, (idx_home, idx_num) = cache_home.val
        k_home.delete()
        v_home.delete()
        k_summary_home.delete()
        idx_home.delete()
        idx_num.clear()
        
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

        cache = device.init_cache_one_gpu_batch_qwen(self.config, self.task, self.policy)
        cache_home.store(cache)


    def load_summary(self, cache_home, cache_read_buf, i):
        k_cache, v_cache, block_cache, idx_cache = cache_home.val
        cache_read_buf.store(
            ( block_cache.smart_copy(self.attention_compute), )
        ) 

    def load_cache(self, cache_home, cache_read_buf, i):
        """ proj-load-cache"""
        if i == 0:
            # prefill
            return
        
        # print(f" >>> step {i} layer {self.layer_id} proj load cache: {id(cache_home)}")
        k_home, v_home, summary_home, idx_home = cache_home.val

        context_len = self._context_len(i)
        block_size = self.policy.sparse_config.block_size

        dst = self.attention_compute
        place_holder = (None, True)

        if not self._sparse_stage(i):
            indices = (slice(None), slice(None), slice(0, context_len))
            cache_read_buf.store(
                (   k_home.smart_copy(dst, indices), 
                    v_home.smart_copy(dst, indices), 
                    place_holder, 
                    place_holder)
            )
        else:
            if i == 0 or not self._sparse_stage(i-1):
                # load all kv to compute summary
                cache_read_buf.store( 
                    (   k_home.smart_copy(dst, (slice(None), slice(None), slice(0, context_len-1))), 
                        place_holder, 
                        place_holder, 
                        idx_home ))
            else:
                tail_len = tail_length(context_len, block_size)
                n_blk = num_block(context_len, block_size)

                summary_idx = (slice(None), slice(None), slice(0, n_blk))
                if context_len % block_size == 0:
                    tail_k_idx = (  slice(None), slice(None), 
                                    slice(context_len-tail_len, context_len-1)    )
                    cache_read_buf.store(
                        (   k_home.smart_copy(dst, tail_k_idx),
                            place_holder,
                            summary_home.smart_copy(dst, summary_idx),
                            idx_home        )
                    )
                else:
                    cache_read_buf.store(
                        (   place_holder, place_holder, 
                            summary_home.smart_copy(dst, summary_idx),
                            idx_home    )
                    )

    def store_cache(self, cache_home, cache_write_buf, i):
        """proj-store-cache"""
        # shape: (s, b * n_head, head_dim)
        # current impl load&store whole summary
        # print(f" >>> step-{i} layer-{self.layer_id} proj-store-cache: {id(cache_home)}")
        k_home, v_home, k_summary_home, idx= cache_home.val
        k_new, v_new, k_summary = cache_write_buf.pop()
        
        # if i == self.task.gen_len - 1:  # last token, no need to store cache
        #     return

        # store k,v
        if i == 0:  # prefill
            # knew: (b, h, s, d)
            indices = (slice(None), slice(None), slice(0, k_new.shape[2]),slice(None))
        else:  # decoding
            pos = self.task.prompt_len + i
            indices = (slice(None), slice(None), slice(pos - k_new.shape[0], pos), slice(None))
        general_copy(k_home, indices, k_new, None)
        general_copy(v_home, indices, v_new, None)

        # store summary
        if k_summary is not None:
            context_len = self._context_len(i)
            
            block_size = self.policy.sparse_config.block_size
            num_blk = context_len // block_size - 1

            # print(f" >>> context_len: {self._context_len(i)}")
            # print(f" >>> summary shape: {k_summary.shape}") 
            if i == 0 or not self._sparse_stage(i-1):
                assert num_blk == k_summary.shape[2]
                dst_idx = slice(0, num_blk)
            else:
                dst_idx = slice(num_blk-1, num_blk)
            general_copy(k_summary_home, (  slice(None),
                                            slice(None), 
                                            dst_idx ), 
                         k_summary, 
                         None)
        # store idx
        

    def forward(self, hidden, cache_read_buf, weight_read_buf, attention_mask,
                position_embeddings, cache_write_buf, i, k):
        """ proj-forward """
        # print(f" >>> step-{i},layer-{self.layer_id},batch-{k} proj-forward")
        donate = [False] * (1+1+1+7+2)
        # 0:hidden, 1:mask, 2:pos_embd, 345678 qkv_wb, 9 w_ln, 10-11 tail_kv
        
        h, donate[0] = hidden.val, True
        
        n_qhead = self.config.num_attention_heads
        n_kvhead = self.config.num_key_value_heads

        if k == self.policy.num_gpu_batches - 1:
            (( w_q,  donate[3]), (b_q, donate[4]), 
            (  w_k, donate[5]), (b_k, donate[6]),
            (  w_v, donate[7]), (b_v, donate[8]),
            (  w_ln, donate[9])) = weight_read_buf.pop()
        else:
            (   (w_q, _), (b_q, _), 
                (w_k, _), (b_k, _), 
                (w_v, _), (b_v, _), 
                (w_ln, _) ) = weight_read_buf.val

        position_embed, donate[2] = position_embeddings.val.smart_copy(self.compute)
        
        # TODO handle attn-mask
        context_len = self._context_len(i)
        block_size = self.policy.sparse_config.block_size

        if i == 0:
            q, k, v, k_summary = self.compute.qkv_proj_prefill(h, 
                                                               None, position_embed, 
                                                               w_q, b_q, w_k, b_k, w_v, b_v, w_ln, 
                                                               (n_qhead, n_kvhead), 
                                                               enable_sparse=self._sparse_stage(i), 
                                                               block_size=block_size)
            cache_write_buf.store((k, v, k_summary))
            hidden.val = q
        else:
            (tail_k, donate[2]), (tail_v, donate[2]), \
            (summary, donate[2]), (idx_home, idx_cnt)= cache_read_buf.pop()
            q, k, v, new_summary, idx = self.compute.qkv_proj_decode(
                h, None, position_embed, 
                w_q, b_q, w_k, b_k, w_v, b_v, w_ln,
                tail_k, tail_v, summary, (n_qhead, n_kvhead), context_len, 
                enable_sparse=self._sparse_stage(i),block_size=block_size)
            
            hidden.val = (q, k, v)
            
            assert not (self._sparse_stage(i) and idx is None)
            if idx is not None:
            # idxhome:(b, head, k) 
                window_size = idx.data.shape[-1]
                n_kvhead = self.config.num_attention_heads
                dst_idx = (slice(None), slice(None), slice(window_size))
                src_idx = (slice(None), slice(None), 0) 
                if idx_cnt.val is not None:
                    idx_cnt.clear()
                idx_cnt.store(idx.shape[2])
                general_copy(idx_home, dst_idx, idx, src_idx)

            cache_write_buf.store((k, v, new_summary))


class AttentionAfterProj(TransformerComponent):
    def __init__(self, config, env, policy, layer_id):
        super().__init__(config, env, policy, layer_id)

        self.compute:TorchDevice= self.env.gpu
        self.attention_compute = (self.env.cpu if self.policy.cpu_cache_compute
            else self.env.gpu)
        self.weight_load_dst = (self.compute.compressed_device if policy.compress_weight
            else self.compute)

        self.task = None

    def set_task(self, task):
        self.task = task

    def init_weight(self, weight_home, path):
        h, dtype = (self.config.hidden_size, self.config.dtype)
        head_dim = h // self.config.num_attention_heads
        kv_dim = head_dim * self.config.num_key_value_heads
        layer_path = os.path.join(path, f"layers.{self.layer_id}")
        attn_path = layer_path + ".self_attn"

        weight_specs = [
            # w_out
            ((h, h), dtype, attn_path + ".o_proj.weight"),
        ]
        weights = init_weight_list(weight_specs, self.policy, self.env)
        weight_home.store(weights)

    def load_weight(self, weight_home, weight_read_buf, k):
        if k == 0:
            w_out, = weight_home.val
            dst = self.weight_load_dst
            weight_read_buf.store(
                    ( w_out.smart_copy(dst),)
            )

    def _sparse_stage(self, i):
        return self._context_len(i) >= 2048+self.policy.sparse_config.block_size

    def load_cache(self, cache_home, cache_read_buf, i):
        """ attn-load-cache"""
        if i == 0:  # prefill, no cache
            return
        # print(f" >>> step-{i},layer-{self.layer_id} attn-load-cache: {id(cache_home)}")
        # assert isinstance(cache_home, ValueHolder)
        k_home, v_home, summary, (idx_home, idx_cnt) = cache_home.val
        
        dst = self.attention_compute
        context_len = self._context_len(i)

        if not self._sparse_stage(i):
            # load all kv
            curr_cache_idx = ( slice(None), slice(None), slice(0, context_len))
            cache_read_buf.store(
                (   k_home.smart_copy(dst, curr_cache_idx), 
                    v_home.smart_copy(dst, curr_cache_idx),
                    (None, True),
                    (None, True)    )
            )
        else:
            # 1. load blocks    2.load tails
            assert idx_cnt.val is not None
            idx_cnt = idx_cnt.pop()
            
            idx, _= idx_home.smart_copy(k_home.device)
            
            indices = idx.data[:, :, :idx_cnt]

            block_size = self.policy.sparse_config.block_size
            n_tail_cache =  context_len% block_size + block_size

            # context_len-1 since new_k wont be load
            tail_idx = (slice(None), slice(None), slice(context_len-n_tail_cache, context_len-1))

            cache_read_buf.store(
                (   k_home.smart_copy(dst, indices),
                    v_home.smart_copy(dst, indices),
                    k_home.smart_copy(dst, tail_idx),
                    v_home.smart_copy(dst, tail_idx)    )
            )

    
    def forward(self, hidden, cache_read_buf, weight_read_buf, attention_mask,
        position_embeddings, cache_write_buf, i, k):
        """ attn-forward"""
        # print(f" >>> step-{i},layer-{self.layer_id},batch-{k} attn-forward")
        donate = [False] * (1+1+4)

        if k == self.policy.num_gpu_batches - 1:
            ( (w_o, donate[1]), ) = weight_read_buf.pop()
        else:
            ( (w_o, _       ),  ) = weight_read_buf.val

        if i == 0:
            q = hidden.val
            h = self.compute.out_proj(q, w_o)
        else:
            q, new_k, new_v = hidden.val
            (selected_k, donate[1]), (selected_v, donate[2]), \
            (tail_k, donate[3]), (tail_v, donate[4]) = cache_read_buf.pop()

            h = self.compute.gqa_after_proj(q, new_k, new_v, 
                                            w_o, 
                                            tail_k, tail_v, selected_k, selected_v  )        

        hidden.val = h

class MLP(TransformerComponent):
    def __init__(self, config, env, policy, layer_id):
        super().__init__(config, env, policy, layer_id)
        self.compute = self.env.gpu
        self.weight_load_dst = (self.compute.compressed_device if policy.compress_weight
            else self.compute)

        self.task = None

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
        # mlp-load-weight
        w_gate, w_up, w_down , w_ln = weight_home.val
        if k == 0:
            dst1 = self.weight_load_dst
            dst2 = self.compute
            weight_read_buf.store(
                ( w_gate.smart_copy(dst1),
                w_up.smart_copy(dst1),
                w_down.smart_copy(dst2), 
                w_ln.smart_copy(dst2) )
            )


    def input_act_shape_and_dtype(self, batch_size, seq_len):
        return (batch_size, seq_len, self.config.hidden_size), self.config.dtype

    def forward(self, hidden, cache_read_buf, weight_read_buf, attention_mask, position_embed,
                cache_write_buf, i, k):
        # forward-mlp
        donate = [False] * 7
        h, donate[0] = hidden.val, True

        if k == self.policy.num_gpu_batches - 1:
            # Clear the weight_read_buf if it is the last gpu batch
            (   (w_gate,    donate[1]), 
                (w_up,      donate[2]), 
                (w_down,    donate[3]), 
                (w_ln,      donate[4])  ) = weight_read_buf.pop()
        else:
            (   (w_gate,    _), 
                (w_up,      _), 
                (w_down,    _), 
                (w_ln,      _)  ) = weight_read_buf.val

        h = self.compute.qwen_mlp(h, w_gate, w_up, w_down, w_ln, donate, self.layer_id)
        hidden.val = h

        dump_hidden(h.data, "final" , self.layer_id)


class ShiftedTransformerLayer:
    def __init__(self, config, env, policy, layerno):
        self.layer_id = layerno
        self.attn_after_proj = AttentionAfterProj(config, env, policy, layerno-1)
        self.mlp = MLP(config, env, policy, layerno-1)
        self.qkv_proj = QKVProj(config, env, policy, layerno)
        
        self.policy = policy
        self.compute = env.gpu

    def set_task(self, task):
        self.qkv_proj.set_task(task)
        self.mlp.set_task(task)
        self.attn_after_proj.set_task(task)

    def init_weight(self, weight_home, path):
        
        home1, home2, home3 = ValueHolder(),  ValueHolder(),ValueHolder()
        
        self.attn_after_proj.init_weight(home1, path)
        self.mlp.init_weight(home2, path)
        self.qkv_proj.init_weight(home3, path)

        weight_home.store((home1, home2, home3))

    def delete_cache(self, cache_home):
        attn_cache_home, proj_cache_home = cache_home.pop()
        self.attn_after_proj.delete_cache(attn_cache_home)
        self.qkv_proj.delete_cache(proj_cache_home)

    def init_cache_one_gpu_batch(self, cache_home):
        proj_cache_home = ValueHolder()
        
        self.qkv_proj.init_cache_one_gpu_batch(proj_cache_home)
        # self.attn_after_proj.init_cache_one_gpu_batch(attn_cache_home)

        cache_home.store( ( None, proj_cache_home)   )

    def load_weight(self, weight_home, weight_read_buf, k):
        # transformerlayer-load-weight
        if k == 0:
            read_buf1, read_buf2, read_buf3 = ValueHolder(), ValueHolder(), ValueHolder()
            home1, home2, home3 = weight_home.val
            self.attn_after_proj.load_weight(home1, read_buf1, k)
            self.mlp            .load_weight(home2, read_buf2, k)
            self.qkv_proj       .load_weight(home3, read_buf3, k)

            weight_read_buf.store((read_buf1, read_buf2, read_buf3))

    def load_cache(self, cache_home, cache_read_buf, i):
        """ transformer-load-cache"""
        if i == 0:
            return 

        attn_cache_home, proj_cache_home = cache_home.val
        attn_cache_buf, proj_cache_buf = ValueHolder(), ValueHolder()

        self.attn_after_proj.load_cache(attn_cache_home, attn_cache_buf, i)
        self.qkv_proj.load_cache(proj_cache_home, proj_cache_buf, i)
        cache_read_buf.store(
            (   attn_cache_buf, proj_cache_buf )
        )

    def forward(self, 
                hidden, 
                cache_read_buf, weight_read_buf, 
                attention_mask, position_embedding, 
                cache_write_buf, 
                i, k):
        # i : decode step
        # k : id of gpu batch
        if k == self.policy.num_gpu_batches - 1:
            attn_read_buf, mlp_read_buf, proj_read_buf = weight_read_buf.pop()
        else:
            attn_read_buf, mlp_read_buf, proj_read_buf = weight_read_buf.val

        if i == 0:
            attn_cache_buf, proj_cache_buf = None, None
        else:
            attn_cache_buf, proj_cache_buf = cache_read_buf.pop()
        
        attn_cache_write_buf, proj_cache_write_buf = ValueHolder(), ValueHolder()


        self.attn_after_proj.forward(hidden, 
                                     attn_cache_buf, attn_read_buf, 
                                     attention_mask, position_embedding, 
                                     attn_cache_write_buf, 
                                     i, k)

        self.mlp.forward(hidden, None, mlp_read_buf, attention_mask, position_embedding, None,  i, k)

        self.qkv_proj.forward(hidden, 
                              proj_cache_buf, proj_read_buf, None, position_embedding, proj_cache_write_buf, i, k)

        cache_write_buf.store(
            (   attn_cache_write_buf, proj_cache_write_buf  )
        )

    def store_cache(self, cache_home, cache_write_buf, i):
        attn_cache_write_buf, proj_cache_write_buf = cache_write_buf.pop()
        attn_cache_home, proj_cache_home = cache_home.val
        
        # print(f" >>> step{i}, layer{self.layer_id}, store cache")
        self.attn_after_proj.store_cache(attn_cache_home, attn_cache_write_buf, i)
        self.qkv_proj.store_cache(proj_cache_home, proj_cache_write_buf, i)

class TransformerLayer:
    def __init__(self, config, env, policy, layer_id):
        self.attention = SelfAttention(config, env, policy, layer_id)
        self.mlp = MLP(config, env, policy, layer_id)
        self.policy = policy
        self.compute = self.attention.compute

    def set_task(self, task):
        self.attention.set_task(task)
        self.mlp.set_task(task)

    def init_weight(self, weight_home, path):
        home1, home2 = ValueHolder(), ValueHolder()
        self.attention.init_weight(home1, path)
        self.mlp      .init_weight(home2, path)
        weight_home.store((home1, home2))

    def load_weight(self, weight_home, weight_read_buf, k):
        # transformerlayer-load-weight
        read_buf1, read_buf2 = ValueHolder(), ValueHolder()
        home1, home2 = weight_home.val
        self.attention.load_weight(home1, read_buf1, k)
        self.mlp.load_weight(home2, read_buf2, k)
        if k == 0:
            weight_read_buf.store((read_buf1, read_buf2))

    def init_cache_one_gpu_batch(self, cache_home):
        self.attention.init_cache_one_gpu_batch(cache_home)
        
    def load_summary(self, cache_home, cache_read_buf, i):
        self.attention.load_summary(cache_home, cache_read_buf, i)

    def load_cache(self, cache_home, cache_read_buf, i):
        self.attention.load_cache(cache_home, cache_read_buf, i)

    def store_cache(self, cache_home, cache_write_buf, i):
        self.attention.store_cache(cache_home, cache_write_buf, i)

    def forward(self, 
                hidden, 
                cache_read_buf, weight_read_buf, 
                attention_mask, position_embedding, 
                cache_write_buf, 
                i, k):
        # i : decode step
        # k : id of gpu batch
        if k == self.policy.num_gpu_batches - 1:
            attn_read_buf, mlp_read_buf = weight_read_buf.pop()
        else:
            attn_read_buf, mlp_read_buf = weight_read_buf.val

        self.attention.forward(hidden, cache_read_buf, attn_read_buf, 
                               attention_mask, position_embedding, cache_write_buf, i, k)
        self.mlp.forward(hidden, None, mlp_read_buf, attention_mask, position_embedding, None,  i, k)
    

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

    def delete_cache(self, cache_home):
        pass
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
        w_ln, w_token = weight_home.val
        if k == 0:
            dst1 = self.weight_load_dst
            dst2 = self.compute
            weight_read_buf.store(
                (w_ln.smart_copy(dst2), w_token.smart_copy(dst1)))

    def delete_cache(self, cache_home):
        pass

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
                 config: QwenConfig,
                 env: ExecutionEnv,
                 path: str,
                 policy: Policy):
        self.config = config
        self.env = env
        self.path = path
        self.policy = policy
        self.num_gpu_batches = policy.num_gpu_batches
        self.shift_forward = policy.sparse_config.mode == "block"

        self.layers = None
        self.num_layers = 0
        self.construct_layers()

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

    def construct_layers(self):
        layers = []

        num_hidden_layers = self.config.num_hidden_layers
        
        if self.policy.sparse_config.mode == "block":
            layers.append(InputEmbed(self.config, self.env, self.policy))
            layers.append(QKVProj(self.config, self.env, self.policy, 0))

            for j in range(1, num_hidden_layers-1):
                layers.append(ShiftedTransformerLayer(self.config, self.env, self.policy, j))
                
            layers.append(AttentionAfterProj(self.config, self.env, self.policy, num_hidden_layers-1))
            layers.append(MLP(self.config, self.env, self.policy, num_hidden_layers-1))
            layers.append(OutputEmbed(self.config, self.env, self.policy))

        else:
            layers.append(InputEmbed(self.config, self.env, self.policy))
            for j in range(num_hidden_layers):
                if self.policy.sep_layer:
                    layers.append(SelfAttention(self.config, self.env, self.policy, j))
                    layers.append(MLP(self.config, self.env, self.policy, j))
                else:
                    layers.append(TransformerLayer(self.config, self.env, self.policy, j))
            layers.append(OutputEmbed(self.config, self.env, self.policy))

        self.layers = layers
        self.num_layers = len(self.layers)

    def init_weight(self, j):
        expanded_path = os.path.abspath(os.path.expanduser(
            os.path.join(self.path, f"{self.config.name}-np")))
        check_path = os.path.join(expanded_path, "embed_tokens.weight")

        if not os.path.exists(check_path) and DUMMY_WEIGHT not in check_path:
            convert_qwen_weights(self.config.name, self.path)

        self.layers[j].init_weight(self.weight_home[j], expanded_path)

    def load_weight(self, i, j, k, overlap=True):         # generate-load-weight
        # gen-load-weight
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

        # print(f" >>> load cache: ({i},{j},{k})")
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
        if self.policy.sparse_config.mode == "block":
            self.layers[j].delete_cache(self.cache_home[j][k])

    def move_hidden(self, i, j, k):
        self.store_hidden(i, j-1, k)
        self.load_hidden(i, j, k)

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
            val = self.hidden[i][j-1][k].pop()  #.move(dst)
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
        # else:  # move to home
        #     x = self.hidden[i][j][k]
        #     if x.val:  # x may already be moved due to overlapping
        #         x.val = x.val.move(self.act_home)

    def compute_layer(self, i, j, k):
        # Update the hidden in place
        # Clear the weight_read_buf if it is the last gpu batch
        # Clear the cache_read_buf
        # Run layer computation

        # input/output embed layer
        self.layers[j].forward(
                self.hidden[i][j][k], 
                self.cache_read_buf[j][k],
                self.weight_read_buf[j], 
                self.attention_mask[k], 
                self.position_embedding[k], 
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
            head_dim = self.config.hidden_size // self.config.num_attention_heads
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
    def link_cache(self, j, k):
        if j == 2:
            _, proj_cache_home = self.cache_home[j][k].pop()
            last_proj_home = self.cache_home[j-1][k]
            self.cache_home[j][k].store(    (last_proj_home, proj_cache_home)   )
            
        elif j > 2 and j < self.num_layers-3:
            _, proj_cache_home = self.cache_home[j][k].pop()
            _, last_proj_home = self.cache_home[j-1][k].val
            self.cache_home[j][k].store(    (last_proj_home, proj_cache_home)   )

        elif j == self.num_layers - 3:
            _, proj_cache_home = self.cache_home[j-1][k].val
            self.cache_home[j][k] = proj_cache_home

    def check_cache(self):
        for j in range(self.num_layers):
            for k in range(self.num_gpu_batches):
                cache = self.cache_home[j][k]
                assert isinstance(cache, ValueHolder)
                if j == 0 or j == self.num_layers - 1:
                    assert cache.val is None
                elif j == 1:
                    # assert isinstance(cache, ValueHolder)
                    assert isinstance(cache.val, tuple)
                elif j == self.num_layers - 2:  # mlp
                    assert self.cache_home[j][k].val is None
                elif j == self.num_layers -3:   # attn
                    assert isinstance(self.cache_home[j][k], ValueHolder)
                else:
                    assert isinstance(cache.val, tuple) and len(cache.val) == 2
                
    def generate(self,
                 inputs: Union[np.ndarray, List[List[int]]],
                 max_new_tokens: int = 32,
                 do_sample: bool = False,
                 temperature: float = 1.0,
                 stop: Optional[int] = None,
                 debug_mode: Optional[str] = None,
                 cut_gen_len: Optional[int] = None,
                 verbose: int = 0):
        # model.generate
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
            self.position_embedding[k].clear()

        self.hidden = array_3d(gen_len, num_layers, num_gpu_batches, ValueHolder)

        # Init cache
        self.set_task(task)
        for j in range(num_layers):
            for k in range(num_gpu_batches):
                self.init_cache(j, k)
        if self.policy.sparse_config.mode == "block":
            for j in range(num_layers):
                for k in range(num_gpu_batches):
                    self.link_cache(j, k)
            self.check_cache()
            
        if self.policy.cpu_cache_compute:
            self.env.cpu.init_attention_compute_workspace(self.config, self.task, self.policy)

        # Generate
        if debug_mode is None:
            if self.policy.sparse_config.mode == "block":
                self.generation_loop_block_sparse()
            elif not overlap:
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
        elif debug_mode == "output_hidden":
            # No overlap, fewer batches, execution time breakdown
            self.generation_loop_debug_single_batch()
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

    def generation_loop_block_sparse(self):
        # gen-loop-sparse
        # use config to determine which to use
        # return self.generation_loop_overlap_multi_batch()
        print(f" >>> gen-loop-blocksparse")
        for k in range(self.num_gpu_batches):
            self.load_weight(0, 0, k)
        self.load_hidden(0, 0, 0)
        self.sync()

        # Generate
        for i in range(self.execute_gen_len):
            print(f" >>> step-{i}") 
            timers("generate").start()
            for k in range(self.num_gpu_batches):
                self.update_attention_mask(i, k)
                self.update_position_embedding(i, k)
            for j in range(self.num_layers):
                for k in range(self.num_gpu_batches):
                    self.store_hidden(i, j, k-1)
                    self.store_cache(i, j, k-1)

                    self.load_weight(i, j+1, k)
                    self.load_hidden(i, j, k+1)
                    self.load_cache(i, j, k+1)

                    self.compute_layer(i, j, k)
                    self.sync()
            timers("generate").stop()

        # Epilogue
        self.store_hidden(
            self.execute_gen_len-1, self.num_layers-1, self.num_gpu_batches-1)

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
        # gen-loop-sparse
        # Prologue
        for k in range(self.num_gpu_batches):
            self.load_weight(0, 0, k)
        self.sync()

        # Generate
        for i in range(self.execute_gen_len):
            print(f" >>> step-{i}")
            timers("generate").start()
            self.update_attention_mask(i, 0)
            self.update_position_embedding(i, 0)

            for j in range(self.num_layers):
                self.load_hidden(i, j, 0)
                self.compute_layer(i, j, 0)

                self.load_weight(i, j+1, 0)
                self.load_cache(i, j+1, 0)
                self.store_cache(i, j-1, 0)
                self.store_hidden(i, j, 0)
                self.sync()
            timers("generate").stop()

            if self.task.stop and np.all(self.stopped):
                break

    def generation_loop_overlap_multi_batch(self):
        # generate-loop-overlap
        # Prologue
        for k in range(self.num_gpu_batches):
            self.load_weight(0, 0, k)
        self.load_hidden(0, 0, 0)
        self.sync()

        # Generate
        for i in range(self.execute_gen_len):
            print(f" >>> step-{i}")
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
                    self.store_cache(i, j, k-1)
                    self.compute_layer(i, j, k)
                    self.sync()
            timers("generate").stop()

        # Epilogue
        self.store_hidden(
            self.execute_gen_len-1, self.num_layers-1, self.num_gpu_batches-1)

    def generation_loop_debug_single_batch(self):
        execute_num_batches = 20
        batch_ct = 0
        pbar = tqdm.tqdm(total=execute_num_batches)
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


def get_env():
    gpu = TorchDevice("cuda:0")
    cpu = TorchDevice("cpu")
    disk = TorchDisk(args.offload_dir)
    env = ExecutionEnv(gpu=gpu, cpu=cpu, disk=disk, mixed=TorchMixedDevice([gpu, cpu, disk]))
    return env


def build_policy(args):
    policy = Policy(args.gpu_batch_size, args.num_gpu_batches,
                    args.percent[0], args.percent[1],
                    args.percent[2], args.percent[3],
                    args.percent[4], args.percent[5],
                    args.overlap, args.sep_layer, args.pin_weight,
                    args.cpu_cache_compute, 
                    SparseConfig(mode=args.sparse_mode, block_size=args.sparse_block_size),
                    args.compress_weight,
                    CompressionConfig(num_bits=4, group_size=64,
                                      group_dim=0, symmetric=False),
                    args.compress_cache,
                    CompressionConfig(num_bits=4, group_size=64,
                                      group_dim=2, symmetric=False))
    assert not (args.compress_cache and args.sparse_config.mode != "dense" ), "Not implemented"
    return policy


def build_env(args):
    gpu = TorchDevice("cuda:0")
    cpu = TorchDevice("cpu")
    disk = TorchDisk(args.offload_dir)
    env = ExecutionEnv(gpu=gpu, cpu=cpu, disk=disk, mixed=TorchMixedDevice([gpu, cpu, disk]))

    return env

    
def basic_test(args):
    input_ids = [[100, 200, 300, ]]
    # input_ids = [[100, 200, 300, 279, 1156]]

    policy = build_policy(args)
    config = get_qwen_config(args.model)

    env = get_env()

    my_model = QwenLM(config, env, args.path, policy)

    try:
        outputs = my_model.generate(input_ids, max_new_tokens=args.gen_len)
    finally:
        env.close_copy_threads()

    print(outputs)

def predict_mem_and_log(qwen_config, num_prompts, prompt_len, gen_len):
    context_len = prompt_len + gen_len 
    cache_size = qwen_config.cache_bytes(num_prompts, context_len)
    hidden_size = qwen_config.hidden_bytes(num_prompts, context_len)
    print(f" >>> genlen: {context_len}, ")
    print(f"model size: {qwen_config.model_bytes()/GB:.3f} GB, "
    f"cache size: {cache_size/GB:.3f} GB, "
    f"hidden size (prefill): {hidden_size/GB:.3f} GB")

    return hidden_size, cache_size



def run_flexllmgen(args):
    print(f"<run_flexllmgen>: args.model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side="left")
        # print(f" >>> tokenizer use model: {args.model}")
        
    # prompt_len, gen_len, cut_gen_len = args.prompt_len, args.gen_len, args.cut_gen_len
    gen_len = args.gen_len
    cut_gen_len = args.cut_gen_len
    num_prompts = args.num_gpu_batches * args.gpu_batch_size

    # Task and policy
    if args.input_file is not None:  
        input_in_tokens = get_file_inputs(args.input_file, num_prompts, tokenizer, args.prompt_len)
    elif args.prompt_len is not None:
        input_in_tokens = get_test_inputs(args.prompt_len, num_prompts, tokenizer)
    else:
        input_in_tokens = get_compact_test_inputs(num_prompts, tokenizer)

    # in shape (B, S)
    warmup_inputs = get_test_inputs(32, num_prompts, tokenizer)
    wramup_prompt_len = len(warmup_inputs[0])
    prompt_len = len(input_in_tokens[0])
    
    # policy, env preparations
    policy = build_policy(args)    
    env = build_env(args)
    
    qwen_config = get_qwen_config(args.model)

    hidden_size, cache_size  = predict_mem_and_log(qwen_config, num_prompts, prompt_len, gen_len)

    print("init weight...")
    model = QwenLM(qwen_config, env, args.path, policy)

    try:
        print("warmup - generate")
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
    _, gpu_peak_mem = env.gpu.mem_stats()
    _, cpu_peak_mem = env.cpu.mem_stats()

    if DUMMY_WEIGHT not in args.path:
        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        show_str = "Outputs:\n" + 70 * '-' + "\n"
        for i in [0, len(outputs)-1]:
            show_str += f"{i}: {outputs[i]}\n"
            show_str += "-" * 70 + "\n"
        if args.verbose >= 2:
            print(show_str)

    env.gpu.print_stats()
    env.cpu.print_stats()
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
    #args
    parser.add_argument("--model", type=str, default="Qwen/Qwen2-0.5B",
        help="The model name.")
    parser.add_argument("--path", type=str, default="~/qwen_weights",
        help="The path to the model weights. If there are no cached weights, "
             "FlexLLMGen will automatically download them from HuggingFace.")
    parser.add_argument("--offload-dir", type=str, default="flexllmgen_offload_dir",
        help="The directory to offload tensors. ")
    parser.add_argument("--gen-len", type=int, default=32)
    parser.add_argument("--cut-gen-len", type=int,
        help="Cut generation length for fast debugging.")
    parser.add_argument("--prompt-len", type=int, 
            help="combined with input-file, get the front tokens, default: None, use the longest-padding of tokenizer")
    parser.add_argument("--debug-mode", type=str,
        choices=["fewer_batch", "breakdown", "output_hidden", "basic"])

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
        const=True, default=False)
    parser.add_argument("--pin-weight", type=str2bool, nargs="?",
        const=True, default=True)

    parser.add_argument("--cpu-cache-compute", action="store_true")
    parser.add_argument("--sparse-mode", type=str, choices=["dense", "naive", "block"], default="dense")
    parser.add_argument("--attn-sparsity", type=float, default=1.0)
    parser.add_argument("--sparse-block-size", type=int, default=16)
    
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

    if args.debug_mode == "basic":
        basic_test(args)
    else:
        run_flexllmgen(args)


