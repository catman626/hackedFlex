"""Implement tensor computations with pytorch."""
from enum import Enum, auto
from functools import partial
from itertools import count
import os
import queue
import shutil
import time
import threading
from typing import Optional, Union, Tuple

import torch
import torch.nn.functional as F
import numpy as np

from flexllmgen.utils import (GB, T, cpu_mem_stats, vector_gather,
    np_dtype_to_torch_dtype, torch_dtype_to_np_dtype,
    torch_dtype_to_num_bytes, expand_block_idx, ValueHolder, num_block, 
    bhsd_to_cache_shape, tail_length, bhsd_to_bsH)

general_copy_compressed = TorchCompressedDevice = None
global_cpu_device = None
global_disk_device = None

DUMP_HIDDEN = False
# DUMP_HIDDEN = True
DUMP_VERBOSE = False

sparse_block_summary: torch.Tensor

def dump_hidden(torch_obj, name, layerno=None, seq_len=None):
    global DUMP_HIDDEN, DUMP_VERBOSE
    if DUMP_HIDDEN == False:
        return
    
    if layerno and layerno > 0:
        return
    
    if seq_len is not None:
        save_name = f"my/{name}-seq-{seq_len}"
        log_info = f" >>> save {name} at seq-len: {seq_len}"
    elif layerno is None:
        save_name = f"my/{name}"
        log_info = f" >>> save {name}"
    else :
        save_name = f"my/layer-{layerno}-{name}" 
        log_info = f" >>> save {name} of layer {layerno}"

    if DUMP_VERBOSE:
        print(log_info)

    torch.save(torch_obj, save_name)
 
def repeat_interleave(kv_data, n_qhead, n_kvhead):
    assert len(kv_data.shape) == 4
    return torch.repeat_interleave(kv_data, n_qhead//n_kvhead, dim=1)

def check_idx(idx):
    mx = idx.max()
    mn = idx.min()
    assert mx < 1000000 and mn >= 0

def fix_recursive_import():
    global general_copy_compressed, TorchCompressedDevice, global_cpu_device
    from flexllmgen import compression
    general_copy_compressed = compression.general_copy_compressed
    TorchCompressedDevice = compression.TorchCompressedDevice

def derive_shapes(q, k):
    b, n_qhead, tgt_s, head_dim = q.shape
    src_s = k.shape[2]
    n_kvhead = k.shape[1]

    return b, n_qhead, n_kvhead, tgt_s, src_s, head_dim

def cache_shape(config, task, policy):
    n_qhead, n_kvhead, hidden_size, prompt_len, gen_len, gpu_batch_size = (
        config.num_attention_heads, config.num_key_value_heads, config.hidden_size, task.prompt_len, task.gen_len,
        policy.gpu_batch_size)
    shape = (prompt_len + gen_len - 1, gpu_batch_size * n_kvhead, hidden_size // n_qhead * n_kvhead)

    return shape

def extract_cache_to_bhsd(cache, b, n_head, s, d):
    return cache.data[:s].view(s, b, n_head, d).permute(1,2,0,3)
  
    # k = k_cache.data[:src_s].view(src_s, b, n_kvhead, head_dim).permute(1, 2, 0, 3)

def select_topk(v:torch.Tensor, indices:torch.Tensor):
    """
    v cache, (b, kvhead, s, d)
    indices: (b, qhead, 1, k)
    """
    assert len(v.shape) == 4
    b, n_qhead, tgt_s, k = indices.shape
    n_kvhead = v.shape[1]
    group_size = n_qhead // n_kvhead    

    indices = indices.view(b, n_kvhead, group_size, tgt_s, k)

    select_shape = (b, n_qhead, k)
    
    idx_b = torch.arange(b).view(-1,1,1).expand(*select_shape)
    idx_h = torch.arange(n_kvhead).repeat_interleave(group_size).view(1, -1, 1).expand(*select_shape)
    idx_k = indices.flatten().view(*select_shape)
    # assert idx_b.shape == idx_k.shape and idx_h.shape == idx_k.shape, \
    #     f" >>>>>> idx_b: {idx_b.shape}, idx_h: {idx_h.shape}, idx_k:{idx_k.shape}"

    return v[idx_b, idx_h, idx_k]

def copy_indices_to_shape(indices:Tuple, src_shape):
    shape = list(src_shape)
    if isinstance(indices, tuple):
        for i, idx in enumerate(indices):
            if idx.stop is not None:
                shape[i] = idx.stop - idx.start
        return tuple(shape)
        
    elif  isinstance(indices, TorchTensor):
        raise NotImplementedError()
    elif isinstance(indices, torch.Tensor):
        assert len(indices.shape) == 3 
        head_dim = src_shape[-1]

        dst_shape = indices.shape + (head_dim, )
        return dst_shape
    else :
        print(f" >>> idx type: {type(indices)}")
        raise NotImplementedError()

def index_into(data:torch.Tensor, idx, ensure_view=False):
    if isinstance(idx, torch.Tensor):
        # case of block sparse
        assert not ensure_view 
        n_kvhead, head_dim = data.shape[1], data.shape[-1]
        n_qhead = idx.shape[1]
        
        selected = data.repeat_interleave(n_qhead//n_kvhead, dim=1)\
                        .gather(dim=-2, 
                                index=idx.unsqueeze(-1).expand(-1,-1,-1,head_dim))

    elif isinstance(idx, tuple):
        selected = data[idx] 
    else :
        raise NotImplementedError()

    return selected

        
class DeviceType(Enum):
    CPU = auto()
    CUDA = auto()
    DISK = auto()
    MIXED = auto()
    COMPRESSED = auto()

    @staticmethod
    def convert(name):
        if name == "cpu":
            return DeviceType.CPU
        elif name == "cuda":
            return DeviceType.CUDA
        elif name == "disk":
            return DeviceType.DISK
        elif name == "mixed":
            return DeviceType.MIXED
        elif name == "compressed":
            return DeviceType.COMPRESSED
        else:
            raise ValueError(f"Invalid name: {name}")


class TorchTensor:
    """
    Wrap pytorch tensors to support
      - Unified representation for normal and compressed tensors on
        GPUs, CPUs, disks and mixed devices.
      - Asynchronous copy between tensors on any formats and any devices.

    This is achieved by implementing the data movement APIs for primitive cases
    and using recursive structures to handle other combinations.

    Note:
    For a tensor on a TorchDevice, self.data is a primitive tensor.
      type: torch.Tensor.
    For a tensor on a TorchDisk, self.data is a filename.
      type: str
    For a tensor on a TorchMixedDevice, self.data is (tensors, segment_points)
      type: Tuple[Tuple[TorchTensor], Tuple[int]]
    For a tensor on a TorchCompressedDevice, self.data is (data, scale, compression_config)
      type: Tuple[TorchTensor, TorchTensor, CompressionConfig]
    """
    name_count = count()

    def __init__(self, shape, dtype, data, device, name=None):
        if isinstance(data, torch.Tensor):
            assert data.device == device.dev

        self.shape = shape
        self.dtype = dtype
        self.data = data
        self.device = device

        # Whether delete the file when the tensor is deleted
        self.delete_file = True

        self.name = name or TorchTensor.next_name()

    @property
    def bytes(self):
        return np.prod(self.shape) * torch_dtype_to_num_bytes[self.dtype]

    @classmethod
    def next_name(cls):
        return f"t_{next(cls.name_count)}"

    @classmethod
    def create_from_torch(cls, data, device, name=None):
        return cls(data.shape, data.dtype, data, device, name=name)

    def delete(self):
        assert self.device is not None, "already deleted"
        if self.device.device_type == DeviceType.DISK:
            self.device.delete(self)
        self.device = self.data = None

    def load_from_np(self, np_array):
        if self.device.device_type == DeviceType.DISK:  # save to np file
            with open(self.data, "wb") as fout:
                np.save(fout, np_array)
        else:
            if self.device.device_type == DeviceType.COMPRESSED:
                tmp = torch.from_numpy(np_array)
                tmp = global_cpu_device.compressed_device.compress(tmp, self.data[2])
                general_copy(self, None, tmp, None)
            else:   
                
                self.data.copy_(torch.from_numpy(np_array))

    def load_from_np_file(self, filename):
        if self.device.device_type == DeviceType.DISK:
            shutil.copy(filename, self.data)
        else:
            # self.load_from_np(torch.load(filename))
            self.load_from_np(np.load(filename))

    def copy(self, dst, src_indices=None):
        if src_indices is not None:
            shape = copy_indices_to_shape(src_indices, self.shape)
        else:
            shape = self.shape

        if dst.device_type == DeviceType.COMPRESSED:
            ret = dst.allocate(shape, torch_dtype_to_np_dtype[self.dtype], self.data[2])
        else:
            ret = dst.allocate(shape, torch_dtype_to_np_dtype[self.dtype])
        general_copy(ret, None, self, src_indices)
        return ret

    def smart_copy(self, dst, src_indices=None):
        """ dst is TorchDevice"""
        if self.device == dst:
            return self, False
        return self.copy(dst, src_indices=src_indices), True

    def move(self, dst):
        if self.device == dst:
            return self
        ret = self.copy(dst)
        self.delete()
        return ret

    def __str__(self):
        return (f"TorchTensor(shape={self.shape}, dtype={str(self.dtype)}, "
                f"device={self.device.name if self.device else None})")

def rms_layernorm(x, w):
    eps = 1e-6
    rms = torch.sqrt(torch.mean(x **2, dim=-1, keepdim=True) + eps)
    return w * (x / rms)

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

class TorchDevice:
    """Wrap tensor and computation APIs of a single CPU or GPU."""

    def __init__(self, name, mem_capacity=None, flops=None):
        self.name = name
        self.mem_capacity = mem_capacity
        self.flops = flops

        self.dev = torch.device(name)
        self.device_type = DeviceType.convert(self.dev.type)
        self.compressed_device = TorchCompressedDevice(self)

        self.links = {}

        self.attention_compute_workspace = None
        self.workspace_pt = 0

        if self.device_type == DeviceType.CPU:
            global global_cpu_device
            global_cpu_device = self

    def add_link(self, link):
        dst = link.b if link.a == self else link.a
        self.links[dst] = link

    def allocate(self, shape, dtype, pin_memory=None, name=None):
        if self.device_type == DeviceType.CPU:
            pin_memory = True if pin_memory is None else pin_memory
        else:
            pin_memory = False
        dtype = np_dtype_to_torch_dtype[dtype]
        # dtype = torch.bfloat16
        data = torch.empty(shape, dtype=dtype, pin_memory=pin_memory, device=self.dev)
        return TorchTensor.create_from_torch(data, self, name=name)

    def delete(self, tensor):
        pass

    def init_attention_compute_workspace(self, config, task, policy):
        if self.device_type != DeviceType.CPU:
            return  # Only CPU requires this fp32 workspace

        if not policy.compress_cache:
            b = policy.gpu_batch_size
            n_head = config.n_head
            head_dim = config.input_dim // n_head
            max_seq_len = task.prompt_len + task.gen_len - 1
            self.attention_compute_workspace = []
            self.workspace_pt = 0

            # We currently separate SelfAttention and MLP as two layers,
            # so we only need one workspace instead of two.
            for i in range(1 if policy.sep_layer else 2):
                shape = (max_seq_len, b * n_head, head_dim)
                k_cache = self.allocate(shape, np.float32, pin_memory=False)
                v_cache = self.allocate(shape, np.float32, pin_memory=False)
                self.attention_compute_workspace.append((k_cache, v_cache))
        else:
            self.compressed_device.init_attention_compute_workspace(
                config, task, policy)

    def next_attention_compute_workspace(self):
        self.workspace_pt = (self.workspace_pt + 1) % len(
            self.attention_compute_workspace)
        return self.attention_compute_workspace[self.workspace_pt]

    def del_attention_compute_workspace(self):
        self.attention_compute_workspace = None

    def gen_attention_mask(self, token_ids, pad_token_id, donate):
        data = token_ids.data.ne(pad_token_id)
        if donate[0]: token_ids.delete()
        return TorchTensor.create_from_torch(data, self)

    def extend_attention_mask(self, attention_mask, donate):
        bs = attention_mask.shape[0]
        data = torch.concat((attention_mask.data,
             torch.ones((bs, 1), dtype=attention_mask.dtype, device=self.dev)), dim=1)
        if donate[0]: attention_mask.delete()
        return TorchTensor.create_from_torch(data, self)

    def opt_input_embed(self, inputs, attention_mask, w_token, w_pos, pad_token_id, donate):
        # decompress weights
        if w_token.device.device_type == DeviceType.COMPRESSED:
            w_token = w_token.device.decompress(w_token)
            w_pos = w_pos.device.decompress(w_pos)

        token_ids = inputs.data
        mask = attention_mask.data
        if donate[0]: inputs.delete()
        if donate[1]: attention_mask.delete()

        # token embedding
        token_embed = F.embedding(token_ids, w_token.data, pad_token_id)

        # pos embedding
        positions = torch.cumsum(mask, dim=1).int() * mask + 1

        # cut positions if `past_key_values_length` is > 0
        past_key_values_length = mask.shape[1] - token_ids.shape[1]
        positions = positions[:, past_key_values_length:]

        pos_embed = F.embedding(positions, w_pos.data)

        data = token_embed + pos_embed
        return TorchTensor.create_from_torch(data, self)

    def opt_output_embed(self, inputs, w_ln, b_ln, w_token, donate,
                         do_sample, temperature):
        # decompress weights
        if w_token.device.device_type == DeviceType.COMPRESSED:
            w_token = w_token.device.decompress(w_token)

        b, s, h = inputs.shape

        hidden = F.layer_norm(inputs.data, (h,), weight=w_ln.data, bias=b_ln.data)
        if donate[0]: inputs.delete()

        # output embedding
        logits = F.linear(hidden, w_token.data)
        last_token_logits = logits[:,-1,:]

        if do_sample and not temperature < 1e-5:
            probs = torch.softmax(last_token_logits / temperature, dim=-1)
            ids = torch.multinomial(probs, num_samples=1)
        else:
            ids = last_token_logits.argmax(dim=1, keepdim=True)
        return TorchTensor.create_from_torch(ids, self)

    def qwen_input_embed(self, inputs, w_token, pad_token_id, donate):
        # decompress weights
        if w_token.device.device_type == DeviceType.COMPRESSED:
            w_token = w_token.device.decompress(w_token)

        token_ids = inputs.data
        if donate[0]: inputs.delete()

        # only token-embedding, no position embeddings
        token_embed = F.embedding(token_ids, w_token.data, pad_token_id)

        dump_hidden(token_embed, "embed_tokens")

        return TorchTensor.create_from_torch(token_embed, self)

    def qwen_output_embed(self, inputs, w_ln, w_token, donate,
                         do_sample, temperature):
        # decompress weights
        if w_token.device.device_type == DeviceType.COMPRESSED:
            w_token = w_token.device.decompress(w_token)

        b, s, h = inputs.shape

        hidden = rms_layernorm(inputs.data, w_ln.data)
        # hidden = F.layer_norm(inputs.data, (h,), weight=w_ln.data, bias=b_ln.data)
        if donate[0]: inputs.delete()

        # output embedding
        logits = F.linear(hidden, w_token.data)
        last_token_logits = logits[:,-1,:]

        if do_sample and not temperature < 1e-5:
            probs = torch.softmax(last_token_logits / temperature, dim=-1)
            ids = torch.multinomial(probs, num_samples=1)
        else:
            ids = last_token_logits.argmax(dim=1, keepdim=True)
        return TorchTensor.create_from_torch(ids, self)

    def init_cache_one_gpu_batch_qwen(self, config, task, policy):
        n_qhead, n_kvhead, hidden_size, prompt_len, gen_len, gpu_batch_size = (
                config.num_attention_heads, config.num_key_value_heads, config.hidden_size, 
                task.prompt_len, task.gen_len,
                policy.gpu_batch_size)

        head_dim = hidden_size // n_qhead
        shape = (gpu_batch_size, n_kvhead, prompt_len + gen_len - 1, head_dim)

        pin_memory = False
        k_cache = self.allocate(shape, config.dtype, pin_memory=pin_memory)
        v_cache = self.allocate(shape, config.dtype, pin_memory=pin_memory)

        if policy.sparse_config.mode != "block":
            return k_cache, v_cache

        # the s%block_sz tokens must be indiced
        block_size = policy.sparse_config.block_size 
        num_block = (prompt_len + gen_len) // block_size + block_size

        summary_shape = (gpu_batch_size , n_kvhead, num_block, head_dim)
        
        summary_cache = self.allocate(
                summary_shape, 
                config.dtype, pin_memory=pin_memory)
        idx_cache = self.allocate(
                (gpu_batch_size, n_qhead, prompt_len+gen_len),
                np.int32, pin_memory=pin_memory)

        return k_cache, v_cache, summary_cache, (idx_cache, ValueHolder())

    def init_cache_one_gpu_batch(self, config, task, policy):
        if config.name.startswith("qwen"):
            return self.init_cache_one_gpu_batch_qwen(config, task, policy)

        assert config.name.startswith("opt")
        n_head, hidden_size, prompt_len, gen_len, gpu_batch_size = (
                config.n_head, config.hidden_size, task.prompt_len, task.gen_len,
                policy.gpu_batch_size
                )
        shape = (prompt_len + gen_len-1, gpu_batch_size *n_head, hidden_size//n_head)

        # print(f" >>> init cache in shape: {shape}")
        # shape = cache_shape(config, task, policy)
        # NOTE: disable pin_memory due to high memory overhead
        pin_memory = False
        k_cache = self.allocate(shape, config.dtype, pin_memory=pin_memory)
        v_cache = self.allocate(shape, config.dtype, pin_memory=pin_memory)

        return k_cache, v_cache

    def out_proj(self, h, w_o):
        h = F.linear(h.data, w_o.data)

        return TorchTensor.create_from_torch(h, self)
        

    def _qkv_proj(self, h, 
                 attn_mask, position_embed, 
                 w_q, b_q, w_k, b_k, w_v, b_v, w_ln, 
                 n_head, return_form="bhsd"):
        """ input, return:  torch.tensor
            return: (b, h, s, d) 
        """
        hidden =  rms_layernorm(h, w_ln)
        cos, sin = position_embed
        n_qhead, n_kvhead = n_head
        b, s, h = h.shape
        head_dim = h//n_qhead

        # shape: (b, s, h)
        q = F.linear(hidden, w_q, bias=b_q) 
        k = F.linear(hidden, w_k, bias=b_k)
        v = F.linear(hidden, w_v, bias=b_v)

        # shape: (b, s, n_head, head_dim) -> (b, head, s, head_dim)
        q = q.view(b, s, n_qhead,  head_dim).transpose(1,2)
        k = k.view(b, s, n_kvhead, head_dim).transpose(1,2)
        v = v.view(b, s, n_kvhead, head_dim).transpose(1,2)

        # (b, head, s, d)
        q_pos = apply_rope(q, (cos, sin), unsqueeze_dim=0)
        k_pos = apply_rope(k, (cos, sin), unsqueeze_dim=0)

        if return_form == "bsH":
            q_pos = bhsd_to_bsH(q_pos)
            k_pos = bhsd_to_bsH(k_pos)
            v = bhsd_to_bsH(v)

        return q_pos, k_pos, v 

    def qkv_proj_prefill(self, h, attn_mask, position_embed, 
                         w_q, b_q, w_k, b_k, w_v, b_v, w_ln, 
                         n_head, enable_sparse=False, block_size=0):
        """ return q:(bsH)"""
        assert len(h.shape) == 3
        b, s, hidden_dim = h.shape
        n_qhead, n_kvhead = n_head
        head_dim = hidden_dim // n_qhead

        cos = position_embed.data[0][:s]
        sin = position_embed.data[1][:s]

        mask = attn_mask.data if attn_mask is not None else None
        q, k, v = self._qkv_proj(h.data, mask, (cos, sin), 
                                 w_q.data, b_q.data, w_k.data, b_k.data, w_v.data, b_v.data, w_ln.data,
                                 n_head)
        
        if enable_sparse:
            last_q = q[:, :, -block_size:]
            k1 = torch.repeat_interleave(k, n_qhead//n_kvhead, dim=1)
            attn_score = torch.matmul(last_q, k1.transpose(2, 3))
            n_sample_kv = s//10
            

            q = q * (head_dim ** -0.5)
            sparse_attn_score = torch.matmul(q, k1[:,:,-n_sample_kv:].transpose(2,3)    ) 

            # (b, h, s, k)
            attn_weight = torch.softmax(sparse_attn_score, dim=-1)

            # print(f" >>> n-sample-kv:{n_sample_kv}")

            # (b, h, s, k) * (b, h, k, d) -> (b, h, s, d)
            v1 = repeat_interleave(v[:, :, -n_sample_kv:], n_qhead, n_kvhead)

            # print(f" >>> shape of attn_weight: {attn_weight.shape}")
            attn_out = torch.matmul(attn_weight, v1)

            attn_out = attn_out.permute(0, 2, 1, 3).flatten(start_dim=2)

            tail_len = s % block_size + block_size
            n_blk = num_block(s, block_size)
            summary = k[:,:,:-tail_len].reshape(b, n_kvhead, n_blk, block_size, head_dim).mean(dim=-2)

            # print(f" >>> shape of summary: {summary.shape}")
            
            summary = TorchTensor.create_from_torch(summary, self)

        else:
            attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
            summary = None
            # bhsd -> bshd -> bsH
            attn_out = bhsd_to_bsH(attn_out)
        
        return TorchTensor.create_from_torch(attn_out, self), \
            TorchTensor.create_from_torch(k, self), \
                TorchTensor.create_from_torch(v, self), \
                    summary

    def qkv_proj_decode(self, 
                        h, attn_mask, position_embed, 
                        w_q, b_q, w_k, b_k, w_v, b_v, w_ln, 
                        tail_k, tail_v, k_summary,
                        n_head, context_len, enable_sparse=False, block_size=None):
        """ k_summary: (b, h, blk, d)"""
        head_dim = h.shape[-1]
        cos = position_embed.data[0][context_len-1:context_len]
        sin = position_embed.data[1][context_len-1:context_len]

        mask = None if attn_mask is None else attn_mask.data
        q, k, v = self._qkv_proj(h.data, mask, (cos, sin), 
                                 w_q.data, b_q.data, w_k.data, b_k.data, w_v.data, b_v.data, w_ln.data,
                                 n_head)
        # get idx
        if not enable_sparse:
            # use q for return value 
            q = F.scaled_dot_product_attention(q, k, v, enable_gqa=True)
            
            idx = None
            summary_ret = None
        else:
            if k_summary is None:
                tail_len = tail_length(context_len, block_size)
                assert tail_k.data.shape[2] == context_len-tail_len
                b, n_kvhead, s, head_dim = tail_k.data.shape
                assert s % block_size == 0
                # (b, head, s, d) -> (b, head, nblk, blk, d)
                summary = tail_k.data.reshape(b, n_kvhead, -1, block_size, head_dim) \
                                        .mean(dim=-2, keepdim=False)
                
                summary_ret = TorchTensor.create_from_torch(summary, self)

            elif context_len % block_size == 0:
                tail_len = tail_length(context_len, block_size)
                last_block = torch.concat([tail_k.data, k], dim=2)

                # print(f" >>> shape of k: {k.shape}")
                # print(f" >>> shape of tail_k.data: {tail_k.data.shape}")
                assert last_block.shape[2] == block_size, \
                    f" !!!block-size:{last_block.shape[2]}, context-len:{context_len}"

                new_summary = last_block.mean(dim=2, keepdim=True)
                summary = torch.concat([k_summary.data, new_summary], dim=2)
                summary_ret = TorchTensor.create_from_torch(new_summary, self)
            else:
                summary = k_summary.data
                summary_ret = None  # no summary update 

            b, n_kvhead, num_blk, head_dim = summary.shape

            blk_attn_score = self._gqa_attn_weights(q, summary, mask=None)

            topk = max(num_blk//5, 1)
            # (b, h, k)
            top_weight, top_idx = torch.topk(blk_attn_score, num_blk // 5, dim=-1, sorted=False)
            idx = expand_block_idx(top_idx, block_size)
            check_idx(idx)
            idx = TorchTensor.create_from_torch(idx, self)

        return  TorchTensor.create_from_torch(q, self), \
                TorchTensor.create_from_torch(k, self),  \
                TorchTensor.create_from_torch(v, self), \
                summary_ret, idx

       
    def _summary(self, k, block_size, new_k=None, block_range=None):
        """ k, v: (s, b*h, d) 
            return torch.Tensor"""
        k = k.data

        if new_k is not None:
            k = torch.concat([k, new_k.data], dim=2)

            assert len(k.shape) == 4
            assert k.shape[2] % block_size == 0, f"invalid summary time when tail-len={k.shape[2]}"
            

        b, n_kvhead, s, head_dim = k.shape
        if block_range is not None:
            s = min(s, block_range)

        num_block = s // block_size
        k = k[:num_block*block_size]

        cache_shape = k.shape
        k = k.view(num_block, block_size, *cache_shape)
        
        k = k.mean(dim=1)
        
        return k

    def gqa_after_proj(self, q, k, v, 
                       w_o, 
                       tail_k, tail_v, k_cache, v_cache):
        """ q:      (b, head, s, d)
            kv:     (s, b*head, d)
            ret:    (b,head, s, d)
        """
        b, n_qhead, s, head_dim = q.shape
        n_kvhead = tail_k.shape[1]
        
        q, k,v = q.data, k.data, v.data

        # (b, head, s, d)
        concated_k = []
        concated_v = []
        if k_cache is not None:
            k_cache = k_cache.data
            v_cache = v_cache.data
            if k_cache.shape[-2] != n_qhead:
                k_cache = repeat_interleave(k_cache, n_qhead, n_kvhead)
                v_cache = repeat_interleave(v_cache, n_qhead, n_kvhead)
            concated_k.append(k_cache)
            concated_v.append(v_cache)
            
        if tail_k is not None:
            tail_k = torch.repeat_interleave(tail_k.data, n_qhead//n_kvhead, dim=1)
            tail_v = torch.repeat_interleave(tail_v.data, n_qhead//n_kvhead, dim=1)
            concated_k.append(tail_k)
            concated_v.append(tail_v)
        
        concated_k.append(k)
        concated_v.append(v)
        
        k = torch.concat(concated_k, dim=-2)
        v = torch.concat(concated_v, dim=-2)
                
        # (b, head, s, d)
        attn_output = F.scaled_dot_product_attention(q, k, v, enable_gqa=True)
        # 
        attn_output = attn_output.permute(0, 2, 1, 3).view(b, s, n_qhead * head_dim)
        output = F.linear(attn_output, w_o.data)
        
        return TorchTensor.create_from_torch(output, self)

    def gqa(self, inputs, attention_mask, position_embed, w_q, b_q, w_k, b_k, w_v, b_v,
            w_out, w_ln, n_head, donate, compress_cache, comp_config, layerno):
        """Multi-head attention (prefill phase)."""
        # decompress weights
        if w_q.device.device_type == DeviceType.COMPRESSED:
            w_q = w_q.device.decompress(w_q)
            w_k = w_k.device.decompress(w_k)
            w_v = w_v.device.decompress(w_v)
            w_out = w_out.device.decompress(w_out)

        n_qhead, n_kvhead = n_head
        n_head = None      # in case visited later

        b, s, h = inputs.shape
        head_dim = h // n_qhead

        cos = position_embed.data[0][:s]
        sin = position_embed.data[1][:s]

        dump_hidden(cos, "cos", layerno)
        
        dump_hidden(inputs.data, "rms-input", layerno)
        hidden = rms_layernorm(inputs.data, w_ln.data)
        dump_hidden(hidden, "attn-input", layerno)

        # shape: (b, s, h)
        # q = F.linear(hidden, w_q.data, bias=b_q.data) * scaling
        q = F.linear(hidden, w_q.data, bias=b_q.data) 
        k = F.linear(hidden, w_k.data, bias=b_k.data)
        v = F.linear(hidden, w_v.data, bias=b_v.data)
        # shape: (b, s, n_head, head_dim) -> (b, head, s, head_dim)
        q = q.view(b, s, n_qhead,  head_dim).transpose(1,2)
        k = k.view(b, s, n_kvhead, head_dim).transpose(1,2)
        v = v.view(b, s, n_kvhead, head_dim).transpose(1,2)

        # (b, head, s, d)
        q_pos = apply_rope(q, (cos, sin), unsqueeze_dim=0)
        k_pos = apply_rope(k, (cos, sin), unsqueeze_dim=0)

        dump_hidden(q_pos, "q_pos", layerno)
        dump_hidden(k_pos, "k_pos", layerno)

        # shape: (b, 1, s, s)
        idx = torch.arange(s, device=self.dev)
        causal_mask = (idx <= idx.view(s, 1)).view(1, 1, s, s)
        mask = attention_mask.data.view(b, 1, 1, s) & causal_mask

        # attention-core
        attn_output = F.scaled_dot_product_attention(q_pos, k_pos, v, attn_mask=mask, enable_gqa=True)
        # attn_output = fi.

        dump_hidden(attn_output, "sdpa", layerno)   # in shape (b, head, s, h)
        
        # -> (B, nhead, s, h) -> (B, s, nhead,  h) -> (B, S, H)
        attn_output = attn_output.transpose(1, 2).contiguous().view(b, s, -1)

        attn_output = F.linear(attn_output, w_out.data, bias=None)
        
        dump_hidden(attn_output, "out-proj", layerno)

        attn_output.add_(inputs.data)
        
        dump_hidden(attn_output, "attn-output", layerno)

        if donate[0]: inputs.delete()
        if donate[1]: attention_mask.delete()

        #  (b, n_head, s, d) -> (s, b, head, d) -> (s, b * n_head, d)
        k = bhsd_to_cache_shape(k_pos)
        v = bhsd_to_cache_shape(v)
    

        if compress_cache:
            k = self.compressed_device.compress(k, comp_config)
            v = self.compressed_device.compress(v, comp_config)
        else:
            k = TorchTensor.create_from_torch(k, self)
            v = TorchTensor.create_from_torch(v, self)


        return TorchTensor.create_from_torch(attn_output, self), k, v
    

    def mha(self, inputs, attention_mask, w_q, b_q, w_k, b_k, w_v, b_v,
            w_out, b_out, w_ln, b_ln, n_head, donate, compress_cache, comp_config):
        """Multi-head attention (prefill phase)."""
        # decompress weights
        if w_q.device.device_type == DeviceType.COMPRESSED:
            w_q = w_q.device.decompress(w_q)
            w_k = w_k.device.decompress(w_k)
            w_v = w_v.device.decompress(w_v)
            w_out = w_out.device.decompress(w_out)

        b, s, h = inputs.shape
        head_dim = h // n_head
        scaling = head_dim ** -0.5

        hidden = F.layer_norm(inputs.data, (h,), weight=w_ln.data, bias=b_ln.data)

        # shape: (b, s, h)
        q = F.linear(hidden, w_q.data, bias=b_q.data) * scaling
        k = F.linear(hidden, w_k.data, bias=b_k.data)
        v = F.linear(hidden, w_v.data, bias=b_v.data)
        # shape: (b, s, n_head, head_dim)
        q = q.view(b, s, n_head, head_dim)
        k = k.view(b, s, n_head, head_dim)
        v = v.view(b, s, n_head, head_dim)

        # shape: (b * n_head, s, head_dim)
        q = q.permute(0, 2, 1, 3).reshape(b * n_head, s, head_dim)
        # shape: (b * n_head, head_dim, s)
        k = k.permute(0, 2, 3, 1).reshape(b * n_head, head_dim, s)
        # shape: (b * n_head, s, head_dim)
        v = v.permute(0, 2, 1, 3).reshape(b * n_head, s, head_dim)

        # shape: (b * n_head, s, s)
        attn_weights = torch.bmm(q, k)

        # shape: (b, 1, s, s)
        idx = torch.arange(s, device=self.dev)
        causal_mask = (idx <= idx.view(s, 1)).view(1, 1, s, s)
        mask = attention_mask.data.view(b, 1, 1, s) & causal_mask

        # shape: (b, n_head, s, s)
        attn_weights = attn_weights.view(b, n_head, s, s)
        attn_weights = torch.where(mask, attn_weights, -1e4)
        attn_weights = attn_weights.view(b * n_head, s, s)
        attn_weights = F.softmax(attn_weights, dim=2)
        # shape: (b, n_head, s, head_dim)
        value = torch.bmm(attn_weights, v).view(b, n_head, s, head_dim)
        # shape: (b, s, h)
        value = value.transpose(1, 2).reshape(b, s, h)
        value = F.linear(value, w_out.data, bias=b_out.data)

        value.add_(inputs.data)

        if donate[0]: inputs.delete()
        if donate[1]: attention_mask.delete()

        # (s, b * n_head, head_dim) -> (s, b*head, headd)
        k = k.permute(2, 0, 1)
        v = v.permute(1, 0, 2)

        if compress_cache:
            k = self.compressed_device.compress(k, comp_config)
            v = self.compressed_device.compress(v, comp_config)
        else:
            k = TorchTensor.create_from_torch(k, self)
            v = TorchTensor.create_from_torch(v, self)

        return TorchTensor.create_from_torch(value, self), k, v

    def _gqa_value(self, 
                   q, k_new, v_new, 
                   k_cache, v_cache, 
                   attention_mask, 
                   layerno):
        """
        new q, k, v in shape (b, head, s, h)

        return : (b, head, s, h)
        """
        src_s = attention_mask.shape[-1]
        b, n_qhead, tgt_s, head_dim = q.shape
        n_kvhead = k_new.shape[1]
        k = extract_cache_to_bhsd(k_cache, b, n_kvhead, src_s, head_dim)
        v = extract_cache_to_bhsd(v_cache, b, n_kvhead, src_s, head_dim)
        
        # k = k_cache.data[:src_s].view(src_s, b, n_kvhead, head_dim).permute(1, 2, 0, 3)
        # v = v_cache.data[:src_s].view(src_s, b, n_kvhead, head_dim).permute(1, 2, 0, 3)

        dump_hidden(k, f"load-k", layerno, src_s-1)
    
        k[:, :, src_s-1:src_s, :] = k_new
        v[:, :, src_s-1:src_s, :] = v_new

        dump_hidden(k, "k", layerno)
        dump_hidden(v, "v", layerno)

        # mask: (B, S) -> (B, 1, 1, S)
        mask = attention_mask.data.view(b, 1, 1, -1)
        value = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, enable_gqa=True)
        return value

    def gqa_gen(self, inputs, attention_mask, position_embed, w_q, b_q, w_k, b_k, w_v, b_v,
                w_out, w_ln, n_head:tuple[int, int], k_cache, v_cache, donate,
                sparse_config, compress_cache, comp_config, layerno=None):
        """Multi-head attention (decoding phase)."""
        # decompress weights
        if w_q.device.device_type == DeviceType.COMPRESSED:
            w_q = w_q.device.decompress(w_q)
            w_k = w_k.device.decompress(w_k)
            w_v = w_v.device.decompress(w_v)
            w_out = w_out.device.decompress(w_out)

        b, tgt_s, h = inputs.shape
        src_s = attention_mask.shape[1]
        n_qhead, n_kvhead = n_head
        head_dim = h // n_qhead

        cos = position_embed.data[0][src_s-1:src_s]
        sin = position_embed.data[1][src_s-1:src_s]

        hidden = rms_layernorm(inputs.data, w_ln.data)

        # this part prepares new qkv for sdpa
        # -> (b, tgt_s, H)
        q = F.linear(hidden, w_q.data, bias=b_q.data) 
        k1 = F.linear(hidden, w_k.data, bias=b_k.data)
        v1 = F.linear(hidden, w_v.data, bias=b_v.data)
        # -> (b, tgt_s, head, h) -> (b, head, tgt-s, h)
        q =  q.view(    b, tgt_s, n_qhead,  head_dim).permute(0, 2, 1, 3)
        k1 = k1.view(   b, tgt_s, n_kvhead, head_dim).permute(0, 2, 1, 3)
        v1 = v1.view(   b, tgt_s, n_kvhead, head_dim).permute(0, 2, 1, 3)

        dump_hidden(k1, "k_new", layerno)
        dump_hidden(q, "q", layerno)

        q_pos  = apply_rope(q,       (cos, sin), unsqueeze_dim=0)
        k1_pos = apply_rope(k1,   (cos, sin), unsqueeze_dim=0)

        k1_cache = k1_pos
        v1_cache = v1

        dump_hidden(k1_pos, "k_pos", layerno)
        dump_hidden(q_pos, "q_pos", layerno)

        # gqa-gen-sparse
        if sparse_config.mode == "dense":
            # kv-cache
            # (src_s, b*head, h) -> (src_s, b, head, h) -> (b, head, src_s, h)
            # warning, position src_s-1 is for new kv
            # k = k_cache.data[:src_s-1].view(src_s-1, b, n_kvhead, head_dim).permute(1, 2, 0, 3)
            # v = v_cache.data[:src_s-1].view(src_s-1, b, n_kvhead, head_dim).permute(1, 2, 0, 3)
            attn_value = self._gqa_value(q_pos, k1_pos, v1, 
                                    k_cache, v_cache, 
                                    attention_mask, 
                                    layerno)

        else:
            attn_value = self._sparse_gqa_value(q_pos, k1_pos, v1, 
                                                k_cache, v_cache,
                                                attention_mask,
                                                sparse_config,
                                                layerno)
            
        dump_hidden(attn_value, "sdpa", layerno=layerno)

        # -> (b, s, head, h) -> (b, s, H)
        value = attn_value.permute(0, 2, 1, 3).view(b, tgt_s, h)
        value = F.linear(value, w_out.data, bias=None)

        value.add_(inputs.data)

        if donate[0]: inputs.delete()
        if donate[1]: attention_mask.delete()

        dump_hidden(k1_cache, "store-k", layerno, src_s-1)
        dump_hidden(v1_cache, "store-v", layerno, src_s-1)

        # (b, head, tgt_s, h) -> (tgt_s, b, head, h) -> (tgt_s, b*head, h)
        k1_cache = bhsd_to_cache_shape(k1_cache)
        v1_cache = bhsd_to_cache_shape(v1_cache)

        # cache compression related
        if compress_cache:
            assert 0
            if comp_config.group_dim == 0:
                s_ = src_s // comp_config.group_size * comp_config.group_size
                k1_cache = k[:, :, s_:].permute(2, 0, 1)
                v1_store = v[:, s_:, :].permute(1, 0, 2)
            k1_store = self.compressed_device.compress(k1_cache, comp_config)
            v1_store = self.compressed_device.compress(v1_store, comp_config)
        else:
            k1_cache = TorchTensor.create_from_torch(k1_cache, self)
            v1_cache = TorchTensor.create_from_torch(v1_cache, self)

        return TorchTensor.create_from_torch(value, self), k1_cache, v1_cache

    def mha_gen(self, inputs, attention_mask, w_q, b_q, w_k, b_k, w_v, b_v,
                w_out, b_out, w_ln, b_ln, n_head, k_cache, v_cache, donate,
                attn_sparsity, compress_cache, comp_config):
        """Multi-head attention (decoding phase)."""
        # decompress weights
        if w_q.device.device_type == DeviceType.COMPRESSED:
            w_q = w_q.device.decompress(w_q)
            w_k = w_k.device.decompress(w_k)
            w_v = w_v.device.decompress(w_v)
            w_out = w_out.device.decompress(w_out)

        b, tgt_s, h = inputs.shape
        src_s = attention_mask.shape[1]
        head_dim = h // n_head
        scaling = head_dim ** -0.5

        hidden = F.layer_norm(inputs.data, (h,), weight=w_ln.data, bias=b_ln.data)

        # shape: (b, 1, h)
        q = F.linear(hidden, w_q.data, bias=b_q.data) * scaling
        k = F.linear(hidden, w_k.data, bias=b_k.data)
        v = F.linear(hidden, w_v.data, bias=b_v.data)
        # shape: (b, 1, n_head, head_dim)
        q = q.view(b, tgt_s, n_head, head_dim)
        k = k.view(b, tgt_s, n_head, head_dim)
        v = v.view(b, tgt_s, n_head, head_dim)

        # shape: (b * n_head, 1, head_dim)
        q = q.permute(0, 2, 1, 3).reshape(b * n_head, tgt_s, head_dim)
        # shape: (1, b * n_head, head_dim)
        k_new = k.permute(1, 0, 2, 3).reshape(tgt_s, b * n_head, head_dim)
        # shape: (1, b * n_head, head_dim)
        v_new = v.permute(1, 0, 2, 3).reshape(tgt_s, b * n_head, head_dim)

        if isinstance(k_cache, TorchTensor):
            if attn_sparsity >= 1.0:  # Dense attention
                if compress_cache:
                    # shape: (s, b * n_head, head_dim)
                    k = k_cache.device.decompress(k_cache)[:src_s]
                    v = v_cache.device.decompress(v_cache)[:src_s]
                else:
                    # shape: (s, b * n_head, head_dim)
                    k = k_cache.data[:src_s]
                    v = v_cache.data[:src_s]
                k[src_s - 1:src_s] = k_new
                v[src_s - 1:src_s] = v_new

                # shape: (b * n_head, head_dim, s)
                k = k.permute(1, 2, 0).reshape(b * n_head, head_dim, src_s)
                # shape: (b * n_head, s, head_dim)
                v = v.permute(1, 0, 2).reshape(b * n_head, src_s, head_dim)

                if k.is_cuda:
                    value = self._attention_value(q, k, v, attention_mask.data,
                        b, src_s, tgt_s, n_head, head_dim)
                else:
                    q = q.float().cpu()
                    k, v = k.float(), v.float()
                    value = self._attention_value(q, k, v, attention_mask.data,
                        b, src_s, tgt_s, n_head, head_dim).cuda().half()
            else:  # Sparse attention
                # gen-sparse
                # shape: (s, b * n_head, head_dim)
                k = k_cache.data[:src_s]
                k[src_s - 1:src_s] = k_new
                # shape: (b * n_head, head_dim, s)
                k = k.permute(1, 2, 0).reshape(b * n_head, head_dim, src_s)

                if k.is_cuda:
                    value = self._sparse_attention_value(q, k, v_new, v_cache,
                        attention_mask.data, b, src_s, tgt_s, n_head, head_dim,
                        attn_sparsity)
                else:
                    q = q.float().cpu()
                    value = self._sparse_attention_value(q, k, v_new, v_cache,
                        attention_mask.data, b, src_s, tgt_s, n_head, head_dim,
                        attn_sparsity).cuda().half()
        else:  # Mixed device attention
            assert attn_sparsity >= 1.0
            value = self._mixed_device_attention(q, k_cache, v_cache,
                k_new, v_new, attention_mask.data, b, src_s, tgt_s,
                n_head, head_dim)

        # shape: (b, 1, h)
        value = value.transpose(1, 2).view(b, tgt_s, h)
        value = F.linear(value, w_out.data, bias=b_out.data)

        value.add_(inputs.data)

        if donate[0]: inputs.delete()
        if donate[1]: attention_mask.delete()

        if compress_cache:
            if comp_config.group_dim == 0:
                s_ = src_s // comp_config.group_size * comp_config.group_size
                k_new = k[:, :, s_:].permute(2, 0, 1)
                v_new = v[:, s_:, :].permute(1, 0, 2)
            k_new = self.compressed_device.compress(k_new, comp_config)
            v_new = self.compressed_device.compress(v_new, comp_config)
        else:
            k_new = TorchTensor.create_from_torch(k_new, self)
            v_new = TorchTensor.create_from_torch(v_new, self)

        return TorchTensor.create_from_torch(value, self), k_new, v_new

    def _attention_weights(self, q:torch.Tensor, k, mask, b, src_s, n_head):
        # calculate masked attn-score(include softmax)
        # scaling done, "k" is kt
        # shape: (b * n_head, 1, s)
        # return (b*head, 1, s)
        attn_weights = torch.bmm(q, k)
        # shape: (b, 1, 1, s)
        mask = mask.view(b, 1, 1, src_s)
        # shape: (b * n_head, 1, s)
        attn_weights = attn_weights.view(b, n_head, 1, src_s)
        attn_weights = torch.where(mask, attn_weights, -1e4)
        attn_weights = attn_weights.view(b * n_head, 1, src_s)
        attn_weights = F.softmax(attn_weights, dim=2)
        return attn_weights

    def _gqa_attn_weights(self, q:torch.Tensor, k:torch.Tensor, mask:TorchTensor):
        """
        q,k in shape (b, n_qhead, src_s, h)  
        ret: (b, nqhead, 1, src_s)
        """
        # b, n_qhead, tgt_s, head_dim = q.shape
        # _, n_kvhead, src_s, _ = k.shape[2]
        b, n_qhead, n_kvhead, tgt_s, src_s, head_dim = derive_shapes(q, k)
        
        k1 = k.repeat_interleave(n_qhead//n_kvhead, dim=1)
        
        if mask is not None:
            mask = mask.data.view(b, 1, 1, src_s)
        
        # -> (b*nqhead, src_s)
        q =  q.view(b*n_qhead, 1, head_dim)
        k = k1.view(b*n_qhead, src_s, head_dim).transpose(-2, -1)

        attn_weights = torch.bmm(q, k)
        attn_weights = attn_weights.view(b, n_qhead, 1, -1)

        if mask is not None:
            attn_weights = attn_weights.where(mask, -1e4)

        attn_weights = F.softmax(attn_weights, dim=-1)
        return attn_weights

    def _attention_value(self, q, k, v, mask, b, src_s, tgt_s, n_head, head_dim):
        # shape: (b * n_head, 1, s)
        attn_weights = self._attention_weights(q, k, mask, b, src_s, n_head)
        # shape: (b, n_head, 1, head_dim)
        return torch.bmm(attn_weights, v).view(b, n_head, tgt_s, head_dim)

    def _sparse_attention_value(self, q, k, v_new, v_cache, mask, b,
                                src_s, tgt_s, n_head, head_dim, attn_sparsity):
        # shape: (b * n_head, 1, s)
        attn_weights = self._attention_weights(q, k, mask, b, src_s, n_head)
        topk = int(attn_sparsity * (attn_weights.shape[2] - 1))
        topk_weights, topk_indices = attn_weights[:, :, :-1].topk(
            topk, dim=2, sorted=False)
        topk_indices = topk_indices.view(b * n_head, topk).transpose(0, 1)
        # shape: (b * n_head, 1, topk+1)
        attn_weights = torch.cat([topk_weights,
            attn_weights[:, :, -1].unsqueeze(-1)], dim=-1)

        if k.is_cuda:
            v_home = v_cache
            v_buf = self.allocate((topk+1, b*n_head, head_dim), np.float16)
            topk_indices = topk_indices.cpu()
        else:
            (v_home, v_buf) = v_cache

        # shape: (s, b * n_head, head_dim)
        indices_src = topk_indices
        indices_tgt = (slice(0, indices_src.shape[0]), slice(0, v_home.shape[1]))
        general_copy(v_buf, indices_tgt, v_home, indices_src)
        v_home.device.synchronize()

        # shape: (topk+1, b * n_head, head_dim)
        v = v_buf.data[:topk+1]
        v[topk:topk+1] = v_new
        # shape: (b * n_head, topk+1, head_dim)
        v = v.permute(1, 0, 2).reshape(b * n_head, topk+1, head_dim)

        # shape: (b * n_head, 1, head_dim)
        return torch.bmm(attn_weights, v).view(b, n_head, tgt_s, head_dim)


    def _naive_sparse_gqa_value(self, q, k_new, v_new, 
                                      k_cache, v_cache, 
                                      mask, sparsity_config, 
                                      layerno):

        b, n_qhead, n_kvhead, tgt_s, src_s, head_dim = derive_shapes(q, k_new, mask)

        group_size = n_qhead // n_kvhead
        #get kcache
        k = extract_cache_to_bhsd(k_cache, b, n_kvhead, src_s, head_dim)
        
        # attn-score    (b, qhead, 1, s)
        attn_weights = self._gqa_attn_weights(q, k, mask)
        
        topk = int(sparsity_config.sparsity * (src_s - 1))
        topk_weights, topk_indices = attn_weights[:, :, :, :-1].topk(
            topk, dim=-1, sorted=False)
        # (b, head, 1, k) -> (b* ngroup, group_head* k)
        # topk_indices = topk_indices.view(b * n_kvhead, group_size * topk).T

        # shape: (b, n_head, 1, topk+1)
        attn_weights = torch.cat([topk_weights,
            attn_weights[:, :, :, -1:]], dim=-1)
    
        # temporary on cuda, using only
        v = extract_cache_to_bhsd(v_cache, b, n_kvhead, src_s, head_dim)

        # (b, head, k, d)
        v = select_topk(v, topk_indices)

        # (b, qhead, k+1, d)
        v_new = v_new.repeat_interleave(group_size, dim=1)
        v = torch.concat([v, v_new], dim=2)

        attn_out = torch.bmm(
            attn_weights.view(b*n_qhead, tgt_s, topk+1),
            v.view(b*n_qhead, topk+1, head_dim)
        )
        attn_out = attn_out.view(b, n_qhead, tgt_s, head_dim)
        
        return attn_out

    def _block_sparse_gqa_value(self, q, k_new, v_new, k_cache, v_cache, k_summary, mask, attn_sparsity, layerno, sparse_config):
        # get summary in some way
        #   from parameters?

        block_weights = _gqa_attn_weights(q, k_cache, mask=None)

        # selected tokens (idx) in shape (b, qhead, tgt_s, topk)
        selected_tokens = expand_block_idx(block_weights, sparse_config.block_size)
        
        # emit load task
        seleceted_kv_shape = (b, qhead, tgt_s, topk)
        selected_k = self.allocate(selected_shape, dtype=k_new.dtype)
        selected_v = self.allocate(selected_shape, dtype=k_new.dtype)
        
        selected_k = general_copy(selected_k, None, k_cache, selected_tokens)
        selected_v = general_copy(selected_v, None, k_cache, selected_tokens)

        F.scaled_dot_product_attention(q, selected_k, selected_v)
        

    def _sparse_gqa_value(self, q, k_new, v_new, 
                          k_cache, v_cache, 
                          mask, 
                          sparse_config, layerno, k_summary=None):
        """
        q, k, v: (b, head, s, h)
        """
        if sparse_config.mode == "naive":
            return self._naive_sparse_gqa_value(q, k_new, v_new, k_cache, v_cache, mask, sparse_config, layerno)
        elif sparse_config.mode == "block":
            assert k_summary is not None

    def _mixed_device_attention(self, q, k_cache, v_cache, k_new, v_new,
            mask, b, src_s, tgt_s, n_head, head_dim):
        # The caches are stored on both gpu and cpu.
        # Compute attention on gpu for caches stored on gpu.
        # Compute attention on cpu for caches stored on cpu.
        k_gpu, k_cpu = k_cache[0].data, k_cache[1].data
        v_gpu, v_cpu = v_cache[0].data, v_cache[1].data
        seg = k_gpu.shape[1]

        # Compute GPU part
        b_gpu = seg // n_head
        q_gpu = q[:seg]
        # shape: (s, b * n_head, head_dim)
        k_gpu = k_gpu[:src_s, :seg, :]
        v_gpu = v_gpu[:src_s, :seg, :]
        k_gpu[src_s-1:src_s, :, :] = k_new[:, :seg, :]
        v_gpu[src_s-1:src_s, :, :] = v_new[:, :seg, :]
        # shape: (b * n_head, head_dim, s)
        k_gpu = k_gpu.permute(1, 2, 0)
        # shape: (b * n_head, s, head_dim)
        v_gpu = v_gpu.permute(1, 0, 2)

        mask_gpu = mask[:b_gpu].cuda()
        value_gpu = self._attention_value(q_gpu, k_gpu, v_gpu, mask_gpu,
            b_gpu, src_s, tgt_s, n_head, head_dim)

        # Compute CPU Part
        b_cpu = b - b_gpu
        q_cpu = q[seg:].float().cpu()
        # shape: (s, b * n_head, head_dim)
        k_cpu = k_cpu[:src_s, seg:, :]
        v_cpu = v_cpu[:src_s, seg:, :]
        k_cpu[src_s-1:src_s, :, :] = k_new[:, seg:, :]
        v_cpu[src_s-1:src_s, :, :] = v_new[:, seg:, :]
        # shape: (b * n_head, head_dim, s)
        k_cpu = k_cpu.permute(1, 2, 0)
        # shape: (b * n_head, s, head_dim)
        v_cpu = v_cpu.permute(1, 0, 2)

        mask_cpu = mask[b_gpu:]
        value_cpu = self._attention_value(q_cpu, k_cpu, v_cpu, mask_cpu,
            b_cpu, src_s, tgt_s, n_head, head_dim)

        value = torch.cat([value_gpu, value_cpu.cuda().half()], dim=0)
        return value

    def mlp(self, inputs, wi, bi, wo, bo, w_ln, b_ln, donate):
        # decompress weights
        if wi.device.device_type == DeviceType.COMPRESSED:
            wi = wi.device.decompress(wi)
            wo = wo.device.decompress(wo)

        b, s, h = inputs.shape

        out = F.layer_norm(inputs.data, (h,), weight=w_ln.data, bias=b_ln.data)
        out = F.linear(out, wi.data, bias=bi.data)
        F.relu(out, inplace=True)
        out = F.linear(out, wo.data, bias=bo.data)

        out.add_(inputs.data)
        if donate[0]: inputs.delete()
        return TorchTensor.create_from_torch(out, self)

    def qwen_mlp(self, inputs, w_gate, w_up, w_down, w_ln, donate, layerno):
        # decompress weights
        if w_gate.device.device_type == DeviceType.COMPRESSED:
            w_gate = w_gate.device.decompress(w_gate)
            w_up = w_up.device.decompress(w_up)
            w_down = w_down.device.decompress(w_down)

        b, s, h = inputs.shape

        global mlp_layerno

        dump_hidden(inputs.data, "mlp-input", layerno)

        norm = rms_layernorm(inputs.data, w_ln.data)

        gate = F.linear(norm, w_gate.data, bias=None)

        gate = F.silu(gate, inplace=True)
        
        up = F.linear(norm, w_up.data, bias=None)

        out = gate * up
        
        out = F.linear(out, w_down.data, bias=None)

        out.add_(inputs.data)


        if donate[0]: inputs.delete()
        
        return TorchTensor.create_from_torch(out, self)

    def synchronize(self):
        torch.cuda.synchronize()

    def mem_stats(self):
        if self.device_type == DeviceType.CUDA:
            cur_mem = torch.cuda.memory_allocated(self.dev)
            peak_mem = torch.cuda.max_memory_allocated(self.dev)
        elif self.device_type == DeviceType.CPU:
            cur_mem = cpu_mem_stats()
            peak_mem = 0
        else:
            raise NotImplementedError()

        return cur_mem, peak_mem

    def print_stats(self, output_file=None):
        torch.cuda.synchronize()
        cur_mem, peak_mem = self.mem_stats()

        if output_file is not None:
            with open(output_file, "w") as f:
                f.write(f"TorchDevice: {self.name}\n")
                f.write(f"  cur_mem: {cur_mem/GB:.4f} GB, "
                        f" peak_mem: {peak_mem/GB:.4f} GB\n")
        else:
            print(f"TorchDevice: {self.name}")
            print(f"  cur_mem: {cur_mem/GB:.4f} GB, "
                  f" peak_mem: {peak_mem/GB:.4f} GB")

        return cur_mem, peak_mem

    def __str__(self):
        return f"TorchDevice(name={self.name})"


class TorchDisk:
    """Manage tensors stored on a disk."""

    def __init__(self, path, mem_capacity=None, cuda_id=0, num_copy_threads=4):
        self.name = path
        self.path = os.path.abspath(os.path.expanduser(path))
        self.mem_capacity = mem_capacity

        self.device_type = DeviceType.DISK
        self.compressed_device = TorchCompressedDevice(self)

        if os.path.exists(self.path):
            assert os.path.isdir(self.path)
        else:
            os.makedirs(self.path)

        self.links = {}

        # Copy threads
        self.copy_queue = queue.Queue()
        self.copy_threads = [
            threading.Thread(
                target=copy_worker_func, args=(self.copy_queue, cuda_id)
            ) for _ in range(num_copy_threads)
        ]
        for t in self.copy_threads:
            t.start()

        global global_disk_device
        global_disk_device = self

    def add_link(self, link):
        dst = link.b if link.a == self else link.a
        self.links[dst] = link

    def allocate(self, shape, dtype, pin_memory=None, name=None):
        name = name or TorchTensor.next_name()
        path = os.path.join(self.path, name)
        np.lib.format.open_memmap(path, mode="w+", shape=shape, dtype=dtype)
        return TorchTensor(shape, np_dtype_to_torch_dtype[dtype],
                           path, self, name=name)

    def delete(self, tensor):
        if os.path.exists(tensor.data) and tensor.delete_file:
            os.remove(tensor.data)

    def init_cache_one_gpu_batch(self, config, task, policy):
        n_qhead, n_kvhead, hidden_size, prompt_len, gen_len, gpu_batch_size = (
            config.num_attention_heads, config.num_key_value_heads, config.hidden_size, task.prompt_len, task.gen_len,
            policy.gpu_batch_size)
        
        shape = (prompt_len + gen_len - 1, gpu_batch_size * n_kvhead, hidden_size // n_qhead)

        k_cache = self.allocate(shape, np.float16)
        v_cache = self.allocate(shape, np.float16)
        return k_cache, v_cache

    def submit_copy(self, *args):
        self.copy_queue.put_nowait(args)

    def synchronize(self):
        self.copy_queue.join()

    def close_copy_threads(self):
        for _ in range(len(self.copy_threads)):
            self.copy_queue.put_nowait(None)
        for t in self.copy_threads:
            t.join()
        self.copy_queue.join()
        self.copy_queue = None

    def mem_stats(self):
        raise NotImplementedError()

    def print_stats(self):
        raise NotImplementedError()

    def __del__(self):
        if self.copy_queue:
            self.close_copy_threads()


# Segment dimension for tensors stored on TorchMixedDevice
SEG_DIM = 1

class TorchMixedDevice:
    """Manage tensors stored on multiple physical devices."""

    def __init__(self, base_devices):
        self.name = "mixed"
        self.device_type = DeviceType.MIXED
        self.base_devices = base_devices

    def allocate(self, shape, dtype, seg_lengths, pin_memory=None, name=None):
        assert sum(seg_lengths) == shape[SEG_DIM]
        assert len(seg_lengths) == len(self.base_devices)
        seg_points = [0]
        for l in seg_lengths:
            seg_points.append(seg_points[-1] + l)

        devices = self.base_devices
        tensors = []
        for i in range(len(devices)):
            seg_len = seg_points[i+1] - seg_points[i]
            if seg_len == 0:
                tensors.append(None)
            else:
                seg_shape = shape[:SEG_DIM] + (seg_len,) + shape[SEG_DIM+1:]
                tensors.append(devices[i].allocate(seg_shape, dtype,
                    pin_memory=pin_memory))

        return TorchTensor(shape, np_dtype_to_torch_dtype[dtype],
                           (tensors, seg_points), self, name=name)

    def delete(self, tensor):
        for x in self.tensor.data[0]:
            if x:
                x.delete()

    def init_cache_one_gpu_batch(self, config, task, policy):
        # num_head, hidden_size, prompt_len, gen_len, gpu_batch_size = (
        #     config.n_head, config.input_dim, task.prompt_len, task.gen_len,
        #     policy.gpu_batch_size)
        # shape = (prompt_len + gen_len - 1, gpu_batch_size * num_head, hidden_size // num_head)
        shape = cache_shape(config, task, policy)

        # We have to round to a multiple of `num_head`
        if policy.cache_disk_percent == 0:
            len_gpu = int(shape[SEG_DIM] * policy.cache_gpu_percent / 100) // num_head * num_head
            len_cpu = shape[SEG_DIM]  - len_gpu
            len_disk = 0
        else:
            len_gpu = int(shape[SEG_DIM] * policy.cache_gpu_percent / 100) // num_head * num_head
            len_cpu = int(shape[SEG_DIM] * policy.cache_cpu_percent / 100) // num_head * num_head
            len_disk = shape[SEG_DIM] - len_gpu - len_cpu
        lens = [len_gpu, len_cpu, len_disk]

        pin_memory = False
        k_cache = self.allocate(shape, np.float16,
            seg_lengths=lens, pin_memory=pin_memory)
        v_cache = self.allocate(shape, np.float16,
            seg_lengths=lens, pin_memory=pin_memory)
        return k_cache, v_cache


class TorchLink:
    """An I/O link between two devices."""

    def __init__(self, a, b, a_to_b_bandwidth, b_to_a_bandwidth):
        self.a = a
        self.b = b
        self.a_to_b_bandwidth = a_to_b_bandwidth
        self.b_to_a_bandwidth = b_to_a_bandwidth

        a.add_link(self)
        b.add_link(self)

    def io_time(self, src, dst, size):
        if src == self.a:
            assert dst == self.b
            bandwidth = self.a_to_b_bandwidth
        elif src == self.b:
            assert dst == self.a
            bandwidth = self.b_to_a_bandwidth
        else:
            raise ValueError(f"Invalid source {src}")

        if force_io_time is not None:
            return force_io_time

        return size / bandwidth


def general_copy(dst: TorchTensor, dst_indices: Tuple,
                 src: TorchTensor, src_indices: Tuple):
    """Launch a general asynchronous copy between two tensors.
    It is equivalent to `dst[dst_indices] = src[src_indices]` in numpy syntax.
    The copy is asynchronous. To wait for the copy to complete, you need to call
    >>> env.disk.synchronize()
    >>> torch.cuda.synchronize()
    """
    if dst.device.device_type == DeviceType.MIXED:
        # The tensor is on mixed devices, do recursive calls
        assert src.device.device_type != DeviceType.MIXED
        seg_points = dst.data[1]

        for i in range(len(dst.device.base_devices)):
            if seg_points[i] == seg_points[i+1]:
                continue
            src_indices = src_indices or tuple(slice(0, x) for x in src.shape)
            dst_indices = dst_indices or tuple(slice(0, x) for x in dst.shape)
            tmp_src_indices = cut_indices(src_indices, seg_points[i], seg_points[i+1])
            tmp_dst_indices = cut_indices(dst_indices, seg_points[i], seg_points[i+1],
                base=seg_points[i])
            general_copy(dst.data[0][i], tmp_dst_indices, src, tmp_src_indices)
    elif src.device.device_type == DeviceType.MIXED:
        # The tensor is on mixed devices, do recursive calls
        assert dst.device.device_type != DeviceType.MIXED
        seg_points = src.data[1]

        for i in range(len(src.device.base_devices)):
            if seg_points[i] == seg_points[i+1]:
                continue
            src_indices = src_indices or tuple(slice(0, x) for x in src.shape)
            dst_indices = dst_indices or tuple(slice(0, x) for x in dst.shape)
            tmp_src_indices = cut_indices(src_indices, seg_points[i], seg_points[i+1],
                base=seg_points[i])
            tmp_dst_indices = cut_indices(dst_indices, seg_points[i], seg_points[i+1])
            general_copy(dst, tmp_dst_indices, src.data[0][i], tmp_src_indices)
    elif (src.device.device_type == DeviceType.COMPRESSED or
          dst.device.device_type == DeviceType.COMPRESSED):
        # The tensor is compressed, do recursive calls
        general_copy_compressed(dst, dst_indices, src, src_indices)
    elif src.device.device_type == DeviceType.DISK:
        # The tensor is on the disk, dispatch to copy threads for asynchronous copy
        src.device.submit_copy(dst, dst_indices, src, src_indices)
    elif dst.device.device_type == DeviceType.DISK:
        # The tensor is on the disk, dispatch to copy threads for asynchronous copy
        dst.device.submit_copy(dst, dst_indices, src, src_indices)
    elif (src.device.device_type == DeviceType.CUDA and
          dst.device.device_type == DeviceType.CPU and
          not dst.data.is_pinned() and src.shape[0] > 1):
        # The cpu tensor is not pinned, dispatch to copy threads and use pin_memory
        # as a relay
        global_disk_device.submit_copy(dst, dst_indices, src, src_indices)
    elif (src.device.device_type == DeviceType.CPU and
          dst.device.device_type == DeviceType.CUDA and
          not src.data.is_pinned()):
        # The cpu tensor is not pinned, use pin_memory as a relay
        src = index_into(src.data, src_indices) if src_indices is not None else src.data
        dst = index_into(dst.data, dst_indices) if dst_indices is not None else dst.data
        src = src.pin_memory()
        dst.copy_(src, non_blocking=True)
    else:
        # The normal path
        src = index_into(src.data, src_indices) if src_indices is not None else src.data

        dst = index_into(dst.data, dst_indices) if dst_indices is not None else dst.data

        dst.copy_(src, non_blocking=True)


def cut_indices(indices, start, stop, base=0):
    assert all(x.step is None for x in indices)
    seg = indices[SEG_DIM]
    return (indices[:SEG_DIM] +
            (slice(max(seg.start, start) - base, min(seg.stop, stop) - base),) +
            indices[SEG_DIM + 1:])


def map_to_torch_tensor(tensor, indices):
    if tensor.device.device_type == DeviceType.DISK:
        data = torch.from_numpy(np.lib.format.open_memmap(tensor.data))
    else:
        data = tensor.data

    # BC: this is supposed to only handle the sparse v_cache case
    if torch.is_tensor(indices):
        return vector_gather(data, indices)
    return data[indices] if indices else data


def copy_worker_func(queue, cuda_id):
    """The copy worker thread."""
    torch.cuda.set_device(cuda_id)

    cpu_buf = torch.empty((1 * GB,), dtype=torch.float16, pin_memory=True)
    copy_stream = torch.cuda.Stream()

    with torch.cuda.stream(copy_stream):
        while True:
            item = queue.get()
            if item is None:
                queue.task_done()
                return

            dst, dst_indices, src, src_indices = item
            src_data = map_to_torch_tensor(src, src_indices)
            dst_data = map_to_torch_tensor(dst, dst_indices)

            if (src.device.device_type == DeviceType.CUDA or
                dst.device.device_type == DeviceType.CUDA):
                # Use a pinned cpu buffer as a relay
                size = np.prod(src_data.shape)
                tmp_cpu_buf = cpu_buf[:size].view(src_data.shape)
                tmp_cpu_buf.copy_(src_data)
                dst_data.copy_(tmp_cpu_buf)
            else:
                dst_data.copy_(src_data)

            queue.task_done()
