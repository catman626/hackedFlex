"""
The OPT model configurations and weight downloading utilities.

Some functions are adopted from https://github.com/alpa-projects/alpa/tree/main/examples/llm_serving/model.
"""

import argparse
import dataclasses
import glob
import os

import numpy as np
from tqdm import tqdm

from safetensors.torch import load_file, load, save_file


@dataclasses.dataclass(frozen=True)
class QwenConfig :
    name:str="Qwen/Qwen2-0.5B"
    hidden_size :int = 896
    num_attention_heads :int = 14
    num_key_value_heads :int = 2

    intermediate_size :int = 4864
    max_position_embeddings :int = 131072
    num_hidden_layers :int = 24
    vocab_size :int = 151936
    rms_norm_eps = 1e-6
    rope_theta = 1000000.0
    bos_token_id = 151643
    eos_token_id = 151643
    pad_token_id = 151643
    dtype = np.float32

    def model_bytes(self) -> int:
        h, n_layers = self.hidden_size, self.num_hidden_layers
        head_dim = h // self.num_attention_heads
        
        # 嵌入层参数（词嵌入）
        embed = self.vocab_size * h
        # 每层注意力参数（QKV投影+输出投影+RMSNorm）
        attn = h*h + h*(self.num_key_value_heads * head_dim * 2) + h*h + h
        # 每层MLP参数（SwiGLU双投影+输出投影+RMSNorm）
        mlp = h * (self.intermediate_size * 2) + self.intermediate_size * h + h
        
        return (embed + n_layers * (attn + mlp)) * 2  # ×2对应float16字节数

    def cache_bytes(self, batch_size: int, seq_len: int) -> int:
        head_dim = self.hidden_size // self.num_attention_heads
        # 每组KV缓存（K+V各一份）× 层数 × 字节数
        return batch_size * self.num_key_value_heads * seq_len * head_dim * 2 * self.num_hidden_layers * 2

    def hidden_bytes(self, batch_size: int, seq_len: int) -> int:
        # 隐藏状态张量（batch×seq×hidden）× 字节数
        return batch_size * seq_len * self.hidden_size * 2


def get_qwen_config(name):
    if "/" in name:
        name = name.split("/")[-1]
    name = name.lower()

    if name == "qwen2-0.5b" or name == "qwen2-0.5b-instruct":
        return QwenConfig(name=name)
    elif name == "qwen2-7b":
        config = QwenConfig(
            name=name, hidden_size=3584, intermediate_size=18944,
            num_attention_heads = 14, num_hidden_layers = 24, num_key_value_heads = 2,
            vocab_size=152064
        )
        return config
    else:
        assert 0, f" >>> invalid model name: {name}"


global torch_linear_init_backup
global torch_layer_norm_init_backup


def disable_torch_init():
    """
    Disable the redundant torch default initialization to accelerate model creation.
    """
    import torch
    global torch_linear_init_backup
    global torch_layer_norm_init_backup

    torch_linear_init_backup = torch.nn.Linear.reset_parameters
    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)

    torch_layer_norm_init_backup = torch.nn.LayerNorm.reset_parameters
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)


def restore_torch_init():
    """Rollback the change made by disable_torch_init."""
    import torch
    setattr(torch.nn.Linear, "reset_parameters", torch_linear_init_backup)
    setattr(torch.nn.LayerNorm, "reset_parameters", torch_layer_norm_init_backup)


def disable_hf_opt_init():
    """
    Disable the redundant default initialization to accelerate model creation.
    """
    import transformers

    setattr(transformers.models.opt.modeling_opt.OPTPreTrainedModel,
            "_init_weights", lambda *args, **kwargs: None)


model_to_cache = {
    "qwen2-0.5b": "/home/llmserver/.cache/huggingface/hub/models--Qwen--Qwen2-0.5B/snapshots/91d2aff3f957f99e4c74c962f2f408dcc88a18d8",
    "qwen2-7b": "/home/llmserver/.cache/huggingface/hub/models--Qwen--Qwen2-7B/snapshots/453ed1575b739b5b03ce3758b23befdb0967f40e",
    "qwen2-0.5b-instruct": "/home/llmserver/.cache/huggingface/hub/models--Qwen--Qwen2-0.5B-Instruct/snapshots/c540970f9e29518b1d8f06ab8b24cba66ad77b6d"
}
def convert_qwen_weights(model_name, path):
    
    # the seperated weights will be placed at path/model_name-np/
    huggingface_cache = model_to_cache[model_name]
    
    safetensor_files = glob.glob(os.path.join(huggingface_cache, "*.safetensors"))

    seperated_weight_dir = os.path.join(path, f"{model_name}-np")
    seperated_weight_dir = os.path.expanduser(seperated_weight_dir)
    os.makedirs(seperated_weight_dir, exist_ok=True)

    import torch

    print(f" >>> seperated weight dir: {seperated_weight_dir}")
    for safetensor_file in tqdm(safetensor_files, desc="Convert format"):
        state = load_file(safetensor_file)
        for name, param in tqdm(state.items(), leave=False):
            name = name.replace("model.", "")
            # name = name.replace("decoder.final_layer_norm", "decoder.layer_norm")
            param_path = os.path.join(seperated_weight_dir, name)
            with open(param_path, "wb") as f:
                np.save(f, param.cpu().detach().to(torch.float32).numpy())

    
            with open("parameter_names.txt", "+a") as f:
                f.write(f"{name}\n")
            # shared embedding
            # if "decoder.embed_tokens.weight" in name:
            #     shutil.copy(param_path, param_path.replace(
            #         "decoder.embed_tokens.weight", "lm_head.weight"))

    seperated_weights = os.listdir(seperated_weight_dir)
    for w in seperated_weights:
        print(w)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)
    parser.add_argument("--path", type=str, default="opt_weights")
    args = parser.parse_args()

    convert_qwen_weights(args.model, args.path)
