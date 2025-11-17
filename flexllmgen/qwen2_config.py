"""
The OPT model configurations and weight downloading utilities.

Some functions are adopted from https://github.com/alpa-projects/alpa/tree/main/examples/llm_serving/model.
"""

import argparse
import dataclasses
import glob
import os
import shutil

import numpy as np
from tqdm import tqdm

from safetensors.torch import load_file, load, save_file
import torch


@dataclasses.dataclass(frozen=True)
class QwenConfig :
    name="Qwen/Qwen2-0.5B"
    vocab_size = 151936
    hidden_size = 896
    num_attention_heads = 14
    num_key_value_heads = 2
    intermediate_size = 4864
    max_position_embeddings = 131072
    rms_norm_eps = 1e-6
    rope_theta = 1000000.0
    bos_token_id = 151643
    eos_token_id = 151643
    pad_token_id = 151643
    num_hidden_layers = 24
    dtype = np.float32
    n_qhead = 14
    n_kvhead = 2

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
    assert name == "Qwen/Qwen2-0.5B"
    return QwenConfig()

def get_opt_config(name, **kwargs):
    if "/" in name:
        name = name.split("/")[1]
    name = name.lower()

    # Handle opt-iml-30b and opt-iml-max-30b
    if "-iml-max" in name:
        arch_name = name.replace("-iml-max", "")
    elif "-iml" in name:
        arch_name = name.replace("-iml", "")
    else:
        arch_name = name

    if arch_name == "opt-125m":
        config = OptConfig(name=name,
            max_seq_len=2048, num_hidden_layers=12, n_head=12,
            hidden_size=768, input_dim=768, ffn_embed_dim=768 * 4,
        )
    elif arch_name == "opt-350m":
        config = OptConfig(name=name,
            max_seq_len=2048, num_hidden_layers=24, n_head=16,
            hidden_size=1024, input_dim=1024, ffn_embed_dim=1024 * 4,
        )
        raise NotImplementedError("Not implemented because this model "
                                  "has a different architecture")
    elif arch_name == "opt-1.3b":
        config = OptConfig(name=name,
            max_seq_len=2048, num_hidden_layers=24, n_head=32,
            hidden_size=2048, input_dim=2048, ffn_embed_dim=2048 * 4,
        )
    elif arch_name == "opt-2.7b":
        config = OptConfig(name=name,
            max_seq_len=2048, num_hidden_layers=32, n_head=32,
            hidden_size=2560, input_dim=2560, ffn_embed_dim=2560 * 4,
        )
    elif arch_name == "opt-6.7b":
        config = OptConfig(name=name,
            max_seq_len=2048, num_hidden_layers=32, n_head=32,
            hidden_size=4096, input_dim=4096, ffn_embed_dim=4096 * 4,
        )
    elif arch_name == "opt-13b":
        config = OptConfig(name=name,
            max_seq_len=2048, num_hidden_layers=40, n_head=40,
            hidden_size=5120, input_dim=5120, ffn_embed_dim=5120 * 4,
        )
    elif arch_name == "opt-30b":
        config = OptConfig(name=name,
            max_seq_len=2048, num_hidden_layers=48, n_head=56,
            hidden_size=7168, input_dim=7168, ffn_embed_dim=7168 * 4,
        )
    elif arch_name == "galactica-30b":
        config = OptConfig(name=name,
            max_seq_len=2048, num_hidden_layers=48, n_head=56,
            hidden_size=7168, input_dim=7168, ffn_embed_dim=7168 * 4, vocab_size=50000,
        )
    elif arch_name == "opt-66b":
        config = OptConfig(name=name,
            max_seq_len=2048, num_hidden_layers=64, n_head=72,
            hidden_size=9216, input_dim=9216, ffn_embed_dim=9216 * 4,
        )
    elif arch_name == "opt-175b":
        config = OptConfig(name=name,
            max_seq_len=2048, num_hidden_layers=96, n_head=96,
            hidden_size=12288, input_dim=12288, ffn_embed_dim=12288 * 4,
        )
    elif arch_name == "opt-175b-stage":
        config = OptConfig(name=name,
            max_seq_len=2048, num_hidden_layers=24, n_head=96,
            hidden_size=12288, input_dim=12288, ffn_embed_dim=12288 * 4,
        )
    else:
        raise ValueError(f"Invalid model name: {name}")

    return dataclasses.replace(config, **kwargs)


def download_opt_weights_old(model_name, path):
    """Download weights from huggingface."""
    import torch
    from transformers import OPTForCausalLM, BloomForCausalLM

    if "/" in model_name:
        model_name = model_name.split("/")[1].lower()
    path = os.path.join(path, f"{model_name}-np")
    path = os.path.abspath(os.path.expanduser(path))

    if "opt" in model_name:
        hf_model_name = "facebook/" + model_name
        model_class = OPTForCausalLM
    elif "bloom" in model_name:
        hf_model_name = "bigscience/" + model_name
        model_class = BloomForCausalLM
    elif "galactica" in model_name:
        hf_model_name = "facebook/" + model_name
    else:
        raise ValueError("Invalid model name: {model_name}")

    print(f"Load the pre-trained pytorch weights of {model_name} from huggingface. "
          f"The downloading and cpu loading can take dozens of minutes. "
          f"If it seems to get stuck, you can monitor the progress by "
          f"checking the memory usage of this process.")

    disable_torch_init()
    model = model_class.from_pretrained(hf_model_name, torch_dtype=torch.float16,
                                        _fast_init=True)
    restore_torch_init()

    os.makedirs(path, exist_ok=True)

    print(f"Convert the weights to numpy format under {path} ...")
    if "opt" in model_name:
        for name, param in tqdm(list(model.model.named_parameters())):
            name = name.replace("decoder.final_layer_norm", "decoder.layer_norm")
            param_path = os.path.join(path, name)
            with open(param_path, "wb") as f:
                np.save(f, param.cpu().detach().numpy())
    elif "galactica" in model_name:
        for name, param in tqdm(list(model.model.named_parameters())):
            name = name.replace("decoder.final_layer_norm", "decoder.layer_norm")
            param_path = os.path.join(path, name)
            with open(param_path, "wb") as f:
                np.save(f, param.cpu().detach().numpy())
    elif "bloom" in model_name:
        for name, param in tqdm(list(model.transformer.named_parameters())):
            param_path = os.path.join(path, name)
            with open(param_path, "wb") as f:
                np.save(f, param.cpu().detach().numpy())
    else:
        raise ValueError("Invalid model name: {model_name}")


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


def convert_qwen_weights(model_name, path):
    # the seperated weights will be placed at path/model_name-np/
    huggingface_cache = "/home/llmserver/.cache/huggingface/hub/models--Qwen--Qwen2-0.5B/snapshots/91d2aff3f957f99e4c74c962f2f408dcc88a18d8"
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
                # torch.save(param.cpu().detach(), f)

                # save_file(param.cpu().detach(), f)
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
