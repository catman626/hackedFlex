"""Complete sentences with FlexLLMGen and OPT models."""
import argparse

from flexllmgen.flex_opt import (Policy, OptLM, ExecutionEnv, CompressionConfig,
        str2bool)
from transformers import AutoTokenizer
from flexllmgen.opt_config import OptConfig, get_opt_config
from flexllmgen.utils import  GB, T

def get_prompts(prompt_file):
    if prompt_file.endswith("txt"):
        # represent only one query
        prompts = open(prompt_file).read()
        prompts = [ prompts] 
    elif prompt_file.endswith("json"):
        import json
        jsonfile = json.load(open(prompt_file))
        prompts = [ j["prompt"] for j in jsonfile]

    return prompts


def main(args):
    # Prompts
    prompts = [
        "Question: Where were the 2004 Olympics held?\n"
        "Answer: Athens, Greece\n"
        "Question: What is the longest river on the earth?\n"
        "Answer:",

        "Extract the airport codes from this text.\n"
        "Text: \"I want a flight from New York to San Francisco.\"\n"
        "Airport codes: JFK, SFO.\n"
        "Text: \"I want you to book a flight from Phoenix to Las Vegas.\"\n"
        "Airport codes:",
    ]

    # prompts = get_prompts(args.input_file)

    # Initialize environment
    env = ExecutionEnv.create(args.offload_dir)

    # Offloading policy
    policy = Policy(len(prompts), 1,
                    args.percent[0], args.percent[1],
                    args.percent[2], args.percent[3],
                    args.percent[4], args.percent[5],
                    overlap=True, sep_layer=True, pin_weight=args.pin_weight,
                    cpu_cache_compute=args.cpu_cache_compute, attn_sparsity=1.0,
                    compress_weight=args.compress_weight,
                    comp_weight_config=CompressionConfig(
                        num_bits=4, group_size=64,
                        group_dim=0, symmetric=False),
                    compress_cache=args.compress_cache,
                    comp_cache_config=CompressionConfig(
                        num_bits=4, group_size=64,
                        group_dim=2, symmetric=False))

    # Model
    print("Initialize...")
    
    tokenizer = AutoTokenizer.from_pretrained("facebook/opt-30b", padding_side="left")
    tokenizer.add_bos_token = False
    stop = tokenizer("\n").input_ids[0]

    model = OptLM(args.model, env, args.path, policy)

    # Generate
    print("Generate...")
    inputs = tokenizer(prompts, padding="longest")
    print(f" >>> input length padded to {len(inputs.input_ids[0])}")

    prompt_len = len(inputs[0])
    num_prompts = len(inputs)
    gen_len = 32
    opt_config = get_opt_config(args.model)
    cache_size = opt_config.cache_bytes(num_prompts, prompt_len + gen_len)
    hidden_size = opt_config.hidden_bytes(num_prompts, prompt_len + gen_len)
    print(f" >>> model size: {opt_config.model_bytes()/GB:.3f} GB, "
          f" >>> cache size: {cache_size/GB:.3f} GB, "
          f" >>> hidden size (prefill): {hidden_size/GB:.3f} GB")

    output_ids = model.generate(
        inputs.input_ids,
        do_sample=True,
        temperature=0.7,
        max_new_tokens=32,
        stop=stop)
    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    print("Outputs:\n" + 70 * '-')
    for i in [0, len(outputs)-1]:
        print(f"{i}: {outputs[i]}")
        print("-" * 70)

    # Shutdown
    print("Shutdown...")
    env.close_copy_threads()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="facebook/opt-6.7b",
        help="The model name.")
    parser.add_argument("--path", type=str, default="~/opt_weights",
        help="The path to the model weights. If there are no cached weights, "
             "FlexLLMGen will automatically download them from HuggingFace.")
    parser.add_argument("--offload-dir", type=str, default="~/flexllmgen_offload_dir",
        help="The directory to offload tensors. ")
    parser.add_argument("--percent", nargs="+", type=int,
        default=[100, 0, 100, 0, 100, 0],
        help="Six numbers. They are "
         "the percentage of weight on GPU, "
         "the percentage of weight on CPU, "
         "the percentage of attention cache on GPU, "
         "the percentage of attention cache on CPU, "
         "the percentage of activations on GPU, "
         "the percentage of activations on CPU")
    parser.add_argument("--pin-weight", type=str2bool, nargs="?",
        const=True, default=True)
    parser.add_argument("--cpu-cache-compute", action="store_true")
    parser.add_argument("--compress-weight", action="store_true",
        help="Whether to compress weight.")

    parser.add_argument("--compress-cache", action="store_true",
    help="Whether to compress cache.")
    parser.add_argument("--input-file", type=str)

    args = parser.parse_args()
    main(args)
    
