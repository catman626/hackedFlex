# python -m flexllmgen.flex_opt --model facebook/opt-6.7b --input-file ../../warmup.txt --gpu-batch-size 1 --num-gpu-batch 1
# python -m flexllmgen.flex_opt --model facebook/opt-6.7b --input-file ../../warmup.txt --gpu-batch-size 1 --num-gpu-batch 1 --percent 100 0 0 100 100 0 
# python -m flexllmgen.flex_opt --model facebook/opt-6.7b --input-file ../../warmup.txt --gpu-batch-size 1 --num-gpu-batch 1 --percent 100 0 0 100 100 0 --overlap
# python -m flexllmgen.flex_opt --model facebook/opt-6.7b --input-file ../../warmup.txt --gpu-batch-size 1 --num-gpu-batch 1 --percent 100 0 0 100 100 0 
# python -m flexllmgen.flex_opt --model facebook/opt-6.7b --input-file prompt01.txt --gpu-batch-size 1 --num-gpu-batch 1 --percent 100 0 0 100 100 0 

# try to close TORCH_CUBLAS_LT 
# not working
# export TORCH_CUBLAS_LT=0 
# python -m flexllmgen.flex_opt --model facebook/opt-6.7b --input-file prompt01.txt --gpu-batch-size 1 --num-gpu-batch 1 --percent 0 100 0 100 100 0 --overlap false

# try to use cpu compute mha
# not working
python -m flexllmgen.flex_opt --model facebook/opt-125m --prompt-len 100 --gen-len 32 --overlap false
