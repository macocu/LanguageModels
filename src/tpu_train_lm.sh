#!/bin/bash
# Job scheduling info, only for us specifically
#SBATCH --time=23:59:59
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=50G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=rikvannoord@gmail.com

# Train LM from scratch
set -eu -o pipefail

# Read in arguments
config_sh=$1 # Config sh file with experimental settings
source $config_sh # Load all variables to here

# Set variables
num_processes=8
SEED=2810

# Do the TPU training
python transformers/examples/pytorch/xla_spawn.py --num_cores 8 src/run_mlm.py $model_name_or_path --preprocessing_num_workers $num_processes --config_name $tokenizer_name --model_type $model_type --tokenizer_name $tokenizer_name $max_steps $warmup_ratio --train_file $train_file --max_seq_length $max_seq_length $line_by_line --output_dir $output_dir $do_train --per_device_train_batch_size $batch_train $overwrite_cache $overwrite_output_dir --gradient_accumulation_steps $gradient_accumulation_steps --save_steps $save_steps --pad_to_max_length --learning_rate $learning_rate --seed $SEED --logging_steps $logging_steps
