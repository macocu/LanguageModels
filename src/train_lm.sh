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
SEED=100

python src/run_mlm.py --config_name $tokenizer_name --model_type $model_type --tokenizer_name $tokenizer_name $max_steps $warmup_ratio --train_file $train_file --max_seq_length $max_seq_length $line_by_line --output_dir $output_dir $do_train --per_device_train_batch_size $batch_train $overwrite_cache $overwrite_output_dir --gradient_accumulation_steps $gradient_accumulation_steps --save_steps $save_steps --learning_rate $learning_rate --seed $SEED
