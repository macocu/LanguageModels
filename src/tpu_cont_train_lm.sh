#!/bin/bash
# Job scheduling info, only for us specifically
#SBATCH --time=23:59:59
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=50G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=rikvannoord@gmail.com

# Continue training a LM (e.g. XLM-Roberta) on new data
set -eu -o pipefail

# Read in arguments
config_sh=$1 # Config sh file with experimental settings
source $config_sh # Load all variables to here

# Set variables
num_processes=48
SEED=2810

# Do the TPU training from the model
#python transformers/examples/pytorch/xla_spawn.py --num_cores 8 

python src/run_mlm.py --preprocessing_num_workers $num_processes --model_name_or_path $model_name $max_steps --train_file $train_file --max_seq_length $max_seq_length $line_by_line --output_dir $output_dir $do_train --per_device_train_batch_size $batch_train $overwrite_cache $overwrite_output_dir --gradient_accumulation_steps $gradient_accumulation_steps --save_steps $save_steps --pad_to_max_length --seed $SEED --logging_steps $logging_steps --learning_rate $learning_rate $warmup_ratio
