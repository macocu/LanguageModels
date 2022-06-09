#!/bin/bash
# Job scheduling info, only for us specifically, can be ignored
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

# Set variables - change number of processes depending on the CPU cores available
num_processes=16
SEED=2810

# Check if we're running on a TPU
tpu_call=""
pad_to_max=""
if [[ $2 == "tpu" || $2 == "TPU" ]] ; then
    tpu_call="transformers/examples/pytorch/xla_spawn.py --num_cores 8"
    pad_to_max="--pad_to_max_length"
    echo "Running traning on TPU..."
else
    echo "Not running training on TPU..."
fi

# Do the training
python $tpu_call src/run_mlm.py $pad_to_max $model_name_or_path --preprocessing_num_workers $num_processes $tokenizer_name $model_type $tokenizer_name $max_steps $warmup_ratio --train_file $train_file --max_seq_length $max_seq_length $line_by_line --output_dir $output_dir $do_train --per_device_train_batch_size $batch_train $overwrite_cache $overwrite_output_dir --gradient_accumulation_steps $gradient_accumulation_steps --save_steps $save_steps --learning_rate $learning_rate --seed $SEED --logging_steps $logging_steps $piece_masking
