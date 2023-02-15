# NOTE: set correct paths/values here
tokenizer_fol="tok/"
train_file="/path/to/file/"
output_dir="exp"
model_type="--model_type roberta"

# This can often be the same for other experiments
max_seq_length="512"
line_by_line=""
do_train="--do_train"
batch_train="32"
overwrite_cache=""
overwrite_output_dir="--overwrite_output_dir"
# TPU training automatically divides over the 8 cores
# So actual batch size is 8 * batch_train * gradient_accumulation_steps
gradient_accumulation_steps="8"
save_steps="2500"
max_steps="--max_steps 200000"
warmup_ratio="--warmup_ratio 0.05"
learning_rate="5e-4"
logging_steps="500"
model_name_or_path="" # add as --model_name_or_path checkpoint_dir/ when restarting training
piece_masking="" #add as --piece_masking if you want to piece masking instead of whole-word masking
tokenizer_name="--tokenizer_name $tokenizer_fol"

# You can add more arguments, but make sure train_lm.sh knows about them
# E.g. you have to add the argument to the actual training call there
