# NOTE: set correct paths here
tokenizer_name="tok/"
train_file="/path/to/file/"
output_dir="exp"

# This can likely be the same for other experiments
model_type="roberta"
max_seq_length="512"
line_by_line=""
do_train="--do_train"
batch_train="32"
overwrite_cache="--overwrite_cache"
overwrite_output_dir="--overwrite_output_dir"
# TPU training automatically divides over the 8 cores
# So actual batch size is 8 * batch_train * gradient_accumulation_steps
gradient_accumulation_steps="8"
save_steps="2000"
max_steps="--max_steps 150000"
warmup_ratio="--warmup_ratio 0.1"

# You can add more arguments, but make sure train_lm.sh knows about them
# E.g. you have to add the argument to the actual training call there
