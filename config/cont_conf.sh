# NOTE: set correct paths here
model_name="xlm-roberta-base"
train_file="/path/to/file/"
output_dir="cont_exp/"

# This can likely be the same for other experiments
model_name_or_path="--model_name_or_path $model_name"
max_seq_length="512"
line_by_line=""
do_train="--do_train"
batch_train="8"
overwrite_cache=""
overwrite_output_dir="--overwrite_output_dir"
# TPU training automatically divides over the 8 cores
# So actual batch size is 8 * batch_train * gradient_accumulation_steps
gradient_accumulation_steps="16"
save_steps="2000"
max_steps="--max_steps 200000"
logging_steps="250"
piece_masking="--piece_masking" # add as --piece_masking, otherwise does whole word masking
warmup_ratio="--warmup_ratio 0.025"
learning_rate="1e-4"
# The train script expects these variables to exist, but you should leave it empty
# when continuing training from an existing LM
model_type=""
tokenizer_name=""

# You can add more arguments, but make sure the train script (train_lm.sh) knows about them
# E.g. you have to add the argument to the actual training call there
