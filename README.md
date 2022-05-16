# Readme for LM training and evaluation

If you're not working on a Google Cloud TPU, setup a Conda environment, needs Python 3.7 to work with Torch.

```
conda create -n lm python=3.7
conda activate lm
```

Clone the repo:

```
git clone https://github.com/macocu/LanguageModels
cd LanguageModels
```

The run_mlm.py script requires an installation from the source:

```
git clone https://github.com/RikVN/transformers
cd transformers
pip install .
cd ../
```

Then afterwards also run this:

```
pip install -r requirements.txt
```

## Data

First, we have to select a data set to train from. We have to be a bit careful, since we have to specify if separate lines have to be treated as separate documents. This might be the case for some data sets. If we mix individual sentences and documents, make sure a full document or paragraph is on a single line and use --line_by_line later.

For now, I will assume you have a text file in $FILE that we just read as is. Note: the extension has to be .txt, otherwise run_mlm.py will error.

## Tokenizing

We are going to train a byte-level tokenizer with 32k vocab. There are different options, but this seems like a widely accepted option. We could go for normal WordPiece or a lower size as well. The nice thing about byte-level is that there are never unknown characters/words anymore. Since it is byte-level, the file with merges and vocab will look odd, but it should work.

You can specify multiple input files with -i / --input_files and the output folder with -o / --out. If you supply --test, the tokenizer shows the pieces-output of the first 5 sentences of the first file.

```
mkdir -p tok
python src/train_tokenizer.py -i $FILE [$FILE2 $FILE3 ...] -o tok/ --test
```

## Training

TPU-specific instructions are below, this is a general explanation.

We perform the training with ``run_mlm.py``, taken from the Transformers examples repo. I think this is preferable over fairseq, as Huggingface generally has all other pretrained models available, while fairseq does not always have this. It is just generally a bit more developed/popular, so the safer choice. This way we can stick with one framework for all our experiments.

To make things a bit easier, we create a ``config/conf.sh`` config file with the training settings. Please check carefully which settings are used.

We load these settings in ``src/train_lm.sh``, in which we do the actual training, to make things easier for GPU/TPU clusters.

We also need to specify the config of RoBERTa, which we can just take from previous RoBERTa models. Note this is different from the config with our training settings. This one specifies the exact model we are training - number of layers, heads, nodes, etc. Copy this config file in the folder with vocab (otherwise it doesn't work) and then run the training:

```
cp config/roberta.json tok/config.json
```

There are some odd things for this script (see [here](https://discuss.huggingface.co/t/how-to-train-from-scratch-with-run-mlm-py-txt-file/6588/4)), but this seems to work. I'll see if I can get a small example LM to work.


## TPU experiments

This follows closely the steps in the [Roberta tutorial](https://cloud.google.com/tpu/docs/tutorials/roberta-pytorch). In a cloud shell, open a computing instance like this:

```
export PROJECT_ID=macocu-lm
export EXP_NAME="bg"
export ZONE="europe-west4-a"
gcloud config set project ${PROJECT_ID}
gcloud config set compute/zone ${ZONE}
```

Create a new VM. Make sure to get enough memory with all these large data sets and models. I needed to get more than 200GB, though that's also more expensive. If you need to get more disk memory after the VM is created, follow the steps [here](https://stackoverflow.com/questions/22381686/how-can-size-of-the-root-disk-in-google-compute-engine-be-increased). The 32 cores are worth it during tokenizing/chunking, otherwise that will be very slow if you have a large data set.

```
gcloud compute instances create $EXP_NAME --machine-type=n1-standard-32 --image-family=torch-xla --image-project=ml-images --boot-disk-size=600GB --scopes=https://www.googleapis.com/auth/cloud-platform
```

It's important that you specified the correct zone above, because the TPU is only free for certain zones (see the email of TRC). Note that the VM still costs money even though the TPU is free.

SSH to the newly created VM:

```
gcloud compute ssh $EXP_NAME
```

Use tmux so we can start processes that keep running in the background:

```
tmux new -s macocu
```

New SSH instance so do the exports again:

```
export PROJECT_ID=macocu-lm
export EXP_NAME="bg"
export ZONE="europe-west4-a"
```

Launch TPU, hope they are available. Make sure you specified the correct accelerator-type:

```
gcloud compute tpus create $EXP_NAME --zone=${ZONE} --network=default --version=pytorch-1.11  --accelerator-type=v3-8
```

If you get the error that there are no resources available, just keep trying until it works. That's the only solution I have currently. When it was really bad, I wrapped the above command in a while true loop that tries every minute. You don't have to pay close attention, if it succeeds once the other commands will error because the resource already exists.

For saving/storing data, the best option seems a [cloud storage bucket](https://cloud.google.com/compute/docs/disks#gcsbuckets), with [explanation here](https://cloud.google.com/storage/docs/quickstart-gsutil#create). Create a bucket like this:

```
gsutil mb -b on gs://bg_bucket/
```

Easy way of downloading to the bucket directly instead of from local files:

```
curl http://nl.ijs.si/nikola/macocu/bertovski.tgz | gsutil cp - gs://bg_bucket/bertovski.tgz
```

This storage costs some money, but not much.

Now onto the training, load pre-existing environment for TPU exps:

```
conda activate torch-xla-1.11
```

We have to set the TPU_IP explicitly (apparently). You can find it under Compute Engine -> TPUs -> Internal IP.

```
export TPU_IP_ADDRESS="IP_ADDRESS_HERE"
export XRT_TPU_CONFIG="tpu_worker;0;$TPU_IP_ADDRESS:8470"
```

Then install my fork of the Transformers repo and the requirements as described above.

```
git clone https://github.com/RikVN/transformers
cd transformers
pip install -e .
cd ../
pip install -r requirements.txt
```

Load data from the bucket to the VM. Still takes some time, but faster than downloading. Note that if you only need the data once, you might as well just download it to the VM directly.

```
gsutil cp -r gs://bg_bucket/file_name_here.txt .
```

For TPU training, we use the specific tpu script. It runs the xla_spawn.py script automatically and also uses --pad_to_max_length (important for fast training!).

**Note:** go the config/conf.sh and set a location of the training file. You can also change other settings. Make sure these are correct!

A note on the batch size. The TPU automatically uses the specified train batch size on 8 devices of 8GB (v2-8) or 16GB (v3-8). So a batch size of 32 is an actual batch size of 256 already. You have to multiply this also with the gradient_accumulation_steps (e.g. 8) to get the actual batch size. In our case 32 * 8 * 8 = 2048.

I assume you have trained a vocabulary as described above and saved it in tok/. Don't forget to copy the roberta settings to the tok folder and call it config.json:

```
cp config/roberta.json tok/config.json
```

To speed up tokenization set this environment variable. If you get errors set it to false:

```
export TOKENIZERS_PARALLELISM=true
```

Start the training:

```
./src/tpu_train_lm.sh config/conf.sh
```

The conf.sh also specified where everything was saved (exp/ in this case). There we find the trained models we can send to the bucket and download to our local CPU.

### Cleaning up

Stopping instances to avoid unneccesary costs:

```
gcloud compute instances stop $EXP_NAME
gcloud compute tpus delete $EXP_NAME
```

You can also just do this in the Google Cloud Console interface.

### Fine-tuning ###

We do simple Bulgarian POS-finetuning with Simpletransformers. First get the data and process it to the right format for the script:

```
git clone https://github.com/UniversalDependencies/UD_Bulgarian-BTB
cd UD_Bulgarian-BTB
for type in train dev test; do cut -f2,4 bg_btb-ud-${type}.conllu | grep -v "^#" | sed -e "s/[[:space:]]\+/ /g" > ${type}.conll ; done
```

Then either specify your arguments in a json file (example added in config/fine.json) or just use the default ones. Run the model:

```
python src/simple_finetune.py --train_file UD_Bulgarian-BTB/train.conll -d UD_Bulgarian-BTB/dev.conll -a config/fine.json
```

### Push to hub ###

Pushing to the Hub looks straightforward, just follow the steps [here](https://huggingface.co/docs/transformers/model_sharing). I will set this up later.
