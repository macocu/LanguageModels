# Readme for LM training and evaluation

Full overview of training and evaluating language models on either GPU or TPU.

## Setting up: GPU

If you run on TPU, please follow the TPU steps, not this one.

Clone the repo:

```
git clone https://github.com/macocu/LanguageModels
cd LanguageModels
```

Setup a Conda environment, needs Python 3.7 to work with Torch.

```
conda create -n lm python=3.7
conda activate lm
```

The run_mlm.py script requires an installation from the source:

```
git clone https://github.com/RikVN/transformers
cd transformers
pip install -e .
cd ../
```

Also install the requirements:

```
pip install -r requirements.txt
```

## Setting up: TPU

This outlines how to train a LM on a Google Cloud TPU.

### TPU: create VM
This follows closely the steps in the [Roberta tutorial](https://cloud.google.com/tpu/docs/tutorials/roberta-pytorch).

First of all, you need to have set up a [Google Cloud project](https://cloud.google.com/resource-manager/docs/creating-managing-projects) and an associated billing account, i.e. making sure that Google can charge you money.

Then, open a cloud shell, or a terminal with [gcloud](https://cloud.google.com/sdk/docs/install) installed. Then do the following:
```
export PROJECT_ID="project-name"
export EXP_NAME="exp-name"
export ZONE="europe-west4-a"
gcloud config set project ${PROJECT_ID}
gcloud config set compute/zone ${ZONE}
```

Of course, fill in your own project name and experiment name. It's **important** that you specified the correct zone above, because the TPU is only free for certain zones (see the email of TRC). So europe-west4-a might not be the right one for you.

Now we have to create a new VM. You will have to pay for this, even if the TPU time is free. Make sure to get enough memory with all these large data sets and models. For me, 600GB seems to have been enough, but it also depends on how many models you plan on saving (and their size). With 1TB you can hardly go wrong.

If you need to get more disk memory after the VM is created, follow the steps [here](https://stackoverflow.com/questions/22381686/how-can-size-of-the-root-disk-in-google-compute-engine-be-increased). The cores speed up tokenizing/chunking quite a bit (be sure to specify the number of processes in ``src/train_lm.sh`` later), but you only need to do that once. In terms of training time it's also a bit faster to use more cores. The number of cores you use scale linearly in [price](https://cloud.google.com/compute/all-pricing). I needed at least 16 to load the XLM-R-large model.

```
gcloud compute instances create $EXP_NAME --machine-type=n1-standard-16 --image-family=torch-xla --image-project=ml-images --boot-disk-size=1000GB --scopes=https://www.googleapis.com/auth/cloud-platform
```

**Note**: due to some odd loading of the weights by Pytorch/HuggingFace, if I restarted the training from a saved checkpoint, I actually needed a lot more CPU memory. We can specify a custom machine with more CPUs that have enough memory to load a trained XLM-R model:

```
gcloud compute instances create $EXP_NAME --machine-type=n1-custom-30-196608 --image-family=torch-xla --image-project=ml-images --boot-disk-size=1000GB --scopes=https://www.googleapis.com/auth/cloud-platform
```

This seemed pretty wasteful to me, as you pay for the CPUs/memory, but don't really need it after loading. Suggestions welcome.

You can SSH to the newly created VM like this:

```
gcloud compute ssh $EXP_NAME
```

But it might be easier to open an SSH connection through the console (Compute Engine Instances).

### TPU: run on VM

Important: we have to use tmux (or screen) so we can start processes that keep running in the background:

```
tmux new -s macocu
```

New SSH instance so do the exports again:

```
export PROJECT_ID="project-name"
export EXP_NAME="exp-name"
export ZONE="europe-west4-a"
gcloud config set project ${PROJECT_ID}
gcloud config set compute/zone ${ZONE}
```

Launch TPU, hope they are available. Make sure you specified the correct accelerator-type:

```
gcloud compute tpus create $EXP_NAME --zone=${ZONE} --network=default --version=pytorch-1.11  --accelerator-type=v3-8
```

If you get the error that there are no resources available, just keep trying until it works. That's the only solution I have currently. When it was really bad, I wrapped the above command in a while true loop that tries every X seconds. You don't have to pay close attention, if it succeeds once the other commands will error because the resource already exists.

```
while true; do gcloud compute tpus create $EXP_NAME --zone=${ZONE} --network=default --version=pytorch-1.11 --accelerator-type=v3-8 ; sleep 10 ; done
```

Generally, I had to wait between 10-60 minutes, but on Saturdays (and I'm assuming also Sundays) it was nearly impossible to get one.

You can run the following steps already even if you were not assigned a TPU yet. I always tried to get a TPU as soon as possible if I didn't have one yet.

Now onto the training, you can load a pre-existing environment for TPU exps:

```
conda activate torch-xla-1.11
```

We have to set the TPU_IP **explicitly**. You can find it in the console under Compute Engine -> TPUs -> Internal IP. If you get a random TPU error, it's likely that setting this went wrong:

```
export TPU_IP_ADDRESS="IP_ADDRESS_HERE"
export XRT_TPU_CONFIG="tpu_worker;0;$TPU_IP_ADDRESS:8470"
```

Clone the repo:

```
git clone https://github.com/macocu/LanguageModels
cd LanguageModels
```

Then install my fork of the Transformers repo (needed for whole-word masking for RoBERTa) and the requirements:

```
git clone https://github.com/RikVN/transformers
cd transformers
pip install -e .
cd ../
pip install -r requirements.txt
```

For TPU training, we have to add the "tpu" argument to ``src/train_lm.sh`` to let it know we train on a TPU. It runs the xla_spawn.py script automatically and also uses --pad_to_max_length (important for fast training!). We will go into this below.

### Cleaning up

If you're not running anything, you can stop or delete instances to avoid unneccesary costs:

```
gcloud compute instances stop $EXP_NAME
gcloud compute tpus delete $EXP_NAME
```

But it's probably easier to do this in the Google Cloud Console interface. Note that if you stop a TPU, you might have to wait for one to became available again if you want to start it again.

## Data

We have to select a data set to train from. We have to be a bit careful, since we have to specify if separate lines have to be treated as separate documents. This might be the case for some data sets. If that's the case, make sure a full document or paragraph is on a single line and use --line_by_line later.

For saving/storing data, an option could be a [cloud storage bucket](https://cloud.google.com/compute/docs/disks#gcsbuckets), with [explanation here](https://cloud.google.com/storage/docs/quickstart-gsutil#create). Create a bucket like this:

```
gsutil mb -b on gs://bucket/
```

You can download to the bucket directly instead of from local files:

```
curl $LINK | gsutil cp - gs://bucket/your_file_name.tgz
```

This storage costs some money, but not much. If you want to have the same data across multiple VMs, this is a lot faster than downloading it each time from a different server. If not, downloading it directly is probably the most efficient.

This is how you load data from your bucket, which is a lot faster than downloading:

```
gsutil cp -r gs://bucket/your_file_name.tgz .
```

For now, I will assume you have a text file in $FILE that we just read as is.

**Note:** the extension has to be .txt, otherwise ``run_mlm.py`` will error.

## Tokenizing

We are going to train a byte-level tokenizer with 32k vocab, but you can specify your own settings. There are different options, but this seems like a widely accepted option. The nice thing about byte-level is that there are never unknown characters/words anymore. Since it is byte-level, the file with merges and vocab will look odd, but it should work.

You can specify multiple input files with -i / --input_files and the output folder with -o / --out. If you supply --test, the tokenizer shows the pieces-output of the first 5 sentences of the first file.

```
mkdir -p tok
python src/train_tokenizer.py -i $FILE [$FILE2 $FILE3 ...] -o tok/ --test
```

We now have the vocabulary saved in the ``tok/`` folder.

## Training

We perform the training with ``run_mlm.py``, taken from the Transformers examples repo. I think this is preferable over fairseq, as Huggingface generally has all other pretrained models available, while fairseq does not always have this. It is just generally a bit more developed/popular, so the safer choice. This way we can stick with one framework for all our experiments.

To make things a bit easier, we create a ``config/conf.sh`` config file with the training settings. We load these settings in ``src/train_lm.sh``, in which we do the actual training, to make things easier for GPU/TPU clusters.

Please check **carefully** which settings are used! First, open ``config/conf.sh`` and specify the location of the training file, tokenization folder and output folder.

A note on the batch size. The TPU automatically uses the specified train batch size on 8 devices of 8GB (v2-8) or 16GB (v3-8). So a batch size of 32 is an actual batch size of 256 already. You have to multiply this also with the gradient_accumulation_steps (e.g. 8) to get the actual batch size. In our case 32 * 8 * 8 = 2048. If you run on GPU, you can just specify the actual batch size you want.

I found that for the v3 TPUs, sticking with multiples of 8, I could use a max batch size of 32 for RoBERTa models from scratch, 8 for continuing form XLM-R base, and 4 for continuing from XLM-R-large.

We also need to specify the config of RoBERTa, which we can just take from previous RoBERTa models. Note this is different from the config with our training settings. This one specifies the exact model we are training - number of layers, heads, nodes, etc. Copy this config file in the folder with vocab (otherwise it doesn't work):

```
cp config/roberta.json tok/config.json
```

There were some odd things for this script (see [here](https://discuss.huggingface.co/t/how-to-train-from-scratch-with-run-mlm-py-txt-file/6588/4)), but this seems to work.

Set this variable to false to avoid warnings:

```
export TOKENIZERS_PARALLELISM=false
```

Now, check if ``src/train_lm.sh`` specified the correct of number processes in your setup (default is 16). You can set it to the number of CPU cores, or a bit less. For large files this can speed up tokenization and chunking quite a bit.

Now we can start the actual training. If you run on TPU, please assign "tpu" or "TPU" as the second argument. If not, please add any other string.

```
./src/train_lm.sh config/conf.sh tpu
```

Tokenization takes quite some time for large files, but once you did it once it's cached, so if training fails somehow it restarts very quickly. **Some advice**: try training a model with a reasonably small data set first (e.g. 50k lines), so that you can more quickly debug issues.

The conf.sh also specified where everything was saved (exp/ in this case). There we find the trained models, which you can just copy with scp to your local server.

### Continue training from LM

It's also possible to continue training from an already existing LM, XLM-R for example. Please open ``config/cont_conf.sh`` and add the correct paths. It uses a lower learning rate, batch size and number of total steps. It also does piece masking, instead of the whole-word masking we did for training RoBERTa from scratch, because that's how XLM-R was trained.

You then start the training in the same way:

```
./src/train_lm.sh config/cont_conf.sh tpu
```

If you start from XLM-R-large, training will be about 3 times as slow as training a RoBERTa model from scratch. But you also need a lot less steps to get excellent performance. XLM-R-base is about twice as fast as XLM-R-large, it seems.

### Restarting training from checkpoint ###

You might want to restart the training due to a variety of reasons (see below for common errors). You can easily do so, but you have to specify this in the config file. In the config file you trained with, add the following line with the correct checkpoint folder:

```
model_name_or_path="--model_name_or_path /path/to/checkpoint-30000/"
```

**Also**, you should **not** overwrite the output folder anymore, so change it to empty (and do not remove, because the script expects the variable to exist):

```
overwrite_output_dir=""
```

It will read the exact training state from the saved optimizer state. It is possible to specify a different number of total steps though, which can be useful if you want to train longer than you initially thought.

### TPU Errors ###

I've encountered quite a lot of TPU-related errors. Here is an overview:

If the loading of the model fails with a SIGKILL error, this is probably due to your VM going out of memory.

If the script immediately errors with a TPU error, you might have exported the wrong TPU IP-address. It needs to be the TPU IP, not the VM IP. It's also possible you started the training script without the "tpu" argument.

If you get a TPU error very quickly into training, it might also be that your batch size is to large for the TPU. You can try a very small batch size to see if that was the issue.

If the training suddenly fails after a random amount of steps (often with a SIGABRT error), probably something happened to the TPU. I'v had this a few times and didn't find a cause. It seemed to happen randomly sometimes. For me, sometimes it worked to simply restart the training from the latest checkpoint. If that didn't help, I deleted the TPU and asked for a new one. Make sure to specify and export the new IP-addresses. That solved the problem.

It happened to me once that the VM became completely unreachable over SSH. Google has a [SSH troubleshooting overview](https://cloud.google.com/compute/docs/troubleshooting/troubleshooting-ssh), but nothing worked. What I did to fix it was delete the VM (and TPU), but keep the boot disk:

```
gcloud compute instances delete $VM_NAME --keep-disks=boot
```

And then start a new VM from this boot disk:

```
gcloud compute instances create $NEW_VM_NAME --disk name=$BOOT_DISK_NAME,boot=yes,auto-delete=no --machine-type=n1-standard-16 --image-family=torch-xla --image-project=ml-images --boot-disk-size=1000GB --scopes=https://www.googleapis.com/auth/cloud-platform
```

If you get this error immediately:

```
OSError: file tok/config.json not found
```

It means you forgot to copy the roberta.json file to the tokenization folder:

```
cp config/roberta.json tok/config.json
```

## Fine-tuning on GPU ##

We also want to fine-tune our trained models on downstream tasks. This I only ran on GPUs.

We do simple token classification with Simpletransformers, for POS tagging and NER.

First, we get the data and process it in the right format for the script. You can run this:

```
./src/get_finetune_data.sh
```

To automatically download and process the POS/NER sets we used.

For UPOS and XPOS-tagging, we use the UD sets for  [Bulgarian](https://github.com/UniversalDependencies/UD_Bulgarian-BTB), [Icelandic](https://github.com/UniversalDependencies/UD_Icelandic-IcePaHC), 
[Macedonian](https://github.com/clarinsi/babushka-bench/),
[Maltese](https://github.com/UniversalDependencies/UD_Maltese-MUDT) and [Turkish](https://github.com/UniversalDependencies/UD_Turkish-BOUN). For NER, we use the [wikiann](https://github.com/afshinrahimi/mmner) datasets for Bulgarian, Macedonian and Turkish, while for Icelandic we use the [MIMIR](https://repository.clarin.is/repository/xmlui/bitstream/handle/20.500.12537/140/train-valid-test-split.zip) NER set.

Then either specify your arguments in a json file (example added in config/fine.json) or just use the default ones. I highly advise to specify your own config file, or at least check out the default settings in ``src/finetune.py``.

You can run a baseline XLM-R model like this, as it's the default:

```
python src/finetune.py --train_file data/bg_POS/train.upos.conll -t data/bg_POS/dev.upos.conll -a config/fine.json
```

You can also run some more experiments and save all results via ``src/finetune.sh``. Only works if you obtained the files by downloading them via ``get_finetune_data.sh``. You need to specify the following 5 arguments:

1. Checkpoint folder (e.g. /path/to/checkpoint-50000) or name of LM (e.g. "xlm-roberta-large") 
2. Folder where we save all results (e.g. /path/to/exp/)
3. Config file with all settings (e.g. config/fine.json)
4. Language iso code, which is needed to find the data files (e.g. bg)
5. Simpletransformers model type (e.g. xlmroberta or roberta)

This script will do UPOS, XPOS and NER experiments for the specified language, and evaluate on both dev and test. For example:

```
./src/finetune.sh xlm-roberta-base bg_exp/ config/fine.json bg xlmroberta
```

Will calculate baseline scores for fine-tuning XLM-R-base on the Bulgarian UPOS, XPOS and NER data sets. All scores are in the \*eval files in the output folder you specified.

By default, we evaluate each epoch and use early stopping (patience=2) based on the F-score. To save space, it does **not** save any of the trained models. You can specify more to be tested files with -t. The first one specified is used as the dev set.

At the end of training, we evaluate on the specified test sets (-t) and print precision, recall and micro F1-score (which is the exact same as the accuracy).

**Note**: we explicitly do not do any caching of the data sets. There were some weird errors in Simpletransformers that seemed to be related to the caching.
