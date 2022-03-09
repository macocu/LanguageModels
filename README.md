# Readme for LM training and evaluation

First setup a Conda environment, needs Python 3.7 to work with Torch.

```
conda create -n lm python=3.7
conda activate lm
```

The run_mlm.py script requires an installation from the source:

```
pip install git+https://github.com/huggingface/transformers
```

Then afterwards also run this:

```
pip install -r requirements.txt
```

## Data

First, we have to select a data set to train from. We have to be a bit careful, since we have to specify if separate lines have to be treated as separate documents. This might be the case for some data sets. If we mix individual sentences and documents, make sure a full document or paragraph is on a single line and use --line_by_line later.

For now, I will assume you have a text file in $FILE that we read line-by-line. Note: the extension has to be .txt, otherwise run_mlm.py will error.

## Tokenizing

We are going to train a byte-level tokenizer with 50k vocab. There are different options, but this seems like a widely accepted option. We could go for normal WordPiece or a lower size as well. The nice thing about byte-level is that there are never unknown characters/words anymore. Since it is byte-level, the file with merges and vocab will look odd, but it should work.

You can specify multiple input files with -i / --input_files and the output folder + name with -o / --out. If you supply --test, the tokenizer shows the pieces-output of the first 5 sentences of the first file.

```
mkdir -p tok
python src/train_tokenizer.py -i $FILE [$FILE2 $FILE3 ...] -o tok/ --test
```

## Training

We perform the training with ``run_mlm.py``, taken from the Transformers examples repo. I think this is preferable over fairseq, as Huggingface generally has all other pretrained models available, while fairseq does not always have this. It is just generally a bit more developed/popular, so the safer choice. This way we can stick with one framework for all our experiments.

To make things a bit easier, we create a ```config/bg.sh``` config file with the training settings. Please check carefully which settings are used.

We load these settings in ````src/train_lm.sh```, in which we do the actual training, to make things easier for GPU/TPU clusters.

We also need to specify the config of RoBERTa, which we can just take from previous RoBERTa models. Note this is different from the config with our training settings. This one specifies the exact model we are training - number of layers, heads, nodes, etc. Copy this config file in the folder with vocab (otherwise it doesn't work) and then run the training:

```
cp config/roberta.json tok/config.json
./src/train_lm.sh config/conf.sh
```

There are some odd things for this script (see [here](https://discuss.huggingface.co/t/how-to-train-from-scratch-with-run-mlm-py-txt-file/6588/4)), but this seems to work. I'll see if I can get a small example LM to work.

## TODO

- Train small example model
- Setup evaluation suites
- Implement fine-tuning script
- Push model to hub
- Select data sets to train on
- Determine experiments we want to run
- Make it work for TPU clusters
- Do actual training + evaluation
