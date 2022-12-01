#!/bin/bash

set -eu -o pipefail

# Download the POS and NER train, dev and test sets for the languages we are interested in

function process_pos_set(){
# $1: link to download set from
# $2: language iso identifier
    git clone $1 data/${2}_POS
    cd data/${2}_POS
    # Macedonian is a bit different
    if [[ $2 == "mk" ]] ; then
        cd datasets/mk/
        filter mk "2,4" upos
        filter mk "2,5" xpos
        # Also put files in main folder
        cp *pos*conll ../../
        cd ../../
    else
        filter $2 "2,4" upos
        filter $2 "2,5" xpos
    fi
    # Move back to original folder
    cd ../../
}

function filter(){
# $1 language identifier
# $2 columns to cut (e.g. 2,4)
# $3 task identifier
    for type in train dev test; do
        if [[ $1 == "mk" ]] ; then
            # Macedonian file do not start with language ID
            cut -f${2} ${type}.conllu | grep -v "^#" | sed -e "s/[[:space:]]\+/ /g" > ${type}.${3}.conll
        else
            cut -f${2} ${1}*-${type}.conllu | grep -v "^#" | sed -e "s/[[:space:]]\+/ /g" > ${type}.${3}.conll
        fi
    done
}

# Get POS data for bg, mk, mt is and tr
process_pos_set https://github.com/UniversalDependencies/UD_Bulgarian-BTB bg
process_pos_set https://github.com/UniversalDependencies/UD_Maltese-MUDT mt
process_pos_set https://github.com/UniversalDependencies/UD_Icelandic-IcePaHC is
process_pos_set https://github.com/UniversalDependencies/UD_Turkish-BOUN tr
process_pos_set https://github.com/clarinsi/babushka-bench/ mk

# NER data is already downloaded and processed by, so that's just a matter of downloading
# For Maltese the NER data is too few to train a system, so we don't use it
cd data
for lang in bg is tr mk; do
    wget https://www.let.rug.nl/rikvannoord/NER/${lang}_NER.zip
    unzip ${lang}_NER.zip
    rm ${lang}_NER.zip
done
cd ../

