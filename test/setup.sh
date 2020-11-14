#!/usr/bin/env bash
# Make data and output directories
mkdir -p data/model
mkdir -p extracted

# Download vocab
wget https://dl.fbaipublicfiles.com/colorless-green-rnns/training-data/English/vocab.txt -P data/model

# Download state_dict of gulordava lstm, as of now stored on google drive
# The model that is stored on the fb server is not used here, as it isn't the saved state dict but the model class itself.
fileid="19Lp3AM4NEPycp_IBgoHfLc_V456pmUom"
filename="data/model/state_dict.pt"
curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}" > /dev/null
curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=${fileid}" -o ${filename}
