# Download model
wget https://dl.fbaipublicfiles.com/colorless-green-rnns/training-data/English/vocab.txt -P model
wget https://github.com/facebookresearch/colorlessgreenRNNs/raw/master/src/language_models/model.py -P model
wget -O model/model.pt https://dl.fbaipublicfiles.com/colorless-green-rnns/best-models/English/hidden650_batch128_dropout0.2_lr20.0.pt -P model


# Create output directory
mkdir extracted

# Download data
# TODO I just put a mini corpus here now, but we should put it somewhere else and download it too

