# Download model
wget https://dl.fbaipublicfiles.com/colorless-green-rnns/training-data/English/vocab.txt -P model

# Download state_dict of gulordava lstm, as of now stored on google drive
fileid="1SdIHXZzzubWbI7DXkMmHe0hMATtiPngX"
filename="model/state_dict.pt"
curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}" > /dev/null
curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=${fileid}" -o ${filename}

# Create output directory
mkdir extracted

# Download data
# TODO I just put a mini corpus here now, but we should put it somewhere else and download it too

