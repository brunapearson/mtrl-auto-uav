echo "downloading the pretrained weights..."

MODEL=./pre_trained_weights.zip
URL_MODEL=https://zenodo.org/record/3338078/files/models.zip?download=1

wget --quiet --no-check-certificate --show-progress $URL_MODEL -O $MODEL

echo "checking the MD5 checksum for downloaded file..."

CHECK_SUM_CHECKPOINTS='e8bc226a0e592825e23f1d7bcb329d17  pre_trained_weights.zip'

echo $CHECK_SUM_CHECKPOINTS | md5sum -c

echo "Unpacking the zip file..."

unzip -q pre_trained_weights.zip && rm pre_trained_weights.zip 

echo "All Done!!"