#!/bin/bash
#SBATCH --time=72:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=125GB

#module load Python/3.6.4-foss-2018a
#module load CUDA/9.1.85
#module load Boost/1.66.0-foss-2018a-Python-3.6.4
pip3 install pycuda --user
pip3 install pyyaml --user

conda install -c pytorch pytorch
python3 train_libri.py /data/s3559734/LibriSpeechRecognition/config/6head-lr005.yaml
