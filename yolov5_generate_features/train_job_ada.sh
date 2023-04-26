#!/bin/bash

#SBATCH -A research
#SBATCH -n 10
#SBATCH --partition=long
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-gpu=10
#SBATCH --mem-per-cpu=10000
#SBATCH --time=4-00:00:00
#SBATCH --output=op_file_exp1_finetune_RGB.txt
##SBATCH --mail-type=END
##SBATCH --reservation ndq

### activate conda env
source activate yolov5

cd /home2/deepti.rawat/home/CV_project/train_yolov5_RGB/fine_tuning/

echo ---Starting Training---

pip install -r requirements.txt

python3 train.py --img 640 --data coco.yaml --epochs 250 --weights yolov5n.pt --batch-size 16

echo ----Training Complete----