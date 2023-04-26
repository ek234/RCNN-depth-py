# RCNN-depth-py

## Extract RGB map, Raw-Depth map, Depth map, Instance map and Label map:
`process_dataset.ipynb` <br/>

It accepts the __nyudv2.mat__ file from __./datasets__ folder

The output is __.npy__ files corresponding to rgb, depth, raw-depth, instance and label map, in __nyudv2__ folder

## Geocentric encoding
`hha.ipynb`

Run the above python notebook in the directory __./hha_encoding__ folder

It saves the hha encoded file in the __.npy__ format in __nyudv2__ folder

## NYUDv2 Dataset
https://drive.google.com/file/d/1YFPXk6EvdhwYZZ9vLRLBbkmBpym6AzXA/view?usp=sharing 

## Regions Proposed
single images: https://drive.google.com/file/d/1y92N_EjhO96unAOfG5tBNkNSJFEeamIL/view?usp=sharing <br />
all images combined: https://drive.google.com/file/d/1H6HzI1_xTMMxivb7zfG5t-BHcB_zk8J9/view?usp=sharing


## Generating the feature vector at SPP layer of YOLOv5 at four different settings
    - Exp1: finetune all layers of YOLOv5 for RGB images
    - Exp2: freeze first 10 layers of YOLOv5 for RGB images
    - Exp3: finetune all layers of YOLOv5 for HHA images
    - Exp4: freeze first 10 layers of YOLOv5 for HHA images
    - WandB results: https://wandb.ai/deepti-rawat/cv_project_yolov5?workspace=user-deepti-rawat

## Acknowledgement
- HHA: https://github.com/charlesCXK/Depth2HHA-python.git <br />
