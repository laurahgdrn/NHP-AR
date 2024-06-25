This project provides code for calibration, pose estimation, and action recognition of macaques. 

# Calibration 

Calibration is performed on the 2 cameras (LA + RA) at J1G13 using Checkerboard Detection (OpenCV). The /synched folder contains 6 synched checkerboard images from each camera. Image quality and checkerboard detection
was improved by applying gamma correction and erosion. The extrinsic and instrinsic parameter values are stored in the corresponding .dat files. 
To use this code to calibrate a new set of cameras, edit the "camera_settings.yaml" file and add new checkerboard images to the synched folder. 
For me details, follow https://temugeb.github.io/opencv/python/2021/02/02/stereo-camera-calibration-and-triangulation.html. 

# Pose estimation 
To perform pose estimation on macaques, a YOLOv8 model was built from scratch. The model was pretrained on the MacaquePose data set (Labuguen et al., 2021). The dataset consists of approximately 12000 images of macaques in varios poses and places along with
along with 17 annotated key points and segments. The data set was transformed to fit the YOLOv8 format which means that the images and label files are stored in the following format: 
```
dataset/
├── train/
│   ├── images/
│   └── labels/
└── val/
    ├── images/
    └── labels/
```

Each image has one label.txt file with the same filename. Each line in the label file represents the annotation for one 'object' in the image, containing the class index, the bounding box coordinates, and the key points: 

< class index bbox_centerx bbox_centery bbox_width bbox_length kp1_x kp1_y kp2_x kp2_y ... >. 

After the YOLOv8 model was pretrained on the MacaquePose dataset, the model was finetuned on data from the BPRC and hyperparameter optimization was performed. The data from the BPRC was manually annotated and augmented. 
Data augmentation methods included rotation, brightening, flipping, and cropping. 
'''
-------------------------------------------------------------------
Name             | Description                        | Value      
-------------------------------------------------------------------
weight_decay     | Weight decay (L2 regularization)   | 0.00035    
-------------------------------------------------------------------
warmup_epochs    | Number of epochs for warmup phase  | 2.43996    
-------------------------------------------------------------------
warmup_momentum  | Initial momentum during warmup     | 0.39855    
-------------------------------------------------------------------
box              | Box regression loss weight         | 6.11354    
-------------------------------------------------------------------
cls              | Classification loss weight         | 0.66992    
-------------------------------------------------------------------
dfl              | Distribution focal loss weight     | 1.90914    
-------------------------------------------------------------------
hsv_h            | Hue augmentation factor            | 0.00832    
-------------------------------------------------------------------
hsv_s            | Saturation augmentation factor     | 0.56134    
-------------------------------------------------------------------
hsv_v            | Value augmentation factor          | 0.38388    
-------------------------------------------------------------------
degrees          | Rotation augmentation degrees      | 0.0        
-------------------------------------------------------------------
translate        | Translation augmentation factor    | 0.10243    
-------------------------------------------------------------------
scale            | Scaling augmentation factor        | 0.13946    
-------------------------------------------------------------------
shear            | Shear augmentation factor          | 0.0        
-------------------------------------------------------------------
perspective      | Perspective augmentation factor    | 0.0        
-------------------------------------------------------------------
flipud           | Vertical flip probability          | 0.0        
-------------------------------------------------------------------
fliplr           | Horizontal flip probability        | 0.41099    
-------------------------------------------------------------------
mosaic           | Mosaic augmentation factor         | 1.0        
-------------------------------------------------------------------
mixup            | Mixup augmentation factor          | 0.0        
-------------------------------------------------------------------
copy_paste       | Copy-paste augmentation factor     | 0.0        
-------------------------------------------------------------------
'''

The model was pretrained on the MacaquePose data set for 200 epochs, and finetuned for 100 epochs. 

# Action recognition 


# References 
Labuguen, R., Matsumoto, J., Negrete, S. B., Nishimaru, H., Nishijo, H., Takada, M., ... & Shibata, T. (2021). MacaquePose: a novel “in the wild” macaque monkey pose dataset for markerless motion capture. Frontiers in behavioral neuroscience, 14, 581154.
