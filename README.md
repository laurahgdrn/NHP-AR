# Automatic Social Action Recognition of Macaques from Video Data 

This project provides code for calibration, pose estimation, and action recognition of macaques. 

## Calibration 

Calibration is performed on the 2 cameras (LA + RA) at J1G13 using Checkerboard Detection (OpenCV). The /synched folder contains 6 synched checkerboard images from each camera. Image quality and checkerboard detection
was improved by applying gamma correction and erosion. The extrinsic and instrinsic parameter values are stored in the corresponding .dat files. 
To use this code to calibrate a new set of cameras, edit the "camera_settings.yaml" file and add new checkerboard images to the synched folder. 
For me details, refer to [this tutorial](https://temugeb.github.io/opencv/python/2021/02/02/stereo-camera-calibration-and-triangulation.html). 

## Pose estimation 
To perform pose estimation on macaques, a YOLOv8 model was built from scratch. The model was pretrained on the [MacaquePose dataset](http://www.pri.kyoto-u.ac.jp/datasets/macaquepose/). The dataset consists of approximately 12000 images of macaques in varios poses and places along with
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
```
< class index bbox_centerx bbox_centery bbox_width bbox_length kp1_x kp1_y kp2_x kp2_y ... >. 
```
After the YOLOv8 model was pretrained on the MacaquePose dataset, the model was finetuned on data from the BPRC and hyperparameter optimization was performed. The data from the BPRC was manually annotated and augmented. 
Data augmentation methods included rotation, brightening, flipping, and cropping. 
```
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
```

The model was pretrained on the MacaquePose data set for 200 epochs, and finetuned for 100 epochs. 

## Action recognition 
This project aims at classifying social interactions into grooming and playing behavior of macaques. Action recognition relies on the [mmaction2 framework](https://github.com/open-mmlab/mmaction2). As this framework is heavily focused on Human Action Recognition, the source code was modified and adapted to enable action recognition macaques. 

### Preprocessing 

The data provided for this research consists of hour long videos containing noise and redundant data. To minimize manual workload, a background subtraction algorithm was applied to extract movement by computing the pixel difference in consecutive frames. If the pixel difference is below a certain threshold (30px) over 5 consecutive frames, it is assumed that no relevant movement is occurring. These segments are then removed. Then, the YOLO model described earlier is applied 
to check whether there at least two macaques in each frame. The result are video segments containing (at least) two macaques which were then manually annotated and divided into the distinct action classes. Similar to the pose estimation, the videos were augmented by flipping and cropping. 

### SlowFast 
Wihtout further processing, the video segments were fed to the SlowFast architecture. 

### 2D Spatial-Temporal Graph Convolutional Network (2D-ST-GCN)
For 2D skeleton-based action recognition, the YOLO pose estimation model was applied to each camera viewpoint and the detected key points were extracted and stored in a pickle file according to the mmaction2 documentation: 
```
{
    "split":
        {
            'xsub_train':
                ['S001C001P001R001A001', ...],
            'xsub_val':
                ['S001C001P003R001A001', ...],
            ...
        }

    "annotations:
        [
            {
                {
                    'frame_dir': 'S001C001P001R001A001',
                    'label': 0,
                    'img_shape': (1080, 1920),
                    'original_shape': (1080, 1920),
                    'total_frames': 103,
                    'keypoint': array([[[[1032. ,  334.8], ...]]])
                    'keypoint_score': array([[[0.934 , 0.9766, ...]]])
                },
                {
                    'frame_dir': 'S001C001P003R001A001',
                    ...
                },
                ...

            }
        ]
}
```
### 3D Spatial-Temporal Graph Convolutional Network (3D-ST-GCN)
For 3D skeleton-based action recognition, the 2D skeletons were triangulated using the intrinsic and extrinsic parameters obtained during the camera calibration. 

### Results 

To ensure a fair comparison across models, common parameters unrelated to specific architectural differences were standardized. For example, the optimization strategy for each model was a stochastic gradient descent (SGD) with a fixed learning rate of 0.1, momentum of 0.9, and weight decay of 0.0005. Each model was trained for 20 epochs. 

```
-------------------------------------------------------------------
Model        | Input    | Memory | Accuracy | Precision | Recall 
-------------------------------------------------------------------
SlowFast     | RGB      | 8116   | 0.61     | 0.52      | 0.45   
-------------------------------------------------------------------
2D-ST-GCN    | 2D skel. | 1186   | 0.75     | 0.79      | 0.81   
-------------------------------------------------------------------
3D-ST-GCN    | 3D skel. | 1186   | 0.71     | 0.39      | 0.53   
-------------------------------------------------------------------
```

