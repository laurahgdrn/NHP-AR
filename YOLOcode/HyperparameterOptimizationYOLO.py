#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 14:17:08 2024

@author: hagedorn
"""

from ultralytics import YOLO 
video = "/Users/hagedorn/Desktop/synched_actions/synched-playing/RA/RA_Playing_Fragments/ra-playing5_fragment_9.mp4"

data="/Users/hagedorn/Desktop/YOLO/yaml files/pose-custom-all.yaml" 
model_path="/Users/hagedorn/runs/pose/train8_macaquepose_only/weights/best.pt" 

model = YOLO(model_path)

# model.train(data=data, model=model, epochs=5, augment=False)

model.tune(data=data, epochs=10, iterations=300, optimizer='AdamW', plots=True, save=False, val=True)

# model.predict(video,save=True)