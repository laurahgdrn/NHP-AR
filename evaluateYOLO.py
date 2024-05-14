from ultralytics import YOLO

model=YOLO("/Users/hagedorn/runs/pose/train/weights/best.pt") 

video = "/Users/hagedorn/Desktop/synched_actions/TSN/Merged/ra-playing2_fragment_6.mp4"
# model.predict(video, save=True, conf=0.5)

# Validate the model
metrics = model.val()  # no arguments needed, dataset and settings remembered
metrics.box.map    # map50-95
metrics.box.map50  # map50
metrics.box.map75  # map75
metrics.box.maps   # a list contains map50-95 of each category

# Pose map50-95 
# train101: 0.182
# train7: 0.0621 
# train3: 0.147
# train: 0.237





