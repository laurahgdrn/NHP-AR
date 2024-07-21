from ultralytics import YOLO

video = "/Users/hagedorn/Desktop/synched_actions/synched-playing/LA/LA_Playing_Fragments/ra-playing3_fragment_6.mp4"
model_path = "/Users/hagedorn/runs/pose/train3/weights/best.pt"
model = YOLO(model_path)

model.predict(video, save=True)


# generate more training data for YOLO model: use extracted frames, union of bouning box and filename 

# use triangulation only for distance computation 

# run action recognition YOLO model only when distance is low 

# general workflow: is there movement? Is there (more than) one macaque? Are the macaques close to each other? Are they grooming or playing? --> compare 3 models.  
