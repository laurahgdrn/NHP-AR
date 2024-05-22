"""
This code employs YOLO to iterate through a directory of videos and creates
3D skeletons. The 2D skeletons are then stored in a pickle file, with the following format: 

Each pickle file corresponds to an action recognition dataset. The content of a pickle file is a dictionary with two fields: split and annotations
Split: The value of the split field is a dictionary: the keys are the split names, while the values are lists of video identifiers that belong to the specific clip.
Annotations: The value of the annotations field is a list of skeleton annotations, each skeleton annotation is a dictionary, containing the following fields:
frame_dir (str): The identifier of the corresponding video.
total_frames (int): The number of frames in this video.
img_shape (tuple[int]): The shape of a video frame, a tuple with two elements, in the format of (height, width). Only required for 2D skeletons.
original_shape (tuple[int]): Same as img_shape.
label (int): The action label.
keypoint (np.ndarray, with shape [M x T x V x C]): The keypoint annotation.
M: number of persons;
T: number of frames (same as total_frames);
V: number of keypoints (25 for NTURGB+D 3D skeleton, 17 for CoCo, 18 for OpenPose, etc. );
C: number of dimensions for keypoint coordinates (C=2 for 2D keypoint, C=3 for 3D keypoint).
keypoint_score (np.ndarray, with shape [M x T x V]): The confidence score of keypoints. Only required for 2D skeletons.

For 3D skeleton generation, the 2D skeletons from each camera view are triangulated. For the triangulation process,
the projection matrices obtained from the camera calibration (CAMERA-CALIBRATION-3.ipynb) are used.

"""



import pickle
import numpy as np
from scipy import linalg

P0 = np.array([[1.26078291e+03, 0.00000000e+00, 9.61881664e+02, 0.00000000e+00], 
               [0.00000000e+00, 1.25918171e+03, 5.58697931e+02, 0.00000000e+00], 
               [0.00000000e+00, 0.00000000e+00, 1.00000000e+00, 0.00000000e+00]])
P1 = np.array([[ 1.05166282e+02,  1.08593358e+03, -1.30331386e+03, 7.83807770e+05], 
               [-4.50725331e+02,  1.28409068e+03,  6.23480989e+02,  3.74518841e+04],
               [ 6.91053762e-01,  7.15637577e-01, -1.01526138e-01,  3.77533551e+02]])

def DLT(P1, P2, point1, point2):
    A = [point1[1]*P1[2,:] - P1[1,:],
         P1[0,:] - point1[0]*P1[2,:],
         point2[1]*P2[2,:] - P2[1,:],
         P2[0,:] - point2[0]*P2[2,:]
        ]
    A = np.array(A).reshape((4,4))
    B = A.transpose() @ A
    U, s, Vh = linalg.svd(B, full_matrices = False)
    return Vh[3,0:3]/Vh[3,3]

def triangulate(LA_monk, RA_monk): 
    monk_3d = []
    for cam1, cam2 in zip(LA_monk, RA_monk):
        _p3d = DLT(P0, P1, cam1, cam2)
        monk_3d.append(_p3d)
    return np.array(monk_3d)

def triangulate_3d(LA_keypoints_per_frame, RA_keypoints_per_frame):
    monk1_3d = triangulate(LA_keypoints_per_frame[0], RA_keypoints_per_frame[0])
    monk2_3d = triangulate(LA_keypoints_per_frame[1], RA_keypoints_per_frame[1])
    return monk1_3d, monk2_3d

# Load the pickle files
with open("macaques_skeleton_2d_onlyRA.pkl", "rb") as f:
    RA = pickle.load(f)
with open("macaques_skeleton_2d_onlyLA.pkl", "rb") as f: 
    LA = pickle.load(f)

# Access annotations
LA_annotations = LA["annotations"]
RA_annotations = RA["annotations"]

# Function to split data randomly
def random_split(data, split_ratio):
    np.random.shuffle(data)
    split_idx = int(len(data) * split_ratio)
    return data[:split_idx], data[split_idx:]

# Set the split ratio
split_ratio = 0.8  # 80% for training, 20% for validation

# Extract video identifiers
video_identifiers = [LA_anno["frame_dir"] for LA_anno in LA_annotations]

# Split video identifiers randomly
train_videos, val_videos = random_split(video_identifiers, split_ratio)

# Initialize the output data dictionary
output_data = {
    "split": {
        "xsub_train": train_videos,
        "xsub_val": val_videos
    },
    "annotations": []
}

# Iterate over each video annotation
for LA_anno, RA_anno in zip(LA_annotations, RA_annotations):
    LA_frame_dir = LA_anno["frame_dir"]
    LA_total_frames = LA_anno["total_frames"]
    img_shape = LA_anno["img_shape"]
    original_shape = LA_anno["original_shape"] 
    label = LA_anno["label"]
    LA_keypoints = LA_anno["keypoint"]

    print(f"Processing {LA_frame_dir}...")

    RA_frame_dir = f"ra-{LA_frame_dir[3:]}"
    RA_anno = next((anno for anno in RA_annotations if anno["frame_dir"] == RA_frame_dir), None)

    if RA_anno is not None: 
        
        RA_frame_dir = RA_anno["frame_dir"]
        print(f"Found RA file: {RA_frame_dir}")
        RA_total_frames = RA_anno["total_frames"]
        RA_keypoints = RA_anno["keypoint"]
        # Ensure both videos have the same number of frames
        num_frames = min(LA_total_frames, RA_total_frames)
        
        # Initialize lists to store 3D keypoints and scores
        keypoints_over_frames_3d = []
        scores_over_frames_3d = []
        
        for frame_index in range(num_frames):
            LA_keypoints_per_frame = LA_keypoints[:, frame_index, :]  # Assuming shape [V, C]
            RA_keypoints_per_frame = RA_keypoints[:, frame_index, :]  # Assuming shape [V, C]

            print("Triangulating...")
            # Triangulate 3D keypoints
            monk1_3d, monk2_3d = triangulate_3d(LA_keypoints_per_frame, RA_keypoints_per_frame)
            
            # Append 3D keypoints to the list
            keypoints_over_frames_3d.append([monk1_3d, monk2_3d])

            # Assuming all keypoints have confidence score 1
            keypoint_score = np.ones((2, 17))  
            scores_over_frames_3d.append(keypoint_score.tolist())
        # Construct 3D annotation dictionary
        annotation_3d = {
            "frame_dir": LA_frame_dir,
            "total_frames": num_frames,
            "img_shape": img_shape,
            "original_shape": original_shape,
            "label": label,
            "keypoint": np.array(keypoints_over_frames_3d),
        }

        # Append 3D annotation to the list
        output_data["annotations"].append(annotation_3d)


# Save the updated split information and annotations as a pickle file
output_pkl_file = "macaques_skeleton_3d.pkl"
with open(output_pkl_file, "wb") as f:
    pickle.dump(output_data, f)
print("Pickle file saved for the 3D action recognition dataset")
