import pickle
import cv2 as cv 
import numpy as np 
import matplotlib.pyplot as plt
from scipy import linalg
import plotly.graph_objects as go
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


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
    # Assuming your triangulate function returns 3D keypoints
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

output_data = {
    "split": {
        "train": [],  # Assuming all frames are part of the training split
        "val": []     # Assuming all frames are part of the validation split
    },
    "annotations": []
}

def main():
    num_3d_skel = 0

    dataset_annotations_3d = []
    split_annotations_3d = {"xsub_train": [], "xsub_val": []}

    for LA_anno, RA_anno in zip(LA_annotations, RA_annotations):
        LA_frame_dir = LA_anno["frame_dir"]
        LA_total_frames = LA_anno["total_frames"]
        img_shape = LA_anno["img_shape"]
        original_shape = LA_anno["original_shape"] 
        label = LA_anno["label"]
        LA_keypoints = LA_anno["keypoint"]

        RA_frame_dir = f"ra-{LA_frame_dir[3:]}"
        RA_anno = next((anno for anno in RA_annotations if anno["frame_dir"] == RA_frame_dir), None)

        if RA_anno is not None: 
            num_3d_skel += 1 
            RA_frame_dir = RA_anno["frame_dir"]
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

                # Triangulate 3D keypoints
                monk1_3d, monk2_3d = triangulate_3d(LA_keypoints_per_frame, RA_keypoints_per_frame)
                
                # Append 3D keypoints to the list
                keypoints_over_frames_3d.append([monk1_3d, monk2_3d])

                # Assuming all keypoints have confidence score 1
                keypoint_score = np.ones((2, 25))  # Assuming 25 keypoints per person
                scores_over_frames_3d.append(keypoint_score.tolist())
            
            # Construct 3D annotation dictionary
            annotation_3d = {
                "frame_dir": LA_frame_dir,
                "total_frames": num_frames,
                "img_shape": img_shape,
                "original_shape": original_shape,
                "label": label,
                "keypoint": np.array(keypoints_over_frames_3d),
                "keypoint_score": np.array(scores_over_frames_3d)
            }

            # Append 3D annotation to the list
            dataset_annotations_3d.append(annotation_3d)

            # Append frame directory to the appropriate split
            split_annotations_3d["xsub_train" if "train" in LA_frame_dir else "xsub_val"].append(LA_frame_dir)
    print(f"Number of 3D poses: {num_3d_skel}")
    # Save 3D dataset annotations as a pickle file
    output_pkl_file = "macaques_skeleton_3d_train.pkl"
    with open(output_pkl_file, "wb") as f:
        pickle.dump({"split": split_annotations_3d, "annotations": dataset_annotations_3d}, f)
    print("Pickle file saved for the 3D action recognition dataset")

if __name__ == "__main__":
    main()

# Access annotations
LA_annotations = LA["annotations"]
RA_annotations = RA["annotations"]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

def update(frame_index):
    ax.clear()
    ax.set_title(f'Frame {frame_index + 1}')
    
    LA_anno = LA_annotations[frame_index]
    LA_frame_dir = LA_anno["frame_dir"]

    RA_frame_dir = f"ra-{LA_frame_dir[3:]}"

    # Find the corresponding annotation in RA_annotations
    RA_anno = next((anno for anno in RA_annotations if anno["frame_dir"] == RA_frame_dir), None)
    if RA_anno is not None:
        LA_keypoints_per_frame = LA_anno["keypoint"]
        RA_keypoints_per_frame = RA_anno["keypoint"]
        monk1_3d = triangulate(LA_keypoints_per_frame[0], RA_keypoints_per_frame[1])
        monk2_3d = triangulate(LA_keypoints_per_frame[1], RA_keypoints_per_frame[0])
        connections = [[0,3],[0,4],[3,4],[0,5],[0,6],[5,6], 
                    [5,7],[6,8],[8,10],[6,12],[7,9],[5,11], 
                    [11,13],[13,15],[11, 12],[12,14],[14, 16]]

        for _c in connections:
            ax.plot(xs=[monk1_3d[_c[0], 0], monk1_3d[_c[1], 0]], 
                    ys=[monk1_3d[_c[0], 1], monk1_3d[_c[1], 1]], 
                    zs=[monk1_3d[_c[0], 2], monk1_3d[_c[1], 2]], c='skyblue')

        # Plot connections for monk2_3d_np
        for _c in connections:
            ax.plot(xs=[monk2_3d[_c[0], 0], monk2_3d[_c[1], 0]], 
                    ys=[monk2_3d[_c[0], 1], monk2_3d[_c[1], 1]], 
                    zs=[monk2_3d[_c[0], 2], monk2_3d[_c[1], 2]], c='lightcoral')

    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])

    # Set view angle
    ax.view_init(azim=-90, elev=-90)

# ani = FuncAnimation(fig, update, frames=len(LA_annotations))

# plt.show()






