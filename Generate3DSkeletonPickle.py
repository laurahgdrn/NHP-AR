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

# Load the pickle files
with open("macaques_skeleton_2d_onlyRA.pkl", "rb") as f:
    RA = pickle.load(f)
with open("macaques_skeleton_2d_onlyLA.pkl", "rb") as f: 
    LA = pickle.load(f)

# Access annotations
LA_annotations = LA["annotations"]
RA_annotations = RA["annotations"]

print("\nAnnotations:")
num_3d_skel = 0
for LA_anno, RA_anno in zip(LA_annotations, RA_annotations):
    LA_frame_dir = LA_anno["frame_dir"]
    LA_total_frames = LA_anno["total_frames"]
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
        
        # Access keypoints for each frame
        for frame_index in range(LA_total_frames)[:1]:
            
            LA_keypoints_per_frame = LA_keypoints[:, frame_index, :, :]  # Assuming shape [M, V, C]
            RA_keypoints_per_frame = RA_keypoints[:, frame_index, :, :]  # Assuming shape [M, V, C]
            # # Perform triangulation
            monk1_3d = triangulate(np.array(LA_keypoints_per_frame[0]), np.array(RA_keypoints_per_frame[0]))
            monk2_3d = triangulate(np.array(LA_keypoints_per_frame[1]), np.array(RA_keypoints_per_frame[1]))
    print(f"Total number of 3D poses: {num_3d_skel}")
def visualize_3D(monk1_3d, monk2_3d): 
    # Plot connections for monk1_3d_np
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

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

    # Set title
    ax.set_title('3D Skeleton of 2 Macaques Performing Grooming Behavior', fontsize=16)

    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])

    # Set view angle
    ax.view_init(azim=-90, elev=-90)

    plt.show()

visualize_3D(monk1_3d, monk2_3d)

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
from matplotlib.animation import FuncAnimation

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
    print(LA_frame_dir)
    print(RA_frame_dir)
    # Find the corresponding annotation in RA_annotations
    RA_anno = next((anno for anno in RA_annotations if anno["frame_dir"] == RA_frame_dir), None)
    if RA_anno != "":
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






