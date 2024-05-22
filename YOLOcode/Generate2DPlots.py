import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from PIL import Image
import pickle
import numpy as np 
from scipy import linalg
import matplotlib.animation as animation 

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

with open("macaques_skeleton_2d_onlyRA.pkl", "rb") as f:
    RA = pickle.load(f)
with open("macaques_skeleton_2d_onlyLA.pkl", "rb") as f: 
    LA = pickle.load(f)

# Access annotations
LA_annotations = LA["annotations"]
RA_annotations = RA["annotations"]

import os
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pickle
import numpy as np
import matplotlib.animation as animation

# Load pickle files and other necessary setup
# (P0, P1, DLT function, triangulate function, etc.)
def save_2d_animation(LA_annotations, output_folder):
    for idx, LA_anno in enumerate(LA_annotations):
        LA_frame_dir = LA_anno["frame_dir"]
        label = LA_anno["label"]
        LA_keypoints = LA_anno["keypoint"]
        num_frames = LA_anno["total_frames"]
        if len(LA_frame_dir) < 24:
            print(f"Processing {LA_frame_dir}...")

            label_str = "Grooming" if label == 0 else "Playing"
            
            fig, ax = plt.subplots()
            connections = [[0,3],[0,4],[3,4],[0,5],[0,6],[5,6], 
                        [5,7],[6,8],[8,10],[6,12],[7,9],[5,11], 
                        [11,13],[13,15],[11, 12],[12,14],[14, 16]]

            def update(frame_index):
                ax.clear()
                ax.set_title(f'Behavior: {label_str}, Frame {frame_index + 1}')
                        
                LA_keypoints_per_frame = LA_keypoints[:, frame_index, :]

                monk1_2d = LA_keypoints_per_frame[0]
                monk2_2d = LA_keypoints_per_frame[1]

                for _c in connections:
                    ax.plot(monk1_2d[_c, 0], monk1_2d[_c, 1], c='skyblue')
                    ax.plot(monk2_2d[_c, 0], monk2_2d[_c, 1], c='lightcoral')

                ax.set_xticklabels([])
                ax.set_yticklabels([])

                # Flip the plot vertically
                ax.set_ylim(ax.get_ylim()[::-1])  # Reverses the y-axis

            anim = FuncAnimation(fig, update, frames=num_frames)

            output_file = f"{output_folder}/{LA_frame_dir}.gif"
            writer = animation.PillowWriter(fps=15)
            anim.save(output_file, writer=writer)

            # Save the plot as an image
            filename = f"{output_folder}/frame_{idx:04d}.png"
            plt.savefig(filename, dpi=100)
            plt.close()

# Define output folder for GIFs
output_folder = "2d_animated_plots"
os.makedirs(output_folder, exist_ok=True)

# Call the function to generate animated plots and save as GIF
save_2d_animation(LA_annotations, output_folder)

print("Animated plots saved as GIFs in the output folder.")