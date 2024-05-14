import cv2 as cv 
import glob
import numpy as np 
import matplotlib.pyplot as plt
from scipy import linalg
import plotly.graph_objects as go
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Projection matrices obtained through calibration 

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

LA_image_path = "/Users/hagedorn/Desktop/synched_actions/synched-grooming/LA/LA_Grooming_Extracted_Frames/la-grooming5_extracted_frames/frame_180.jpg"
RA_image_path = "/Users/hagedorn/Desktop/synched_actions/synched-grooming/RA/RA_Grooming_Extracted_Frames/ra-grooming5_extracted_frames/frame_180.jpg"

RA_monk_1 = [[1385.96,284.60],[1376.93,266.54],[1369.62,272.56],[1357.15,248.49],[1339.09,265.25],[1313.29,294.49],[1363.60,302.23],[1312.39,357.44],[1368.07,359.59],[1317.25,415.32],[1369.71,416.39],[1283.71,419.10],[1350.23,425.59],[1281.56,369.35],[1359.43,370.43],[1301.55,425.04],[1333.46,425.59]]

RA_monk_2 = [[1338.86,294.69],[1321.06,276.84],[1337.29,279.54],[1307.48,255.20],[1347.51,271.43],[1256.64,295.23],[1329.71,309.83],[1277.74,354.73],[1328.63,355.81],[1298.88,404.49],[1332.36,401.78],[1240.96,431.53],[1317.76,438.56],[1232.84,380.15],[1323.17,374.74],[1261.02,428.83],[1295.58,432.61]]

LA_monk_1 = [[1094.45,543.28],[1092.22,511.33],[1076.61,519.50],[1084.79,476.41],[1016.43,502.42],[1113.02,519.50],[1003.05,532.14],[1105.59,597.52],[1012.71,600.49],[1095.93,705.26],[985.97,679.99],[1127.88,607.18],[1011.97,619.07],[1142.74,530.65],[1010.48,561.86],[1126.39,627.24],[1067.70,642.10]]

LA_monk_2 = [[1121.19,463.04],[1138.28,446.69],[1133.82,439.26],[1139.03,419.94],[1122.68,393.94],[1093.70,466.75],[1074.38,402.11],[1098.16,494.99],[1020.14,455.61],[1120.45,509.10],[1103.36,510.59],[1014.94,520.99],[981.51,570.77],[1072.15,498.70],[1035.75,513.56],[1093.70,571.52],[1055.07,580.43]]

LA_monk_1_np = np.array(LA_monk_1)
LA_monk_2_np = np.array(LA_monk_2)
RA_monk_1_np = np.array(RA_monk_1)
RA_monk_2_np = np.array(RA_monk_2)


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot connections for monk1_3d_np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

connections = [[0,3],[0,4],[3,4],[0,5],[0,6],[5,6], 
               [5,7],[6,8],[8,10],[6,12],[7,9],[5,11], 
               [11,13],[13,15],[11, 12],[12,14],[14, 16]]

# Plot connections for monk1_3d_np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
monk1_3d_np = triangulate(LA_monk_1_np, RA_monk_1_np)
monk2_3d_np = triangulate(LA_monk_2_np, RA_monk_2_np)

# Plot connections for monk1_3d_np
for _c in connections:
    ax.plot(xs=[monk1_3d_np[_c[0], 0], monk1_3d_np[_c[1], 0]], 
            ys=[monk1_3d_np[_c[0], 1], monk1_3d_np[_c[1], 1]], 
            zs=[monk1_3d_np[_c[0], 2], monk1_3d_np[_c[1], 2]], c='skyblue')

# Plot connections for monk2_3d_np
for _c in connections:
    ax.plot(xs=[monk2_3d_np[_c[0], 0], monk2_3d_np[_c[1], 0]], 
            ys=[monk2_3d_np[_c[0], 1], monk2_3d_np[_c[1], 1]], 
            zs=[monk2_3d_np[_c[0], 2], monk2_3d_np[_c[1], 2]], c='lightcoral')

# Set title
ax.set_title('3D Skeleton of 2 Macaques Performing Grooming Behavior', fontsize=16)

ax.set_xticklabels([])
ax.set_yticklabels([])
ax.set_zticklabels([])

# Set view angle
ax.view_init(azim=-90, elev=-90)

plt.show()






