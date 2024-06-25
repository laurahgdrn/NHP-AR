#!/usr/bin/env python
# coding: utf-8

# In[4]:


import cv2 as cv
import glob
import numpy as np
import sys
from scipy import linalg
import yaml
import os

calibration_settings = {}


# In[5]:


#This will contain the calibration settings from the calibration_settings.yaml file


#Given Projection matrices P1 and P2, and pixel coordinates point1 and point2, return triangulated 3D point.
def DLT(P1, P2, point1, point2):

    A = [point1[1]*P1[2,:] - P1[1,:],
         P1[0,:] - point1[0]*P1[2,:],
         point2[1]*P2[2,:] - P2[1,:],
         P2[0,:] - point2[0]*P2[2,:]
        ]
    A = np.array(A).reshape((4,4))

    B = A.transpose() @ A
    U, s, Vh = linalg.svd(B, full_matrices = False)

    #print('Triangulated point: ')
    #print(Vh[3,0:3]/Vh[3,3])
    return Vh[3,0:3]/Vh[3,3]


# In[6]:


#Open and load the calibration_settings.yaml file
def parse_calibration_settings_file(filename):
    
    global calibration_settings

    if not os.path.exists(filename):
        print('File does not exist:', filename)
        quit()
    
    print('Using for calibration settings: ', filename)

    with open(filename) as f:
        calibration_settings = yaml.safe_load(f)

    #rudimentray check to make sure correct file was loaded
    if 'camera0' not in calibration_settings.keys():
        print('camera0 key was not found in the settings file. Check if correct calibration_settings.yaml file was passed')
        quit()


# In[7]:


parse_calibration_settings_file("calibration_settings.yaml")


# In[9]:


#Calibrate single camera to obtain camera intrinsic parameters from saved frames.
def calibrate_camera_for_intrinsic_parameters(images_folder):
    
    images_names = sorted(glob.glob(images_folder))
    images = []
    for imname in images_names:
        im = cv.imread(imname, 1)
        images.append(im)
    
    #criteria used by checkerboard pattern detector.
    #Change this if the code can't find the checkerboard. 
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.001)

    rows = calibration_settings['checkerboard_rows']
    columns = calibration_settings['checkerboard_columns']
    world_scaling = calibration_settings['checkerboard_box_size_scale'] #this will change to user defined length scale

    #coordinates of squares in the checkerboard world space
    objp = np.zeros((rows*columns,3), np.float32)
    objp[:,:2] = np.mgrid[0:rows,0:columns].T.reshape(-1,2)
    objp = world_scaling* objp

    #frame dimensions. Frames should be the same size.
    width = images[0].shape[1]
    height = images[0].shape[0]
    
    print(height, width)

    #Pixel coordinates of checkerboards
    imgpoints = [] # 2d points in image plane.

    #coordinates of the checkerboard in checkerboard world space.
    objpoints = [] # 3d point in real world space


    for i, frame in enumerate(images):
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        #find the checkerboard
        ret, corners = cv.findChessboardCorners(gray, (rows, columns), None)

        if ret == True:

            #Convolution size used to improve corner detection. Don't make this too large.
            conv_size = (11, 11)

            #opencv can attempt to improve the checkerboard coordinates
            corners = cv.cornerSubPix(gray, corners, conv_size, (-1, -1), criteria)
            cv.drawChessboardCorners(frame, (rows,columns), corners, ret)
            #cv.putText(frame, 'If detected points are poor, press "s" to skip this sample', (25, 25), cv.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 1)

            #cv.imshow('img', frame)
            #k = cv.waitKey(0)

            #if k & 0xFF == ord('s'):
            #    print('skipping')
            #    continue
            
            

            objpoints.append(objp)
            imgpoints.append(corners)

    cv.destroyAllWindows()
    ret, cmtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, (width, height), None, None)
    print('rmse:', ret)
    print('camera matrix:\n', cmtx)
    print('distortion coeffs:', dist)

    return cmtx, dist


# In[28]:


# Calibrate single camera to obtain camera intrinsic parameters from saved frames.
def calibrate_camera_for_intrinsic_parameters(images_folder):

    # New folder for labelled images
    labeled_images_folder = os.path.join('/Users/hagedorn/Desktop/calibration/', "labeled_images_RA")
    os.makedirs(labeled_images_folder, exist_ok=True)

    images_names = sorted(glob.glob(images_folder))
    images = []
    for imname in images_names:
        im = cv.imread(imname, 1)
        images.append(im)

    
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    
    rows = calibration_settings['checkerboard_rows']
    columns = calibration_settings['checkerboard_columns']
    world_scaling = calibration_settings['checkerboard_box_size_scale']

    objp = np.zeros((rows*columns, 3), np.float32)
    objp[:,:2] = np.mgrid[0:rows, 0:columns].T.reshape(-1,2)
    objp = world_scaling * objp

    width = images[0].shape[1]
    height = images[0].shape[0]

    imgpoints = []
    objpoints = []

    for i, frame in enumerate(images):
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        ret, corners = cv.findChessboardCorners(gray, (rows, columns), None)

        if ret == True:
            conv_size = (11, 11)
            corners = cv.cornerSubPix(gray, corners, conv_size, (-1, -1), criteria)
            cv.drawChessboardCorners(frame, (rows, columns), corners, ret)
            #cv.putText(frame, 'If detected points are poor, press "s" to skip this sample', (25, 25), cv.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 1)

            # Save the labelled image
            labeled_image_path = os.path.join(labeled_images_folder, os.path.basename(images_names[i])[:-4] + "_labelled.jpg")
            cv.imwrite(labeled_image_path, frame)

            objpoints.append(objp)
            imgpoints.append(corners)

    cv.destroyAllWindows()

    ret, cmtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, (width, height), None, None)
    print('rmse:', ret)
    print('camera matrix:\n', cmtx)
    print('distortion coeffs:', dist)

    return cmtx, dist




# In[30]:


cmtx_LA, dist_LA = calibrate_camera_for_intrinsic_parameters("/Users/hagedorn/Desktop/calibration/LA/*") 


# In[31]:


cmtx_RA, dist_RA = calibrate_camera_for_intrinsic_parameters("/Users/hagedorn/Desktop/calibration/RA/*") 


# In[32]:


def save_camera_intrinsics(camera_matrix, distortion_coefs, camera_name):

    #create folder if it does not exist
    if not os.path.exists('camera_parameters'):
        os.mkdir('camera_parameters')

    out_filename = os.path.join('camera_parameters', camera_name + '_intrinsics.dat')
    outf = open(out_filename, 'w')

    outf.write('intrinsic:\n')
    for l in camera_matrix:
        for en in l:
            outf.write(str(en) + ' ')
        outf.write('\n')

    outf.write('distortion:\n')
    for en in distortion_coefs[0]:
        outf.write(str(en) + ' ')
    outf.write('\n')


# In[33]:


save_camera_intrinsics(cmtx_LA, dist_LA, 'LA') 


# In[34]:


save_camera_intrinsics(cmtx_RA, dist_RA, 'RA') 


# In[8]:


cmtx_LA = np.array([[1.24548605e+03, 0.00000000e+00, 9.01399729e+02]
                     [0.00000000e+00, 1.23939543e+03, 5.61652782e+02]
                         [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])

dist_LA = np.array([[-0.38274594,  0.08610615,  0.00131274, -0.00078334,  0.06929624]])

cmtx_RA = np.array( [[1.44843298e+03, 0.00000000e+00, 9.61857174e+02]
 [0.00000000e+00, 1.49835501e+03, 5.12223856e+02]
 [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])

dist_RA = np.array([[-0.45907969,  0.34926784, -0.00225474, -0.00320615, -0.21680217]])


# In[35]:


#open paired calibration frames and stereo calibrate for cam0 to cam1 coorindate transformations
def stereo_calibrate(mtx0, dist0, mtx1, dist1, IMAGES_LA, IMAGES_RA):
    idx = 0 
    #read the synched frames
    c0_images_names = sorted(glob.glob(IMAGES_LA))
    c1_images_names = sorted(glob.glob(IMAGES_RA))

    #open images
    c0_images = [cv.imread(imname, 1) for imname in c0_images_names]
    c1_images = [cv.imread(imname, 1) for imname in c1_images_names]

    #change this if stereo calibration not good.
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.001)

    #calibration pattern settings
    rows = calibration_settings['checkerboard_rows']
    columns = calibration_settings['checkerboard_columns']
    world_scaling = calibration_settings['checkerboard_box_size_scale']

    #coordinates of squares in the checkerboard world space
    objp = np.zeros((rows*columns,3), np.float32)
    objp[:,:2] = np.mgrid[0:rows,0:columns].T.reshape(-1,2)
    objp = world_scaling* objp

    #frame dimensions. Frames should be the same size.
    width = c0_images[0].shape[1]
    height = c0_images[0].shape[0]

    #Pixel coordinates of checkerboards
    imgpoints_left = [] # 2d points in image plane.
    imgpoints_right = []

    #coordinates of the checkerboard in checkerboard world space.
    objpoints = [] # 3d point in real world space

    for frame0, frame1 in zip(c0_images, c1_images):
        gray1 = cv.cvtColor(frame0, cv.COLOR_BGR2GRAY)
        gray2 = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
        c_ret1, corners1 = cv.findChessboardCorners(gray1, (rows, columns), None)
        c_ret2, corners2 = cv.findChessboardCorners(gray2, (rows, columns), None)

        if c_ret1 == True and c_ret2 == True:

            corners1 = cv.cornerSubPix(gray1, corners1, (11, 11), (-1, -1), criteria)
            corners2 = cv.cornerSubPix(gray2, corners2, (11, 11), (-1, -1), criteria)

            p0_c1 = corners1[0,0].astype(np.int32)
            p0_c2 = corners2[0,0].astype(np.int32)

            cv.putText(frame0, 'O', (p0_c1[0], p0_c1[1]), cv.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 1)
            cv.drawChessboardCorners(frame0, (rows,columns), corners1, c_ret1)

            cv.putText(frame1, 'O', (p0_c2[0], p0_c2[1]), cv.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 1)
            cv.drawChessboardCorners(frame1, (rows,columns), corners2, c_ret2)
    
            cv.imwrite((f'annotated_frame_{idx}_cam0.jpg'), frame0) 
            cv.imwrite((f'annotated_frame_{idx}_cam1.jpg'), frame1)
            idx += 1 
            
            objpoints.append(objp)
            imgpoints_left.append(corners1)
            imgpoints_right.append(corners2)

    stereocalibration_flags = cv.CALIB_FIX_INTRINSIC
    ret, CM1, dist0, CM2, dist1, R, T, E, F = cv.stereoCalibrate(objpoints, imgpoints_left, imgpoints_right, mtx0, dist0,
                                                                 mtx1, dist1, (width, height), criteria = criteria, flags = stereocalibration_flags)

    print('rmse: ', ret)
    cv.destroyAllWindows()
    return R, T


# In[36]:


R, T = stereo_calibrate(cmtx_LA, dist_LA, cmtx_RA, dist_RA, '/Users/hagedorn/Desktop/calibration/synched/synched-LA/*','/Users/hagedorn/Desktop/calibration/synched/synched-RA/*')


# In[38]:


#Converts Rotation matrix R and Translation vector T into a homogeneous representation matrix
def _make_homogeneous_rep_matrix(R, t):
    P = np.zeros((4,4))
    P[:3,:3] = R
    P[:3, 3] = t.reshape(3)
    P[3,3] = 1
 
    return P
# Turn camera calibration data into projection matrix
def get_projection_matrix(cmtx, R, T):
    P = cmtx @ _make_homogeneous_rep_matrix(R, T)[:3,:]
    return P


# In[45]:


def check_calibration(camera0_name, camera0_data, camera1_name, camera1_data,  video_file=None, _zshift=50.,):
    cmtx0 = np.array(camera0_data[0])
    dist0 = np.array(camera0_data[1])
    R0 = np.array(camera0_data[2])
    T0 = np.array(camera0_data[3])
    cmtx1 = np.array(camera1_data[0])
    dist1 = np.array(camera1_data[1])
    R1 = np.array(camera1_data[2])
    T1 = np.array(camera1_data[3])

    P0 = get_projection_matrix(cmtx0, R0, T0)
    P1 = get_projection_matrix(cmtx1, R1, T1)

    coordinate_points = np.array([[0.,0.,0.],
                                  [1.,0.,0.],
                                  [0.,1.,0.],
                                  [0.,0.,1.]])
    z_shift = np.array([0.,0.,_zshift]).reshape((1, 3))
    draw_axes_points = 5 * coordinate_points + z_shift

    if video_file:
        cap = cv.VideoCapture(video_file)
        width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    else:
        print("No video file specified.")
        return

    while True:
        ret, frame = cap.read()

        if not ret:
            print('Error reading frame.')
            break

        # Follow RGB colors to indicate XYZ axes respectively
        colors = [(0,0,255), (0,255,0), (255,0,0)]
        
        # Draw projections to camera0
        origin = tuple(pixel_points_camera0[0].astype(np.int32))
        for col, _p in zip(colors, pixel_points_camera0[1:]):
            _p = tuple(_p.astype(np.int32))
            cv.line(frame, origin, _p, col, 2)
        
        # Draw projections to camera1
        origin = tuple(pixel_points_camera1[0].astype(np.int32))
        for col, _p in zip(colors, pixel_points_camera1[1:]):
            _p = tuple(_p.astype(np.int32))
            cv.line(frame, origin, _p, col, 2)

        cv.imshow('frame', frame)

        k = cv.waitKey(1)
        if k == 27:
            break

    cap.release()
    cv.destroyAllWindows()


# In[47]:


R0 = np.eye(3, dtype=np.float32)
T0 = np.array([0., 0., 0.]).reshape((3, 1))

#save_extrinsic_calibration_parameters(R0, T0, R, T) 

R1 = R
T1 = T #to avoid confusion, camera1 R and T are labeled R1 and T1

video_file = "/Users/hagedorn/Desktop/calibration/CheckCalibrationLA.mp4"

camera0_data = [cmtx_LA, dist_LA, R0, T0]
camera1_data = [cmtx_RA, dist_RA, R1, T1]
check_calibration('camera0', camera0_data, 'camera1', camera1_data,video_file, _zshift = 50.)


# In[ ]:


frame0 = "/Users/hagedorn/Desktop/calibration/synched/synched-LA/1_LA_calib_81.jpg"
frame1 = "/Users/hagedorn/Desktop/calibration/synched/synched-RA/1_RA_calib_81.jpg"

im0 = cv.imread(frame0)
print(im0.shape)

im1 = cv.imread(frame1)
print(im1.shape)

target_size = im1.shape[:2][::-1]  # (width, height) format

# Resize the first image to match the size of the second image
im0_resized = cv.resize(im0, target_size)

print(im0_resized.shape)


# In[27]:


import os
import cv2 as cv

# Folder containing the images
folder_path = "/Users/hagedorn/Desktop/calibration/synched/synched-RA/"

# Target size
target_size = (1920, 1080)  # Width, Height

# Iterate through each file in the folder
for filename in os.listdir(folder_path):
    # Check if the file is an image
    if filename.endswith((".jpg", ".jpeg", ".png")):
        # Read the image
        image_path = os.path.join(folder_path, filename)
        img = cv.imread(image_path)

        # Resize the image
        resized_img = cv.resize(img, target_size)

        # Save the resized image
        resized_image_path = os.path.join(folder_path, f"resized_{filename}")
        cv.imwrite(resized_image_path, resized_img)

print("All images resized successfully.")


# In[20]:


def get_world_space_origin(cmtx, dist, img_path):

    frame = cv.imread(img_path, 1)

    #calibration pattern settings
    rows = calibration_settings['checkerboard_rows']
    columns = calibration_settings['checkerboard_columns']
    world_scaling = calibration_settings['checkerboard_box_size_scale']

    #coordinates of squares in the checkerboard world space
    objp = np.zeros((rows*columns,3), np.float32)
    objp[:,:2] = np.mgrid[0:rows,0:columns].T.reshape(-1,2)
    objp = world_scaling* objp

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    ret, corners = cv.findChessboardCorners(gray, (rows, columns), None)

    cv.drawChessboardCorners(frame, (rows,columns), corners, ret)
    cv.putText(frame, "If you don't see detected points, try with a different image", (50,50), cv.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 1)
    cv.imshow('img', frame)
    cv.waitKey(0)

    ret, rvec, tvec = cv.solvePnP(objp, corners, cmtx, dist)
    R, _  = cv.Rodrigues(rvec) #rvec is Rotation matrix in Rodrigues vector form

    return R, tvec

def get_cam1_to_world_transforms(cmtx0, dist0, R_W0, T_W0, 
                                 cmtx1, dist1, R_01, T_01,
                                 image_path0,
                                 image_path1):

    frame0 = cv.imread(image_path0, 1)
    frame1 = cv.imread(image_path1, 1)

    unitv_points = 5 * np.array([[0,0,0], [1,0,0], [0,1,0], [0,0,1]], dtype = 'float32').reshape((4,1,3))
    #axes colors are RGB format to indicate XYZ axes.
    colors = [(0,0,255), (0,255,0), (255,0,0)]

    #project origin points to frame 0
    points, _ = cv.projectPoints(unitv_points, R_W0, T_W0, cmtx0, dist0)
    points = points.reshape((4,2)).astype(np.int32)
    origin = tuple(points[0])
    for col, _p in zip(colors, points[1:]):
        _p = tuple(_p.astype(np.int32))
        cv.line(frame0, origin, _p, col, 2)

    #project origin points to frame1
    R_W1 = R_01 @ R_W0
    T_W1 = R_01 @ T_W0 + T_01
    points, _ = cv.projectPoints(unitv_points, R_W1, T_W1, cmtx1, dist1)
    points = points.reshape((4,2)).astype(np.int32)
    origin = tuple(points[0])
    for col, _p in zip(colors, points[1:]):
        _p = tuple(_p.astype(np.int32))
        cv.line(frame1, origin, _p, col, 2)

    cv.imshow('frame0', frame0)
    cv.imshow('frame1', frame1)
    cv.waitKey(0)

    return R_W1, T_W1




# In[29]:


def save_extrinsic_calibration_parameters(R0, T0, R1, T1, prefix = ''):
    
    #create folder if it does not exist
    if not os.path.exists('camera_parameters'):
        os.mkdir('camera_parameters')

    camera0_rot_trans_filename = os.path.join('camera_parameters', prefix + 'camera0_rot_trans.dat')
    outf = open(camera0_rot_trans_filename, 'w')

    outf.write('R:\n')
    for l in R0:
        for en in l:
            outf.write(str(en) + ' ')
        outf.write('\n')

    outf.write('T:\n')
    for l in T0:
        for en in l:
            outf.write(str(en) + ' ')
        outf.write('\n')
    outf.close()

    #R1 and T1 are just stereo calibration returned values
    camera1_rot_trans_filename = os.path.join('camera_parameters', prefix + 'camera1_rot_trans.dat')
    outf = open(camera1_rot_trans_filename, 'w')

    outf.write('R:\n')
    for l in R1:
        for en in l:
            outf.write(str(en) + ' ')
        outf.write('\n')

    outf.write('T:\n')
    for l in T1:
        for en in l:
            outf.write(str(en) + ' ')
        outf.write('\n')
    outf.close()

    return R0, T0, R1, T1


# In[ ]:


R_W0, T_W0 = get_world_space_origin(cmtx_LA, dist_LA, "/Users/hagedorn/Desktop/calibration/synched/synched-LA/1_LA_calib_81.jpg")


# In[ ]:


R_W1, T_W1 = get_cam1_to_world_transforms(cmtx0, dist0, R_W0, T_W0,cmtx1, dist1, R1, T1, "/Users/hagedorn/Desktop/calibration/synched/synched-RA/1_RA_calib_81.jpg")


# In[ ]:


save_extrinsic_calibration_parameters(R_W0, T_W0, R_W1, T_W1, prefix = 'world_to_')


# In[ ]:




