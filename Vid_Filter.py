import cv2
import numpy as np
import time
import open3d as o3d
from copy import deepcopy
 
# Create a VideoCapture object and read from input file
# If the input is the camera, pass 0 instead of the video file name
left = cv2.VideoCapture('Left.mp4')
right = cv2.VideoCapture('Right.mp4')

# left = cv2.VideoCapture(0)
# right = cv2.VideoCapture(1)

# header of the ply file 
ply_header = '''ply
format ascii 1.0
element vertex %(vert_num)d
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
'''

# Function to write the ply
def write_ply(fn, verts, colors):
    verts = verts.reshape(-1, 3)
    colors = colors.reshape(-1, 3)
    verts = np.hstack([verts, colors])
    with open(fn, 'wb') as f:
        f.write((ply_header % dict(vert_num=len(verts))).encode('utf-8'))
        np.savetxt(f, verts, fmt='%f %f %f %d %d %d ')

# Function to extract disparity value when clicked by the left mouse click 
def click_event(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        # print(x, ' ', y)
        # print(depth[y][x])
        # print(depth1[y][x])
        distance = round(depth1[y][x]*1200, 3)
        print('Distance: ',distance, 'cm')
        if distance < 50:
            print("WARNING!")
        if distance >= 50:
            print("SAFE")

def scale(img, percent):
    width = int(img.shape[1] * percent / 100)
    height = int(img.shape[0] * percent / 100)
    dim = (width, height)
    
    # resize image
    resized= cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    
    return resized

# def demo(): 
# Check if camera opened successfully
if (left.isOpened()== False): 
    print("Error opening video stream or file")

prev_frame_time = 0
new_frame_time = 0

# Read until video is completed
while(left.isOpened()):
    # Capture frame-by-frame
    ret, img_left = left.read()
    ret2, img_right = right.read()
    if ret and ret2:
        # time when we finish processing for this frame
        new_frame_time = time.time()
        fps = 1/(new_frame_time-prev_frame_time)
        prev_frame_time = new_frame_time
        # converting the fps into integer
        fps = int(fps)

        resized_left = scale(img_left, 100)
        resized_right = scale(img_right, 100)
        
        # Convert images to grayscale
        gray_left = cv2.cvtColor(resized_left, cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(resized_right, cv2.COLOR_BGR2GRAY)

        # Compute stereo correspondence
        left_matcher = cv2.StereoSGBM_create()
    
        numDisparities = 128
        blockSize = 11
        disp12MaxDiff = 0
        uniquenessRatio = 5
        speckleRange = 2
        speckleWindowSize = 200
        P1 = blockSize * blockSize * 8
        P2 = blockSize * blockSize * 32
        
        #Filter parameters
        lmbda = 5000
        sigma = 3
        
        # Setting the updated parameters before computing disparity map
        left_matcher.setNumDisparities(numDisparities)
        left_matcher.setBlockSize(blockSize)
        left_matcher.setP1(P1)
        left_matcher.setP2(P2)
        left_matcher.setUniquenessRatio(uniquenessRatio)
        left_matcher.setSpeckleRange(speckleRange)
        left_matcher.setSpeckleWindowSize(speckleWindowSize)
        left_matcher.setDisp12MaxDiff(disp12MaxDiff)
        
        right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)
        wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
        wls_filter.setLambda(lmbda)
        wls_filter.setSigmaColor(sigma)

        displ = left_matcher.compute(gray_left, gray_right)  # .astype(np.float32)/16
        dispr = right_matcher.compute(gray_right, gray_left)  # .astype(np.float32)/16
        displ = np.int16(displ)
        dispr = np.int16(dispr)
        filteredImg = wls_filter.filter(displ, gray_left, None, dispr)  # important to put "imgL" here!!!
        depth= cv2.normalize(src=filteredImg, dst=filteredImg, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX)
        depth = depth.astype(np.float32)
        # print(depth.dtype)
        
        depth = ((15 * 698) / (depth + 0.00000001)) / 500
        
        depth1 = deepcopy(depth)
        # print(depth1.dtype)
        depth1 = depth1.astype(np.float32)
        h, w = resized_left.shape[:2]
        f =  698                        # When i use this focal length, we can reconstruct the 3d model
        Q = np.float32([[1, 0, 0, -0.5*w],
                        [0,-1, 0,  0.5*h], # turn points 180 deg around x-axis,
                        [0, 0, 0,     -f], # so that y-axis looks up
                        [0, 0, 1,      0]])
        points = cv2.reprojectImageTo3D(depth1, Q)
        colors = cv2.cvtColor(resized_left, cv2.COLOR_BGR2RGB)
        mask = depth > depth.min()
        points = points[mask]
        colors = colors[mask]
        out_path = r'D:\Spring-2023/Computer Vision\Assignment 2/out.ply'
        
        closest = round(depth.min()*1000,3)
        print(closest)
        
        cv2.putText(depth, f"FPS = {fps}", (230, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)
        cv2.putText(depth, f"Distance = {closest}", (230, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)
        if closest < 50:
            cv2.putText(depth, "WARNING!!!", (230, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)
        if closest >= 50:
            cv2.putText(depth, "SAFE", (230, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)
        cv2.imshow("Original", resized_left)
        cv2.imshow('Depth Map', depth)
        
        cv2.setMouseCallback('Depth Map', click_event)
    
        # Press Q on keyboard to  exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            write_ply(out_path, points, colors)
            print('%s saved' % out_path)
            break
        
    
    # Break the loop
    else: 
        break
        
    
    
# When everything done, release the video capture object
left.release()

# Closes all the frames
cv2.destroyAllWindows()

# demo()
