import numpy as np
import cv2
import cv2.aruco as aruco
import serial
import sys, time, math

# ------------------------------------------------------------------------------

# define variables:
marker_size = 3.35  # - [cm]
sercon = True
port = 'COM3'
camera = 1
camX, camY = 1280, 720
armID = 11
pressed = False

# ------------------------------------------------------------------------------

# define functions:
if sercon:
    ser = serial.Serial(port, 9600)  # COM port, baud rate (9600 for Arduino)
# Checks if a matrix is a valid rotation matrix.
def isRotationMatrix(R):
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6

# Calculates rotation matrix to euler angles
# The result is the same as MATLAB except the order
# of the euler angles ( x and z are swapped ).
def rotationMatrixToEulerAngles(R):
    assert (isRotationMatrix(R))

    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0

    return np.array([x, y, z])

def rotate(X, theta):
    r1 = rotationMatrixToEulerAngles(theta*R_flip)
    r2 = rotationMatrixToEulerAngles(theta)
    '''Rotate multidimensional array `X` `theta` radians around axis `axis`'''
    if np.size(X) == 3:# and np.size(theta) == 3:  # axis == 'x': return
        '''
        cx, sx = np.cos(theta[0]), np.sin(theta[0])
        cy, sy = np.cos(theta[1]), np.sin(theta[1])
        cz, sz = np.cos(theta[2]), np.sin(theta[2])

        # attempting a combination of XYZ-rotation:
        rot_matrix = (np.array([[cz*cx, cz*sy*sx-sz*cx,  cz*sy*cx+sz*sx],
                               [sz*cy,  sz*sy*sx+cz*cx,  sz*sy*cx-cz*sx],
                               [-sy,  cy*sx,  cy*cx]]))
        rot1 = np.dot(X, R_flip*rot_matrix)
        rot2 = np.dot(X, rot_matrix.T)'''

        cx, sx = np.cos(r1[0]), np.sin(r1[0])
        cy, sy = np.cos(r1[1]), np.sin(r1[1])
        cz, sz = np.cos(r1[2]), np.sin(r1[2])

        # attempting a combination of XYZ-rotation:
        rot_matrix = (np.array([[cz*cx, cz*sy*sx-sz*cx,  cz*sy*cx+sz*sx],
                               [sz*cy,  sz*sy*sx+cz*cx,  sz*sy*cx-cz*sx],
                               [-sy,  cy*sx,  cy*cx]]))
        rot1 = np.dot(X, rot_matrix)
        cx, sx = np.cos(r2[0]), np.sin(r2[0])
        cy, sy = np.cos(r2[1]), np.sin(r2[1])
        cz, sz = np.cos(r2[2]), np.sin(r2[2])

        # attempting a combination of XYZ-rotation:
        rot_matrix = (np.array([[cz*cx, cz*sy*sx-sz*cx,  cz*sy*cx+sz*sx],
                               [sz*cy,  sz*sy*sx+cz*cx,  sz*sy*cx-cz*sx],
                               [-sy,  cy*sx,  cy*cx]]))
        rot2 = np.dot(X, rot_matrix)
        Xout = [rot1[0], rot2[1], rot2[2]]
        return Xout

# ------------------------------------------------------------------------------
# Gather initialization data:

# --- Get the camera calibration path
calib_path  = ""
camera_matrix   = np.loadtxt(calib_path+'cameraMatrix_webcam.txt', delimiter=',')
camera_distortion   = np.loadtxt(calib_path+'cameraDistortion_webcam.txt', delimiter=',')

#--- Define the aruco dictionary
aruco_dict  = aruco.getPredefinedDictionary(aruco.DICT_ARUCO_ORIGINAL)
parameters  = aruco.DetectorParameters_create()

#--- 180 deg rotation matrix around the x axis
R_flip  = np.zeros((3,3), dtype=np.float32)
R_flip[0,0] = 1.0
R_flip[1,1] =-1.0
R_flip[2,2] =-1.0

#--- Capture the videocamera (this may also be a video or a picture)
cap = cv2.VideoCapture(camera)
#-- Set the camera size as the one it was calibrated with
cap.set(cv2.CAP_PROP_FRAME_WIDTH, camX)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, camY)

#-- Font for the text in the image
font = cv2.FONT_HERSHEY_PLAIN

# ------------------------------------------------------------------------------
# Main program:

while True:
    #-- Read the camera frame
    ret, frame = cap.read()

    #-- Convert in gray scale
    gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #-- remember, OpenCV stores color images in Blue, Green, Red

    #-- Find all the aruco markers in the image
    corners, ids, rejected = aruco.detectMarkers(image=gray, dictionary=aruco_dict, parameters=parameters,
                              cameraMatrix=camera_matrix, distCoeff=camera_distortion)
    arm = False
    num = 0     # to reset the variables and initialize them at first run.
    which = 0
    if ids is not None:  # and ids[0] == id_to_find
        for i in ids:
            cv2.putText(frame, str(ids[num]), (0,(num*50)+50), font, 3, (0,0,255), 2, cv2.LINE_AA)
            if ids[num] == [armID]:  # checks if arm marker is one of them
                arm = True
                which = num
            num = num+1
            # alternatively just do "print(str(num))" for console spam
        cv2.putText(frame, str(num), (0, 0), font, 3, (0, 255, 0), 2, cv2.LINE_AA)
        aruco.drawDetectedMarkers(frame, corners)
    if num == 2 and arm == True:
        ret = aruco.estimatePoseSingleMarkers(corners, marker_size, camera_matrix, camera_distortion)

        # -- Unpack the output, get only the first two
        if which == 0:
            if ids[1]==7 or ids[1]==15:
                rvec, tvec = ret[0][0, 0, :], ret[1][0, 0, :]
                rvec2, tvec2 = ret[0][1, 0, :], ret[1][1, 0, :]
                aruco.drawAxis(frame, camera_matrix, camera_distortion, rvec, tvec, 5)
                aruco.drawAxis(frame, camera_matrix, camera_distortion, rvec2, tvec2, 5)
            else:
                cv2.putText(frame, str("unknown marker!"), (70, 200), font, 8, (0, 255, 0), 5, cv2.LINE_AA)
        else:
            if ids[0]==7 or ids[0]==15:
                rvec, tvec = ret[0][1, 0, :], ret[1][1, 0, :]
                rvec2, tvec2 = ret[0][0, 0, :], ret[1][0, 0, :]
                aruco.drawAxis(frame, camera_matrix, camera_distortion, rvec, tvec, 5)
                aruco.drawAxis(frame, camera_matrix, camera_distortion, rvec2, tvec2, 5)
            else:
                cv2.putText(frame, str(num), (0, 0), font, 3, (0, 255, 0), 2, cv2.LINE_AA)
        # Debugging:
        '''print("new \n")
        #print("ret:", ret)
        print("\n tvec: ", tvec,"tvec2: ", tvec2)
        print("\n rvec: ", rvec,"rvec2: ", rvec2)'''
        # -- Draw the detected marker and put a reference frame over it
        #aruco.drawDetectedMarkers(frame, corners)

    # ------------------------------------------------------------------------------

    # output & end:
    ki = cv2.pollKey()
    if ki == ord('s') and rvec is not None and not pressed:
        rel = np.subtract(tvec2, tvec)
        # -- Obtain the rotation matrix
        R_ct = np.matrix(cv2.Rodrigues(rvec)[0])
        rot = rotationMatrixToEulerAngles(R_ct.T)
        print("\n relative distance:",rel, "\n rotations: ",rot) #roll, pitch, yaw
        output = rotate(rel, R_ct)#was rot

        str_out = "%4.0f, %4.0f, %4.0f" % ((output[0]-7),
                                            (output[2]-2),
                                            (output[1]+7))

        print("final result: \n", str_out)
        if sercon:
            ser.write(str_out.encode('utf-8'))
        pressed = True
        #break
    elif ki != ord('s') and pressed:
        pressed = False
    # --- Display the frame
    cv2.imshow('frame', frame)

    # --- use 'q' to quit
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        break
