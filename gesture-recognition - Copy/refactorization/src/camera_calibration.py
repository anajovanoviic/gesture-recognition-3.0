import numpy as np
import cv2 as cv
import glob
import os

# https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html
# https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html 

class CameraCalibration:
    
    def __init__(self,
                 chessboard_size = (7,7),   # number of corners on the chessboard in the width and height
                 frame_size = (1280,720)):
        
        self.chessboard_size = chessboard_size
        self.frame_size = frame_size
        self.criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        self.objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
        self.objp[:,:2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1,2)
        self.objpoints = []
        self.imgpoints = []
        
    def load_images(self, path):
        images = glob.glob(path)
        return images
        
    def detect_corners(self, images):
        for fname in images:
            img = cv.imread(fname)
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            
            # Find the chess board corners
            ret, corners = cv.findChessboardCorners(gray, self.chessboard_size, None)
            
            # If found, add object points, image points (after refining them)
            if ret == True:
                self.objpoints.append(self.objp)
                corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), self.criteria)
                self.imgpoints.append(corners2) # corners2 or corners?
                
                # Draw and display the corners
                cv.drawChessboardCorners(img, self.chessboard_size, corners2, ret)
                cv.imshow('img', img)
                cv.waitKey(1000)
    
        cv.destroyAllWindows()
        
        return self.objpoints, self.imgpoints
    
    def calibrate_camera(self, objpoints, imgpoints):
        ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, self.frame_size, None, None)

        print("Camera Calibrated: ", ret)
        print("\nCamera Matrix:\n", mtx)
        print("\nDistortion Parameters:\n", dist)
        print("\nRotation Vectors:\n", rvecs)
        print("\nTranslation Vectors:\n", tvecs)
        
        return mtx, dist, rvecs, tvecs
        
        
    def undistort_image(self, image_path, mtx, dist, images_folder, objpoints, imgpoints, rvecs, tvecs):
        img = cv.imread(image_path)
        h,  w = img.shape[:2]
        newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

        # undistort - I
        dst = cv.undistort(img, mtx, dist, None, newcameramtx)
        # crop the image
        x, y, w, h = roi
        dst = dst[y:y+h, x:x+w]
        # cv.imwrite('calibresult1.jpg', dst)
        cv.imwrite(os.path.join(images_folder, 'calibresult1.jpg'), dst)

        # undistort - II
        mapx, mapy = cv.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w,h), 5)
        dst = cv.remap(img, mapx, mapy, cv.INTER_LINEAR)
        # crop the image
        x, y, w, h = roi
        dst = dst[y:y+h, x:x+w]
        # cv.imwrite('calibresult2.jpg', dst)
        cv.imwrite(os.path.join(images_folder, 'calibresult2.jpg'), dst)

        # Re-projection Error

        mean_error = 0
        for i in range(len(objpoints)):
            imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
            error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2)/len(imgpoints2)
            mean_error += error
        
        print( "total error: {}".format(mean_error/len(objpoints)) )
    