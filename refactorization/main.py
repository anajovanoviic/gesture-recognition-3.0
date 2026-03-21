from src.camera_calibration import CameraCalibration

CALIBRATION_IMAGES_PATH = r'gesture-recognition\refactorization\data\images\*.jpg'
IMAGE_PATH = r'gesture-recognition\refactorization\data\images\WIN_20240913_19_09_15_Pro.jpg'
RESULT_IMAGES_FOLDER = r'gesture-recognition\refactorization\data\images\results'

def main():
    calibrator = CameraCalibration()
    
    images = calibrator.load_images(CALIBRATION_IMAGES_PATH)
    
    objpoints, imgpoints = calibrator.detect_corners(images)
    
    mtx, dist, rvecs, tvecs = calibrator.calibrate_camera(objpoints, imgpoints)
    
    calibrator.undistort_image(IMAGE_PATH, mtx, dist, RESULT_IMAGES_FOLDER, objpoints, imgpoints, rvecs, tvecs)
    
if __name__ == "__main__":
    main()