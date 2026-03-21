import numpy as np
import cv2

from numpy import load

FILEPATH_ARR_DIR = f'C:/Users/anadjj/programs_ana/master-thesis-code-github-1-11-2025/gesture-recognition/refactored/m3_eg/p1_eg'
#FILEPATH_MODE1_ARR_DIR = f'C:/Users/anadjj/programs_ana/master-thesis-code-github-1-11-2025/gesture-recognition/refactored/realtime'
FILEPATH_MODE1_ARR_DIR = f'C:/Users/anadjj/programs_ana/master_thesis_final/gesture-recognition/refactored/realtime'

DATASET = "dataset_27_12.csv"
#REALTIME_DATA = r'C:/Users/anadjj/programs_ana/master_thesis_final/gesture-recognition/refactored/realtimedata.csv'
REALTIME_DATA = "C:\\Users\\anadjj\\programs_ana\\master_thesis_final\\gesture-recognition\\refactored\\realtimedata.csv"

class Coordinates:
    def __init__(self, mode, value=None):
        
        self.mode = mode
        self.value = value
        # Camera matrix 
        self.fx = 945.30635289
        self.fy = 945.76481841
        self.cx = 655.32425423
        self.cy = 379.68382228
        self.dist_coeffs = np.array([0.06577396, -0.20281864, 0.00400832, 0.00620293, -0.05396558], dtype=np.float32)
        self.rotation_vectors = (np.array([[-0.06305334],
                [ 0.26082602],
                [ 1.52527918]]), np.array([[-0.33572848],
                [ 0.00964628],
                [ 0.03038733]]), np.array([[-0.04386297],
                [-0.01317906],
                [ 0.00938416]]), np.array([[-0.78339155],
                [-0.08417855],
                [-0.06895826]]), np.array([[-0.42461359],
                [-0.05341783],
                [ 0.24807439]]))
        self.translation_vectors = (np.array([[-4.93281926],
                [ 0.77853288],
                [25.92701781]]), np.array([[ 3.5722121 ],
                [-1.24019356],
                [20.12646739]]), np.array([[ 2.50598293],
                [-4.23654369],
                [16.14888803]]), np.array([[-4.83367602],
                [-0.39924379],
                [27.45116621]]), np.array([[-3.39441039],
                [-1.14483336],
                [25.5359362 ]]))
       
    def twod_coordinates(self): 

        camera_matrix = np.array([[self.fx, 0, self.cx], 
                                    [0, self.fy, self.cy], 
                                    [0, 0, 1]], np.float32) 
        
        rvec = self.rotation_vectors[0]
        tvec = self.translation_vectors[0]
        
        points = []
        
        two_d_coordinates = {}
        relative_coordinates = {}
        
        c = 0
        for i in range (21):
            
            filepath_arr = f'{FILEPATH_ARR_DIR}/array{i+1}.npy'
            filepath_mode1_arr = f'{FILEPATH_MODE1_ARR_DIR}/array{i+1}.npy'
            array = load(filepath_arr) if self.mode == 0 else load(filepath_mode1_arr)
            
            points.append(array[0])
            points.append(array[1])
            points.append(array[2])
            points_3d = np.array([[[points[c], points[c+1], points[c+2]]]], np.float32) 
            
            points_2d, _ = cv2.projectPoints(points_3d, 
                                            rvec, tvec, 
                                            camera_matrix, 
                                            self.dist_coeffs)
            
            x2 = points_2d[0][0][0]
            y2 = points_2d[0][0][1]
        
            two_d_coordinates.setdefault(i, [x2, y2])
            
            ref_coordinate = two_d_coordinates.get(0)
            
            xref = ref_coordinate[0]
            yref = ref_coordinate[1]
            
            xr = x2 - xref
            yr = y2 - yref
            
            relative_coordinates.setdefault(i, [xr, yr])
            
            c = c + 3
            
        one_dim_array = (np.array(list(relative_coordinates.values()))).flatten()
        
        max_abs_value = max(one_dim_array, key=abs)
        
        normalized_array = one_dim_array / max_abs_value
        
        # var value represents pressed key from save_data() method
        
        if self.mode == 0:
            column_one = np.array([self.value])
            
            row = np.append(column_one, normalized_array)
            
            with open(DATASET, "a") as f:
                
                np.savetxt(f, [row], delimiter=',', fmt='%1.15f') 
                
        elif self.mode == 1:
            with open(REALTIME_DATA, "x+") as f:
                np.savetxt(f, [normalized_array], delimiter=',', fmt='%1.15f')
              
            
            
    