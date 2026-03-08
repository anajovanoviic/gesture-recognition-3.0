import posixpath
import re
import os

from numpy import save

FILEPATH = r'C:\Users\anadjj\programs_ana\master-thesis-code-github-1-11-2025\gesture-recognition\refactored\m3_eg\p1_eg'
FILEPATH_MODE1 = r'C:\Users\anadjj\programs_ana\master_thesis_final\gesture-recognition\refactored\realtime'

#FILEPATH = "C:/Users/anadjj/programs_ana/master-thesis-code-github-1-11-2025/gesture-recognition/refactored/m3_eg/p1_eg"
#FILEPATH_MODE1 = "C:/Users/anadjj/programs_ana/master_thesis_final/gesture-recognition/refactored/realtime"

def save_data(hand_landmarks, mode):

    for i in range(21):
      coordinates = {}
        
      point = hand_landmarks.landmark[i]
      
      coordinates.setdefault(i, []).append(point.x)
      coordinates.setdefault(i, []).append(point.y)
      coordinates.setdefault(i, []).append(point.z)
      
      os.makedirs(FILEPATH, exist_ok=True) if mode == 0 else os.makedirs(FILEPATH_MODE1, exist_ok=True)
      
      joined_path = posixpath.join(FILEPATH, f"array{i+1}.npy") if mode == 0 else posixpath.join(FILEPATH_MODE1, f"array{i+1}.npy")
      
      save(joined_path, coordinates[i])
      
def directory(mode):
    
    output_dir = FILEPATH if mode == 0 else FILEPATH_MODE1
    
    for f in os.listdir(output_dir):
        if re.search(r'array\d+.npy', f):
            os.remove(os.path.join(output_dir, f))
      