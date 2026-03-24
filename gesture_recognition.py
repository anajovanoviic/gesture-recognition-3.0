import os
import mediapipe as mp
import cv2
import time # added per Copilot suggestion 

from points import save_data, directory
from coordinates import Coordinates
from keras import models
from numpy import loadtxt

MODEL = r'model_8_both_hands_19_1_v3.h5'
REALTIME_DATA = "C:\\Users\\anadjj\\programs_ana\\master_thesis_final\\gesture-recognition\\refactored\\realtimedata.csv" 

def run_mediapipe(mode):
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_hands = mp.solutions.hands
    
    row_num = 0
    
    cap = cv2.VideoCapture(0)
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Camera FPS: {fps}")
    
    with mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:
      
      prediction = 5
      gesture = ""  # added per Copilot suggestion
      gesture_time = 0  # Track when gesture was detected # added per Copilot suggestion 
      
      hand_first_seen = None
       
      while cap.isOpened():
        success, image = cap.read()
        if not success:
          print("Ignoring empty camera frame.")
          # If loading a video, use 'break' instead of 'continue'.
          continue
    
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)
        
        # Draw the hand annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        k = cv2.waitKey(5) & 0xFF
        
        if results.multi_hand_landmarks:
            if hand_first_seen is None:
                hand_first_seen = time.time()
        else:
            hand_first_seen = None 
        
        if results.multi_hand_landmarks:
          for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())
            
            if mode == 0:
          
              if k == 48 or k == 49 or k == 50 or k == 51 or k == 52 or k == 53 or k == 54 or k == 55:
                  
                match k:
                  case 48:
                    pressed_key = 0
                  case 49:
                    pressed_key = 1
                  case 50:
                    pressed_key = 2
                  case 51:
                    pressed_key = 3
                  case 52:
                    pressed_key = 4
                  case 53:
                    pressed_key = 5
                  case 54:
                    pressed_key = 6
                  case 55:
                    pressed_key = 7
                    
                save_data(hand_landmarks, mode)
                coord = Coordinates(mode, pressed_key)
                coord.twod_coordinates()
                row_num = row_num + 1
                print("{} row(s) added".format(row_num))
                
                directory(mode)
            
            elif mode == 1: 
              
              if results.multi_hand_landmarks and hand_first_seen and (time.time() - hand_first_seen >= 0.5):
          
                hand_first_seen = None  # Reset immediately to require 5(0.5) seconds for next gesture
                print("App in the form for end users")
                
                save_data(hand_landmarks, mode)
                
                coord = Coordinates(mode)
                coord.twod_coordinates()
                
                model = models.load_model(MODEL)
                
                realtime_data = loadtxt(REALTIME_DATA, delimiter=',')
                realtime_data = realtime_data.reshape(1, 42)
                              
                prediction = model.predict(realtime_data)
                
                label = prediction.argmax()
                
                match label:
                    case 0:
                      gesture = "open palm" 
                    case 1:
                      gesture = "like"
                    case 2:
                      gesture = "victory"
                    case 3:
                      gesture = "rock and roll"
                    case 4:
                      gesture = "ok"
                    case 5:
                      gesture = "closed fist"
                    case 6:
                      gesture = "pointer"
                    case 7:
                      gesture = "dislike"
                
                gesture_time = time.time()  # Record when gesture was detected # added per Copilot suggestion
                directory(mode)
                
                f = open(REALTIME_DATA, "w+")
                f.close()

                os.remove(REALTIME_DATA)

        if mode == 0:
          # Flip the image horizontally for a selfie-view display.
          cv2.imshow('Gesture recognition', cv2.flip(image, 1))
          
        elif mode == 1:
          if gesture and (time.time() - gesture_time < 1): # added per Copilot suggestion
            
            # Generated with Copilot
            # Get text size for background rectangle
            text = f'{gesture}'
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1.5
            thickness = 2
            (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
            
            # Draw background rectangle with padding
            padding = 10
            x, y = 25, 42
            cv2.rectangle(image, 
                         (x - padding, y - text_height - padding), 
                         (x + text_width + padding, y + baseline + padding), 
                         (255, 255, 255),  # background
                         -1)  # Filled rectangle
            
            # Draw text on top of rectangle
            image = cv2.putText(image, text, (x, y), font, font_scale, (99,99,99), thickness, cv2.LINE_AA)
            # Generated with Copilot
          cv2.imshow('Gesture recognition', image)
          
        if k == 27:
          break   
      
    cap.release()
    
if __name__ == "__main__":
  
    """File that will present created dataset is referenced in coordinates.py
          with open(DATASET, "a") as f: ...
          
    In order to collect data press '0-7' on the keybaord depending on the label/class
    you are collecting. 
    """
    
    # following print statements are generated with Copilot
    
    print("\n" + "="*60)
    print("GESTURE RECOGNITION SYSTEM")
    print("="*60)
    print("Select mode:")
    print("  • Mode 0 - Creates custom dataset for ML model")
    print("  • Mode 1 - Starts application for end users")
    print("="*60 + "\n")
    
    mode = input("Enter mode (0 or 1): ")
    
    if mode == '0':
      run_mediapipe(0)
    elif mode == '1':
      run_mediapipe(1)
      
