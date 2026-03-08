# -*- coding: utf-8 -*-
"""
Created on Sat Jan 25 21:30:12 2025

@author: anadjj
"""

import cv2
import mediapipe as mp
import os
import glob
import re
import posixpath
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt2
import csv
import pandas as pd

from numpy import save
from numpy import load

"""This method should be probably refactored in the future because there is
   the similiar one in dataset.py.
"""

def twod_coordinates():
    print("hello")
    
    # Define the camera matrix 
    fx = 945.30635289
    fy = 945.76481841
    cx = 655.32425423
    cy = 379.68382228
    camera_matrix = np.array([[fx, 0, cx], 
                              [0, fy, cy], 
                              [0, 0, 1]], np.float32) 
    
    # Define the distortion coefficients 
    #dist_coeffs = np.zeros((5, 1), np.float32)
    
    dist_coeffs = np.array([0.06577396, -0.20281864, 0.00400832, 0.00620293, -0.05396558], dtype=np.float32)
    
    # Define the rotation and translation vectors 
    # rvec = np.zeros((3, 1), np.float32) 
    #tvec = np.zeros((3, 1), np.float32) 
    
    rotation_vectors = (np.array([[-0.06305334],
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
    
    translation_vectors = (np.array([[-4.93281926],
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
    
    rvec = rotation_vectors[0]
    tvec = translation_vectors[0]
    
    
    
    
    plt.figure()
    ax = plt.axes(projection="3d")
    plt2.figure()
    #bx = plt2.axes(projection="2d")
    
    points = []
    
    two_d_coordinates = {}
    relative_coordinates = {}
    
    c = 0
    for i in range (21):
        array = load(f'C:/Users/anadjj/programs_ana/master_thesis_final/gesture-recognition/mediapipe/eight_gestures/realtime/array{i+1}.npy')
        points.append(array[0])
        points.append(array[1])
        points.append(array[2])
        points_3d = np.array([[[points[c], points[c+1], points[c+2]]]], np.float32) 
        ax.scatter(points[c], points[c+1], points[c+2])
        
    
        # Map the 3D point to 2D point 
        points_2d, _ = cv2.projectPoints(points_3d, 
                                         rvec, tvec, 
                                         camera_matrix, 
                                         dist_coeffs)
        
        x2 = points_2d[0][0][0]
        y2 = points_2d[0][0][1]
    
        #print(x2)
        #print(y2, end='\n')
        
        two_d_coordinates.setdefault(i, [x2, y2])
        
        ref_coordinate = two_d_coordinates.get(0)
        
        xref = ref_coordinate[0]
        yref = ref_coordinate[1]
        
        xr = x2 - xref
        yr = y2 - yref
        
        relative_coordinates.setdefault(i, [xr, yr])
        
        
         
        #print("Ref point: ", ref_coordinate)
        #print("Ref point x value: ", xref)
        #print("Ref point y value: ", yref)
        
        #print("Rel point x value: ", xr)
        #print("Rel point y value: ", yr)
        
        #plt2.scatter(x2, y2) #add additional function to represent 2D form 
        #plt.plot(points[c], points[c+1], points[c+2], 'ro-')
        #x, y, z = [points[c-3], points[c]], [points[c-2], points[c+1]], [points[c-1], points[c+2]] 
        #ax.plot(x, y, z, color='black')
        c = c + 3
        
    print(array)
    print("Dictionary of 2D coordinates", two_d_coordinates)
    print("Dictionary of relative 2D coordinates", relative_coordinates)
    
    one_dim_array = (np.array(list(relative_coordinates.values()))).flatten()
    print("One dimensional array: ", one_dim_array)
    
    print("Array dimensions: ", one_dim_array.shape)
    
    max_abs_value = max(one_dim_array, key=abs)
    
    print("Max abs value: ", max_abs_value)
    
    #plt.figure()
    #plt2.figure()
    
    normalized_array = one_dim_array / max_abs_value
    print("Normalized array: ", normalized_array)
    
    
    with open("C:\\Users\\anadjj\\programs_ana\\master_thesis_final\\gesture-recognition\\realtimedata.csv", "r+") as f:
              np.savetxt(f, [normalized_array], delimiter=',', fmt='%1.15f')
              
              
              
              
def save_model(model, filename):
    model.save(filename)
    print("Model saved to the file")
    
    
    

    

    
    
    
    
    
    
    
    
    
    
    
    
    