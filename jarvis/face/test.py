import os
import numpy as np
import cv2


for file in os.listdir('truth_faces'):
    img = np.loadtxt(os.path.join('truth_faces', file))
    cv2.imshow('Imagen', img)