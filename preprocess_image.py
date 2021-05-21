import warnings

warnings.filterwarnings('ignore')

import cv2
import numpy as np
from tkinter import messagebox
from tensorflow.keras.applications.xception import preprocess_input

class preprocessing:
    def __init__(self, path):
        self.path = path

    def process_image(self):
        try:
            image = cv2.imread(self.path)
        except:
            messagebox.showerror('Error!', 'Please select a image file')
            return None
        img_arr = cv2.resize(image, (500, 250))
        test_image = np.expand_dims(img_arr, axis=0)
        test_image = preprocess_input(test_image)
        return test_image