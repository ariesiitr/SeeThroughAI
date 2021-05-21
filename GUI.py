import warnings
import cv2
from matplotlib.pyplot import matshow

warnings.filterwarnings('ignore')
from tkinter import filedialog
from tkinter import *

class_dictionary = {'10': 0, '100': 1, '20': 2, '200': 3, '2000': 4, '50': 5, '500': 6}
vals = list(class_dictionary.values())
keys = list(class_dictionary.keys())

win = Tk()
win.title("Which note is it?")
win.geometry("500x500")

# Labels

path_label = Label(win)
path_label.pack()

# Entry box

entry_path_var = StringVar()
enter_path = Entry(win, width=30, textvariable=entry_path_var)
enter_path.pack()
enter_path.focus()


# Buttons

# Predict Button

def submit():
    import preprocess_image as pi
    import numpy as np

    paths = entry_path_var.get()
    paths = paths.split(',')
    new_path = []

    import matplotlib.pyplot as plt
    from matplotlib.pyplot import imread
    from tensorflow.keras import models
    from tkinter import messagebox
    import os

    model = models.load_model('static/Xception_model.h5')

    no_of_images = len(paths)
    x = np.zeros((no_of_images, 250, 500, 3))

    for count, path in enumerate(paths):
        if(os.path.exists(path + '1.jpg')):
            image = imread(path + '1.jpg')
            x[count, :, :, :] = image
        else:
            process_img = pi.preprocessing(path)
            process_img = process_img.process_image()
            # plt.imsave(path + '1.jpg', process_img)
            # new_path.append(path + '1.jpg')
            # dim = process_img.shape
            # print(dim)
            x[count, :, :, :] = process_img

    predict = model.predict(x)

    for count, path in enumerate(paths):
        app = Window(win)
        app.image(path)
        idx = np.argmax(predict, axis=1)
        confidence = predict[0, idx] * 100
        digit = keys[vals.index(idx)]
        Label(win, text='Note predicted = ' + str(digit) +
                        '\nProbability of being this note =' + str(confidence)).place(x=800,y=500)
        win.geometry("1000x500")
    enter_path.delete(0, END)


button_path = Button(win, height=2, width=10, text='Currency', bg='#3cba54', fg='black', command=submit)
button_path.pack(pady=5)

# Browse Button

def caption_gen():
    
    import tensorflow as tf
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)
        
    from pickle import load
    from numpy import argmax
    from keras.preprocessing.sequence import pad_sequences
    from keras.applications.vgg16 import VGG16
    from keras.preprocessing.image import load_img
    from keras.preprocessing.image import img_to_array
    from keras.applications.vgg16 import preprocess_input
    from keras.models import Model
    from keras.models import load_model
    from predict import extract_features, generate_desc
    
    global description
    
    tokenizer = load(open('D:\Aries\Image Captioning/tokenizer.pkl', 'rb'))
    max_length = 34
    model = load_model('D:\Aries\Image Captioning/model-ep003-loss3.647-val_loss3.871.h5')

    paths = entry_path_var.get()
    paths = paths.split(',')

    for count, path in enumerate(paths):
        photo = extract_features(path)
        description = generate_desc(model, tokenizer, photo, max_length)

    for count, path in enumerate(paths):
        app = Window(win)
        app.image(path)
        description = description[9:len(description)-7]
        Label(win, text="The description is: " + description).place(x=800,y=500)
        win.geometry("1000x500")
    enter_path.delete(0, END)


button_path = Button(win,height=2, width=10, text='Caption', bg='#3cba54', fg='black', command=caption_gen)
button_path.pack(pady = 5)

# OCR Button

def read():
    global paragraph
    import easyocr

    paths = entry_path_var.get()
    paths = paths.split(',')

    reader = easyocr.Reader(['en'])

    for count, path in enumerate(paths):
        output = reader.readtext(path)
        paragraph = ''
        i = 0
        while i < len(output):
            paragraph += output[i][1] + " "
            i += 1
        
    for count, path in enumerate(paths):
        app = Window(win)
        app.image(path)

        Label(win, text='Text detected = ' + str(paragraph)).place(x=800,y=500)
        win.geometry("1000x1000")
    enter_path.delete(0, END)

button_path = Button(win,height=2, width=10, text='Read', bg='#3cba54', fg='black', command=read)
button_path.pack(pady=5)

def browse_files():
    tk_filenames = filedialog.askopenfilenames(title='Please select one or more files')
    paths = ''
    for count, filename in enumerate(tk_filenames):
        if count == 0:
            paths += filename
        else:
            paths += ',' + filename
    entry_path_var.set(paths)

button_browse = Button(win,height=2, width=10, text='Browse files', bg='yellow', fg='black', command=browse_files)
button_browse.pack(pady = 5)


# Predict Accuracy Button

# def predict_acc():
#     from tkinter import messagebox
#
#     dir_path = entry_path_var.get()
#     if dir_path == '':
#         messagebox.showerror('Error', 'Please select a path for directory')
#         enter_path.delete(0, END)
#         return None
#
#     import preprocess_image as pi
#     import numpy as np
#     from tensorflow.keras.preprocessing.image import ImageDataGenerator as IDG
#     from tensorflow.keras import models
#
#     model = models.load_model('static/Xception_model.h5')
#
#     generator = IDG(rescale=1 / 255.0)
#     acc_generator = generator.flow_from_directory(dir_path, (250, 500), class_mode='binary')
#     try:
#         accuracy = model.evaluate_generator(acc_generator)
#     except:
#         messagebox.showerror("Error", 'Directory does not contains any sub-directories of classes')
#         enter_path.delete(0, END)
#         return None
#     messagebox.showinfo('Accuracy', 'Accuracy on the data set is : ' + str(accuracy[1] * 100))
#     enter_path.delete(0, END)
#
# pred_acc_button = Button(win, text='Predict Accuracy', bg='#f4c20d', fg='black', command=predict_acc)
# pred_acc_button.pack()


# Predict Accuracy Browse Button

'''
def browse_dir():
    tk_dir_path = filedialog.askdirectory(
        title='Please open directory which contains sub-directories for different Indian currency notes')
    entry_path_var.set(tk_dir_path)

button_browse = Button(win, text='Browse directory', bg='#f4c20d', fg='black', command=browse_dir)
button_browse.pack()
'''
# Load and Label Images

""" Copied from https://pythonbasics.org/tkinter-image/"""

from PIL import Image, ImageTk
from tkinter import Frame
from tkinter import BOTH


class Window(Frame):
    def __init__(self, master=None):
        Frame.__init__(self, master)
        self.master = master
        self.pack(fill=BOTH, expand=1)

    def image(self, path):
        load = Image.open(path)
        load = load.resize((300,300))
        render = ImageTk.PhotoImage(load)
        img = Label(self, image=render)
        img.image = render
        img.place(x=250, y=50)



# Window

win.mainloop()