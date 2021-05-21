import matplotlib.pyplot as plt
import cv2
import easyocr
from pylab import rcParams
from IPython.display import Image
from gtts import gTTS
import os
import os.path
from os import path
from playsound import playsound

#rcParams['figure.figsize'] = 8, 16

reader = easyocr.Reader(['en'])
output = reader.readtext('D:\Aries\Image Captioning/1.png')
'''
cord = output[-1][0]
x_min, y_min = [int(min(idx)) for idx in zip(*cord)]
x_max, y_max = [int(max(idx)) for idx in zip(*cord)]

image = cv2.imread('Images/testImage.jpg')
cv2.rectangle(image,(x_min,y_min),(x_max,y_max),(0,0,255),2)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
'''
paragraph = ''
i=0
while i<len(output):
  paragraph += output[i][1]+" "
  i+=1
print(paragraph)
'''
mytext = paragraph
language = 'en'
myobj = gTTS(text=mytext, lang=language, slow=False)
if (path.exists("test.mp3")):
  playsound('C:/Users/ALAKH VERMA/PycharmProjects/ocr/test.mp3')
else:
  myobj.save("test.mp3")
  playsound('C:/Users/ALAKH VERMA/PycharmProjects/ocr/test.mp3')
'''