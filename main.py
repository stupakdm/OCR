# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
# import pandas as pd
import numpy as np
from cv2 import cv2
#from PIL import Image
#import matplotlib.pyplot as plt
#from refactor_img import transform_image, captch_ex
from ocr_text import ocr_core
from Fix_image import find_all_images
from use_doctr import doctr_find_texts
#from Find_all_texts import Passport
from refactor_img import Passport
from flask import Flask, render_template, request
import os, glob
#import matplotlib.pyplot as plt
'''
UPLOAD_FOLDER = 'templates/'

ALLOWED_EXTENSION = set(['png', 'jpg', 'jpeg'])

def allowed_file

app = Flask(__name__)


@app.route('/')
def home_page():
    return render_template('index.html')

'''


if __name__ == '__main__':
    #app.run()
    filename = 'templates/1.jpeg'

    #captch_ex(filename)
    #img = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
    #plt.imshow(img)
    #plt.show()

    #Using tesseract and opencv
    '''img = list(find_all_images(filename))
    #img_fixed = transform_image(filename)
    texts = list(ocr_core(img))
    for i in texts:
        print(i)
    #filename = 'templates/1.pdf'
'''
    #Using doctr
    #doctr_find_texts([filename])

    #Using Passport
    pathname = 'templates/corr3.jpg'
    os.system('mkdir temp1/')
    os.system('cp '+pathname+' temp1/')
    file_list = glob.glob('temp1/*')
    os.system('mv '+file_list[0]+' temp1/test.jpg')
    print(file_list)
    os.system('mogrify -set density 300 temp1/test.jpg')
    pas1 = Passport('templates/corr.jpg')
    pas2 = Passport('temp1/test.jpg')
    #pas = Passport('templates/1.jpeg')
    print("pas1")
    pas1.test_quality()
    print("pas2")
    pas2.test_quality()
    os.system('rm -r temp1/')
    #pas.__read_img()
    #pas.rotate_img()
    #pas.getProcessedNameFilePaths()
    #pas.getProcessedSurnameFilePaths()
    #pas.getProcessedPatronymicFilePaths()
# print(ocr_core('templates/image1.png'))

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
