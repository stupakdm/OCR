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
def change_density(count, path):
    if count ==0:
        os.system('mkdir temp1/')
    os.system('cp ' + path + ' temp1/')
    file_list = glob.glob('temp1/*')
    for i in file_list:
        if 'orig' in i:
            os.system('mv ' + i + ' temp1/test.jpeg')
            break
    os.system('mogrify -set density 300 temp1/test.jpeg')
    pas1 = Passport('temp1/test.jpeg')
    pas1.test_quality2(count, path)
    print(file_list)
    os.system('rm temp1/test.jpeg')
    return None

if __name__ == '__main__':
    #app.run()
    #filename1 = 'templates/1.jpeg'
    #filename2 = 'templates/corr3.jpeg'
    filenames = []
    for i in range(7, 9):
        filenames.append(f'templates/orig{i}.jpeg')
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

    for (count, path) in enumerate(filenames):
        print(path)
        pas1 = Passport(path)
        pas1.test_quality2(0, path)
        change_density(count+1, path)
    #pas1 = Passport('templates/orig3.jpeg')
    #pas1.test_quality2(0, 'templates/orig3.jpeg')
    #change_density(0, 'templates/orig3.jpeg')
    #os.system('rm -r temp1/')
    #pas.__read_img()
    #pas.rotate_img()
    #pas.getProcessedNameFilePaths()
    #pas.getProcessedSurnameFilePths()
    #pas.getProcessedPatronymicFilePaths()
# print(ocr_core('templates/image1.png'))

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
