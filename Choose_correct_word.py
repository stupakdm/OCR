from doctr.io import DocumentFile, read_img_as_numpy, read_img_as_tensor
from doctr.models import linknet_resnet18
from doctr.models import db_resnet50
from doctr.models import crnn_vgg16_bn, crnn_mobilenet_v3_large
import numpy as np
#from doctr.io import read_img_as_numpy, read_img_as_tensor
#import tensorflow as tf
import matplotlib.pyplot as plt

from doctr.models import ocr_predictor

class Doctr:
    filenames = []
    def choose_model(self, straight_page = False, predict = False):
        #if predict == True:
            #self.model = crnn_vgg16_bn(pretrained=True)
        #else:
            #self.model = linknet_resnet18(pretrained=True)
        self.model = ocr_predictor(det_arch='db_resnet50', reco_arch='crnn_vgg16_bn', pretrained=True)
        #self.doc = DocumentFile.from_images(filenames[0])

    #def add_files(self, filename):
    #    self.filenames.append(filename)

    def find_contours(self, img):
        #for img in images:
            doc = DocumentFile.from_images(img)
            #a = [doc]
            #doc
            #print(doc.shape)
            #a =  np.array([img])

            result = self.model(doc)
            result.show(doc)
            #print(result)
            #json_file = result.
            Doctr.__synthetic_pages__(self, result)

    def __synthetic_pages__(self, result):
        synthetic_pages = result.synthesize()
        plt.imshow(synthetic_pages[0])
        plt.axis('off')
        plt.show()
b = '''       
def doctr_find_texts(filenames):
    model = ocr_predictor(pretrained=True)

    doc = DocumentFile.from_images(filenames[0])
    #doc = read_img_as_numpy(filenames[0])
    #doc1 = read_img_as_tensor(filenames[0])
    print(doc)
    result = model(doc)

    result.show(doc)

    synthetic_pages = result.synthesize()
    plt.imshow(synthetic_pages[0])
    plt.axis('off')
    plt.show()'''