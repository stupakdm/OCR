
from doctr.io import DocumentFile, read_img_as_numpy, read_img_as_tensor
import numpy as np
#from doctr.io import read_img_as_numpy, read_img_as_tensor
#import tensorflow as tf
import matplotlib.pyplot as plt

from doctr.models import ocr_predictor


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
    plt.show()
