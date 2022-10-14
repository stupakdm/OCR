import os, time, re

# from doctr.io import read_img_as_numpy, read_img_as_tensor
# import tensorflow as tf
import cv2
import enchant
import matplotlib.pyplot as plt
import numpy as np
import pymorphy2
from doctr.io import DocumentFile  # , read_img_as_numpy, read_img_as_tensor
from doctr.models import ocr_predictor
import pytesseract as pt


class Choose_correct_word:
    def __init__(self, image):
        pass

class Doctr:
    def __init__(self, H, W):
        self.prev_max_block = [(0,0),(0,0)]
        self.max_block = [(0,0),(0,0)]
        self.min_x = W
        self.max_x = 0
        self.min_y = H
        self.max_y = 0
        self.mid_x, self.mid_y = 0, 0
        self.avg_h = 0
        self.avg_w = 0
        '''for block in blocks:
            if block[0] < self.min_x:
                self.min_x = block[0]
            if block[2] > self.max_x:
                self.max_x = block[2]
            if block[1] < self.min_y:
                self.min_y = block[1]
            if block[3] > self.max_y:
                self.max_y = block[3]
            avg_h += (block[3] - block[1])
            avg_w += (block[2] - block[0])
        self.mid_x = self.min_x + (self.max_x - self.min_x) / 2
        self.mid_y = self.min_y + (self.max_y - self.min_y) / 2'''
        self.model = ocr_predictor(det_arch='db_resnet50', reco_arch='crnn_vgg16_bn',
                                   pretrained=True, export_as_straight_boxes=True)  # 'db_resnet50'
        self.model.compiled_metrics = None
        # self.doc = DocumentFile.from_images(filenames[0])
        self.morph = pymorphy2.MorphAnalyzer(lang='ru')


    def detect_image(self, image, flag=False):
        H, W = image.shape[:2]
        if flag == True:
            scale_down = 1.5
            image = cv2.resize(image, None, fx=scale_down, fy=scale_down, interpolation=cv2.INTER_LINEAR)
            #new_image = np.zeros((H * 5, W * 5, 3))
            print('sizes: ', H, ' ',W)
            '''scale_down = 0.6
            image = cv2.resize(image, None, fx = scale_down, fy = scale_down, interpolation= cv2.INTER_LINEAR)
            new_image = np.zeros((H*2, W*2, 3))
            new_image[:,:,:] = 255
            h,w = image.shape[:2]
            new_image[h:h*2, w:w*2,:] = image
            image = new_image.copy()
            plt.imshow(new_image)
            plt.show()'''
            """if W>H:
                print('no')
                new_image = np.zeros((H * 6, 3*W, 3))
                new_image[:,:,1] = 255
                for j in range(5):
                    new_image[j*H+H//2:(j+1)*H+H//2, W//2:W+W//2, :] = image
                    new_image[j * H + H // 2:(j + 1) * H + H // 2, W+W // 2:2*W + W // 2, :] = image
                '''new_image[H//2:H+H//2, W//2:W+W//2, :] = image
                new_image[H//2:H+H//2, W//2:W+W//2,:] = image
                
                new_image[H+H//2:2*H+H//2, :, :] = image
                new_image[2*H+H//2:3*H+H//2, :, :] = image
                new_image[3 * H + H // 2:4 * H + H // 2, :, :] = image
                new_image[4 * H + H // 2:5 * H + H // 2, :, :] = image'''
                image = new_image.copy()
                H,W = image.shape[:2]
                plt.imshow(image)
                for h in range(H):
                    for w in range(W):
                        print('new_image:', image[h,w,:])
                        if 0<sum(image[h,w,:])<3:
                            image[h,w,:] = [1,1,1]

                image = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2GRAY)
                plt.title('after black and white')
                plt.imshow(image)
                plt.show()
            else:
                print('yes')
                new_image = np.zeros((H, W * 6, 3))
                new_image[:, :, 1] = 255
                for j in range(5):
                    new_image[H // 2:H + H // 2, j * W + W // 2:(j + 1) * W + W // 2, :] = image
                    new_image[H + H // 2:2 * H + H // 2, j * W + W // 2:(j + 1) * W + W // 2, :] = image
                '''new_image[:, W // 2:W + W // 2, :] = image
                new_image[:, W + W // 2:2*W + W // 2, :] = image
                new_image[:, 2*W + W // 2:3*W + W // 2, :] = image
                new_image[:, 3 * W + W // 2:4 * W + W // 2, :] = image
                new_image[:, 4 * W + W // 2:5 * W + W // 2, :] = image'''
                image = new_image.copy()"""
                #image = cv2.resize(image, (W, W), interpolation=cv2.INTER_LINEAR)
                #H, W = image.shape[:2]
                #plt.imshow(image)
                #plt.show()
            #elif H//W >3:
            #    image = cv2.resize(image, (H,H), interpolation=cv2.INTER_LINEAR)
            #    #H, W = image.shape[:2]
            #image = cv2.resize(image, (3*W, 3*H), interpolation=cv2.INTER_CUBIC)
            #H, W = image.shape[:2]
            #new_image = np.zeros((H*2, W*2, 3))
            #new_image[:,:,:] = 255
            #new_image[:,:,1] = 255
            #new_image[H//2:H+H//2, W//2:W+W//2,:] = image
            #image = new_image.copy()
        img_path = 'save1.jpg'
        cv2.imwrite(img_path, image)
        #plt.show(image)
        doc = DocumentFile.from_images(img_path)

        result = self.model(doc)
        result.show(doc)
        # Show DocTR result
        if flag:
            input()
            #plt.imshow(image)
            #plt.show()
            #result.show(doc)
        json = result.export()

        blocks = []
        words = []
        res = self.look_geometry(json['pages'], blocks, words)

        self.mid_x = self.min_x + (self.max_x - self.min_x) / 2
        self.mid_y = self.min_y + (self.max_y - self.min_y) / 2

        self.mid_x = int(self.mid_x * W)
        self.mid_y = int(self.mid_y * H)

        self.max_y = int(self.max_y*H)
        self.min_y = int(self.min_y*H)
        self.min_x = int(self.min_x*W)
        self.max_x = int(self.max_x*W)

        self.avg_h = int(H * self.avg_h / len(blocks))
        self.avg_w = int(W * self.avg_w / len(blocks))

        blocks = res[0]
        words = res[1]

        #words = self.sort_by_geometry(words, blocks)
        # for block in json['pages']:
        #    self.look_geometry(block)

        os.system('rm save1.jpg')

        H, W = image.shape[:2]
        blocks = self.to_right_size(H, W, blocks)
        self.max_block_left = self.new_size(H,W, self.max_block)
        self.prev_max_block_left = self.new_size(H, W, self.prev_max_block)
        if self.max_block_left[3] < self.prev_max_block_left[3]:
            bl = self.max_block_left
            self.max_block_left = self.prev_max_block_left
            self.prev_max_block_left = bl
        # result.show(doc)

        blocks = [x for n, x in enumerate(blocks) if x not in blocks[:n]]
        words = [x for n, x in enumerate(words) if x not in blocks[:n]]

        return [blocks, words]

    def to_right_size(self, H, W, blocks):
        new_block = []
        for block in blocks:
            new_block.append(self.new_size(H, W, block))
        return new_block

    def new_size(self, H, W, block):
        return [int(block[0][0] * W), int(block[0][1] * H), int(block[1][0] * W), int(block[1][1] * H)]


    def sort_by_geometry(self, words, blocks):
        pass
        #for i in range(len(blocks)):
        #    for j in range(len(blocks)):



    def look_geometry(self, new_word, blocks, words):
        #print(new_word)
        for block in new_word:
            #if 'value' in block.keys():
            #    words.append(block['value'])
            if 'pages' in block.keys():
                res = self.look_geometry(block['pages'], blocks, words)
                blocks = res[0]
                words = res[1]
            if 'geometry' in block.keys():
                if 'value' in block.keys():
                    block_geom = block['geometry']
                    blocks.append(block_geom)
                    x0, x1 = block_geom[0][0], block_geom[1][0]
                    y0, y1 = block_geom[0][1], block_geom[1][1]
                    if (x1 - x0) > self.max_block[1][0]-self.max_block[0][0]:
                        self.prev_max_block = self.max_block
                        self.max_block = block_geom

                    if block_geom[0][0] < self.min_x:
                        self.min_x = block_geom[0][0]
                    if block_geom[1][0] > self.max_x:
                        self.max_x = block_geom[1][0]
                    if block_geom[0][1] < self.min_y:
                        self.min_y = block_geom[0][1]
                    if block_geom[1][1] > self.max_y:
                        self.max_y = block_geom[1][1]
                    self.avg_w += (block_geom[1][0] - block_geom[0][0])
                    self.avg_h += (block_geom[1][1] - block_geom[0][1])


                    words.append(block['value'])
                #print(block['geometry'])
            if 'blocks' in block.keys():
                res = self.look_geometry(block['blocks'], blocks, words)
                blocks = res[0]
                words = res[1]
            if 'lines' in block.keys():
                res = self.look_geometry(block['lines'], blocks, words)
                blocks = res[0]
                words = res[1]
            if 'words' in block.keys():
                res = self.look_geometry(block['words'], blocks, words)
                blocks = res[0]
                words = res[1]
        return [blocks, words]


class Tesseract:
    def __init__(self):
        pass

    def detect_text(self, img, params=1):
        """This function will handle the core OCR"""
        # rus = 'абвгдеёжзийклмнопрстуфхцчшщЪыьэюя'
        if params == 1:
            text = pt.pytesseract.image_to_string(img, lang='rus')
        else:
            text = pt.pytesseract.image_to_string(img, config='-l rus --psm 5 --oem 1 --dpi 300')
        # text = pt.pytesseract.image_to_string(img, lang='rus', config = f'--psm 12 --oem 3 -c tessedit_char_whitelist={rus}{rus.upper()}ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
        return text.strip().lower().capitalize()

    def ocr_core_boxes(self, img):
        boxes = pt.pytesseract.image_to_boxes(img)
        return boxes
