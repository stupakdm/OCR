import os

# from doctr.io import read_img_as_numpy, read_img_as_tensor
# import tensorflow as tf
import cv2

import pymorphy2
from doctr.io import DocumentFile  # , read_img_as_numpy, read_img_as_tensor
from doctr.models import ocr_predictor

from updating_diplom.optimize_diplom.searchBD_diplom.words_modules import CorrectFunctions


class Doctr:
    def __init__(self):
        '''
        Конструктор
        Запускается в самом начале
        Здесь передаются все изначальные параметры, которые впоследствии не изменяются
        '''

        self.correct_functions = CorrectFunctions()
        self.avg_w, self.avg_h = None, None
        self.mid_y, self.mid_x = None, None
        self.max_y, self.max_x = None, None
        self.min_y, self.min_x = None, None
        self.max_blocks = None
        self.max_square = None
        self.max_block = None
        self.prev_max_block = None
        self.model = ocr_predictor(det_arch='db_resnet50', reco_arch='crnn_vgg16_bn',
                                   pretrained=True, export_as_straight_boxes=True)  # 'db_resnet50'
        self.model.compiled_metrics = None
        # self.doc = DocumentFile.from_images(filenames[0])
        self.morph = pymorphy2.MorphAnalyzer(lang='ru')

    def initialize(self, H, W):
        '''
        Запускается при каждой новой фотографии
        Обнуляет параметры снизу
        H: длина изображения
        W: ширина изображения
        '''
        self.prev_max_block = [(0,0),(0,0)]
        self.max_block = [(0,0),(0,0)]
        self.max_blocks = [[(0,0),(0,0)],[(0,0),(0,0)],[(0,0),(0,0)],[(0,0),(0,0)],[(0,0),(0,0)]]
        self.max_square = 0
        self.min_x = W
        self.max_x = 0
        self.min_y = H
        self.max_y = 0
        self.mid_x, self.mid_y = 0, 0
        self.avg_h = 0
        self.avg_w = 0




    def detect_image(self, image, flag=False):
        '''
        Распознавание текстовых блоков на изображении
        image: 3-мерное изображение
        '''
        H, W = image.shape[:2]
        if flag:
            scale_down = 1.5
            image = cv2.resize(image, None, fx=scale_down, fy=scale_down, interpolation=cv2.INTER_LINEAR)
            #new_image = np.zeros((H * 5, W * 5, 3))
            print('sizes: ', H, ' ',W)


        img_path = 'save1.jpg'
        cv2.imwrite(img_path, image)
        doc = DocumentFile.from_images(img_path)

        result = self.model(doc)
        #result.show(doc)
        # Show DocTR result
        #if flag:
        #    input()
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

        # for block in json['pages']:
        #    self.look_geometry(block)

        os.system('rm save1.jpg')

        H, W = image.shape[:2]
        blocks = self.correct_functions.to_right_size(H, W, blocks)
        self.max_blocks = self.correct_functions.to_right_size(H, W, self.max_blocks)
        ind = 0
        for block in self.max_blocks:
            if block[0] < self.mid_x and ind == 0:
                self.max_block_left = block.copy()
                ind += 1
            elif block[0] < self.mid_x and ind == 1:
                self.prev_max_block_left = block.copy()
                break
        #self.max_block_left = self.new_size(H,W, self.max_block)
        #self.prev_max_block_left = self.new_size(H, W, self.prev_max_block)
        #plt.imshow(image[self.max_block_left[1]:self.max_block_left[3], self.max_block_left[0]:self.max_block_left[2]])
        #plt.title("Max_block")
        #plt.show()
        if self.max_block_left[3] < self.prev_max_block_left[3]:
            bl = self.max_block_left
            self.max_block_left = self.prev_max_block_left
            self.prev_max_block_left = bl

        blocks = [x for n, x in enumerate(blocks) if x not in blocks[:n]]
        words = [x for n, x in enumerate(words) if x not in blocks[:n]]

        left_block, right_block = self.correct_functions.divide_left_right_words(blocks, words, self.mid_x)
        left_blocks, left_words = left_block[0], left_block[1]
        right_blocks, right_words = right_block[0], right_block[1]

        blocks.clear()
        words.clear()

        left_blocks, left_words = self.correct_functions.sort_by_geometry(left_blocks, left_words)
        right_blocks, right_words = self.correct_functions.sort_by_geometry(right_blocks, right_words)

        left_block = (left_blocks, left_words)
        right_block = (right_blocks, right_words)
        return [left_block, right_block]




    def look_geometry(self, new_word, blocks, words):
        '''
        Нахождение текстовых блоков среди всего списка блоков, найденных docTR
        new_word: список содеражащий параметры для каждого распознаного слова
        blocks: список блоков, которые будут возвращены
        words: Список слов, которые будут возвращены
        '''
        for block in new_word:

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
                    square = (x1 - x0)*(y1- y0)
                    for i in range(len(self.max_blocks)):
                        max_square = (self.max_blocks[i][1][0]-self.max_blocks[i][0][0])*(self.max_blocks[i][1][1]-self.max_blocks[i][0][1])
                        if square >= max_square:
                            for j in range(len(self.max_blocks)-1, i, -1):
                                self.max_blocks[j] = self.max_blocks[j-1].copy()
                            self.max_blocks[i] = list(block_geom)
                            break
                    #if (x1 - x0)*(y1- y0) > self.max_square:
                    #    self.prev_max_block = self.max_block
                    #    self.max_block = block_geom
                    #    self.max_square = (self.max_block[1][0]-self.max_block[0][0])*(self.max_block[1][1]-self.max_block[0][1])

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




