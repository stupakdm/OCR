import os, time, re

# from doctr.io import read_img_as_numpy, read_img_as_tensor
# import tensorflow as tf
import random

import cv2
import enchant
#import matplotlib.pyplot as plt
import numpy as np
from pymorphy2 import MorphAnalyzer
from doctr.io import DocumentFile  # , read_img_as_numpy, read_img_as_tensor
from doctr.models import ocr_predictor

from passport.modify_passport.image_updating_passport.findBD_passport.chooseBD_ps import Search
from passport.modify_passport.image_updating_passport.findBD_passport.useful_functions_ps import Correct_Word_Fuctions

class Doctr:
    '''
    Класс функций для распознавания текста на изображении,
    для транслита с английского на русский

    '''
    delimiters = '.', ',', ';'

    comp = re.compile('[^аеоиыуяэюАЕОИЫУЯЭЮ]+[^ьъЬЪ]+')

    letters_to_num = {
        'A': '4',
        'O': '0',
    }
    alphabet = {
        'A': ['Д', 'А', 'Л', 'Я', 'И'],
        'B': ['В', 'Б', 'Ы', 'Ь'],
        'B|': ['Ы'],
        'C': ['С'],
        'D': ['О', 'Ф'],
        'E': ['Е', 'В', 'Б'],
        'F': ['Г'],
        'G': ['О'],
        'Hl': ['П'],
        'H': ['Н', 'И', 'П', 'Ч'],
        '/I': ['Д'],
        'II': ['П'],
        'JI': ['Л', 'П'],
        'JL': ['Л'],
        'IL': ['П', 'Ц', 'Н', 'Щ'],
        'I': ['П', 'Д', 'Г', 'Л', 'И'],
        'J': ['У', 'Л', 'Я'],
        'K': ['К'],
        'L': ['Ь', 'Ц'],
        'M': ['М', 'И', 'Й'],
        'N': ['И', 'Й', 'Н'],
        'O': ['О'],
        'P': ['Р'],
        'Q': ['О'],
        'R': ['Я', 'К'],
        'SI': ['Я'],
        'S': ['Я'],
        'TI': ['П'],
        'T': ['Т', 'Л', 'П', 'Г'],
        'U': ['И', 'Н', 'Ц', 'О'],
        'V': ['У', 'Х', 'Ж', 'И'],
        'W': ['У', 'Х'],
        'X': ['Х', 'У', 'Ж', 'Д'],
        'Y': ['У', 'Х', 'Ж', 'Ч'],
        'Z': ['И', 'Й'],
        '/': ['Д'],
        'a': ['а', 'з'],
        'b': ['о', 'б', 'ь'],
        'c': ['с'],
        'd': ['о', 'a'],
        'e': ['е', 'с'],
        'f': ['г', 'р'],
        'g': ['ф'],
        'h': ['п', 'н'],
        'ii': ['п'],
        'ji': ['л', 'п'],
        'il': ['п', 'ц', 'щ'],
        'i': ['п'],
        'j': ['я'],
        'k': ['к'],
        'l': ['ь', 'б'],
        'm': ['п', 'ш', 'щ'],
        'n': ['п', 'и', 'й', 'н'],
        'o': ['о', 'р'],
        'p': ['р', 'о'],
        'q': ['о', 'a'],
        'r': ['г', 'т', 'л'],
        's': ['я', 'с', 'д'],
        't': ['т'],
        'u': ['п', 'н'],
        'v': ['л', 'п'],
        'w': ['л', 'и', 'м'],
        'x': ['х', 'л', 'к', 'ж', 'д'],
        'y': ['у', 'х'],
        'z': ['я', 'г'],
        '4': ['4', 'Ч'],
        '3': ['3', 'З', 'з'],
        '1': ['1', 'И'],
        '2': ['2'],
        '5': ['5'],
        '6': ['6', 'Б'],
        '7': ['7'],
        '8': ['8'],
        '9': ['9', 'Ч'],
        '0': ['0', 'О'],
        '#': ['Ж', 'Ф'],
        '*': ['Ж', 'Ф']
    }
    prob_words = []

    filenames = []

    def __init__(self):
        self.searching = Search()

        self.correct_functions = Correct_Word_Fuctions()

    def init_sizes(self):
        self.mid_x = 0
        self.mid_y = 0
        self.max_x, self.max_y = 0, 0
        self.min_x, self.min_y = 1, 1
        self.perimetr = 0

    def choose_model(self, straight_page=False, predict=False):
        # if predict == True:
        # self.model = crnn_vgg16_bn(pretrained=True)
        # else:
        # self.model = linknet_resnet18(pretrained=True)
        self.model = ocr_predictor(det_arch='db_resnet50', reco_arch='crnn_vgg16_bn',
                                   pretrained=True, export_as_straight_boxes=True)  # 'db_resnet50'
        self.model.compiled_metrics = None
        # self.doc = DocumentFile.from_images(filenames[0])
        self.morph = MorphAnalyzer(lang='ru')

        #sogl1 = 'йцкгшщзхфвпрджчмьб'
        #sogl2 = 'нслт'

        #self.gl = 'уеёаоэяию'
        #self.sogl = sogl1 + sogl1.upper() + sogl2 + sogl2.upper()
        #self.gl += self.gl.upper()
        self.dict = enchant.Dict('ru_RU')
        self.init_sizes()

    # def add_files(self, filename):
    #    self.filenames.append(filename)
    def poss_words1(self, word):
        count_num = 0
        count_let = 0
        # word = word.lower()
        x = [0 for i in word if i.isalpha()]
        y = [0 for i in word if i.isdigit()]
        count_let = len(x)
        count_num = len(y)
        flag = 0
        if count_let >= count_num:
            flag = 1

        j = 0
        l = len(word)
        while j < l:
            if flag == 1:
                if word[j].isdigit():
                    word = word[0:j] + word[j + 1:]
                    l = len(word)
                    continue
            else:
                if word[j].isalpha():
                    word = word[0:j] + word[j + 1:]
                    l = len(word)
                    continue
            j += 1
        return word, flag

    def poss_word(self, words):

        count_num = 0
        count_let = 0
        for word in words:
            word = word.lower()
            x = [0 for i in word if i.isalpha()]
            count_let += len(x)

            x = [0 for i in word if i.isdigit()]
            count_num += len(x)

        i = 0
        length = len(words)
        if count_num > count_let:
            flag = 0
        else:
            flag = 1

        while i < length:
            if flag == 0:
                x = [j for j in words[i] if not j.isalpha()]
                # x = [0 for j in words[i] if j.isalpha()]
                # count_let += len(x)
                # if count_let > 0:
                #    del words[i]
                #    l = len(words)
                # else:

                i += 1
                # count_let = 0
            else:
                x = [0 for j in words[i] if j.isdigit()]
                count_num += len(x)
                if count_num > 0:
                    del words[i]
                    length = len(words)
                else:

                    i += 1
                count_num = 0

        return words, flag

    def check_before_translate(self, word, size):
        if len(word) > size:
            word = word[:size]
        if self.correct_functions.check_digit(''.join(word)) and '.' not in word and len(word) > 10:
            return False
        self.translate_word(word, [])
        return True

    def translate_word(self, word, words):
        all_words = []
        for (ind, letter) in enumerate(word):
            if letter in self.alphabet.keys():
                if len(self.alphabet[letter]) > 1:
                    for j in range(0, min(3, len(self.alphabet[letter]))):  # Органичим на всего 3 буквы можно заменять
                        t = words.copy()
                        t.append(self.alphabet[letter][j])
                        all_words += self.translate_word(word[ind + 1:len(word)], t)
                words.append(self.alphabet[letter][0])
            else:
                words.append(letter)

        # joint = ''.join(words).lower().capitalize()
        joint_orig = ''.join(words)

        all_words.append(joint_orig)
        # else:
        self.prob_words.append(joint_orig)


        return all_words




    def translating(self, words, flag):

        possible_words = []
        #dictionary = {}

        # Вывод кол-ва слов для оценки производительности
        # print("words: ", words)
        # if len(words) != 0:
        #
        # print('len(words[0]):', len(words[0]))

        # Filter english words before filter
        # print('words before filter:', words)

        words = self.correct_functions.clean_translating_words(words)
        #words = self.clean_translating_words(words)
        # print('words after filter:', words)
        if len(words) == 0:
            return ''
        #t1 = time.time()
        limit = 50000  # Лимит по кол-ву слов на обработку
        if len(words[0]) > 11:
            limit = 50000
        if len(words[0]) > 18:
            return ''
        # print('len(words):', len(words))
        # input()
        print("words: ", words)
        for word in words:
            print("word: ", word)
            if not self.check_before_translate(word, 11):
                return ''

        #if len(possible_words) == 0:
        possible_words = self.prob_words.copy()
        self.prob_words = []

        # Time for this operation

        limit = 20000
        if len(possible_words) > limit:
            possible_words = np.array(possible_words)
            possible_words = list(np.random.choice(possible_words, size=limit))
            # possible_words = possible_words[:75000]

        # print('poss_words2: ', len(possible_words))
        if flag == 0:
            return ''.join(possible_words[0])

        #dict = enchant.Dict('ru_RU')

        max_seq = -1
        all_words = [[]]
        max_pos_words = ['']

        t1 = time.time()
        if flag == 1:
            if len(possible_words[0]) >= 15:
                possible_word = self.correct_functions.median(possible_words)
                #possible_word = self.median(possible_words)
                return possible_word
            if len(possible_words) > limit:
                possible_words = np.array(possible_words)
                possible_words = list(
                    np.random.choice(possible_words, size=limit))  # Ограничиваем кол-во слов после обработки
            # print('poss words: ', possible_words)
            if len(possible_words) > 10000:
                possible_words = list(np.random.choice(possible_words, size=10000 // len(possible_words[0])))
            possible_words = self.correct_functions.filt_words(possible_words)
            #possible_words = self.filt_words(possible_words)

            # possible_words = list(map(self.remove_signs, possible_words))
            # print('after filter: ', len(possible_words))
            if len(possible_words) != 0:
                if '.' in possible_words[0]:
                    all_words = [[] for i in possible_words[0].split('.')]
                    max_pos_words = ['' for i in all_words]
            else:
                return ''

        best_ratio = 0
        median_words = []

        limit_poss_words = 200
        possible_words = self.correct_functions.limit_words(limit_poss_words, possible_words)
        #possible_words = self.limit_words(limit_poss_words, possible_words)


        h = int(limit_poss_words//10)
        for i in range(0, len(possible_words), h):
            word_cmp = self.correct_functions.quickmedian(possible_words[i:i+h])
            #word_cmp = self.quickmedian(possible_words[i:i + h])
            w = [word_cmp]
            if len(all_words) > 1:
                w = word_cmp.split('.')
                w = [i.lower().capitalize() for i in w]
            for (ind1, s1) in enumerate(w):
                if s1 == '':
                    max_pos_words[ind1] = ''
                    all_words[ind1].append(s1)
                    continue
                d = self.dict.suggest(s1)
                # self.printstring(d, 'Dict suggest')
                # print('Dict suggest: ', d)
                if len(d) != 0:
                    opcodes = []
                    all_opcodes = {}
                    for d_word in d:
                        op = self.correct_functions.opcodes(d_word, s1)
                        #op = self.opcodes(d_word, s1)
                        opcodes.append(op)
                        if op in all_opcodes.keys():
                            all_opcodes[op].append(d_word)
                        else:
                            all_opcodes[op] = [d_word]
                    op = min(opcodes)
                    new_word = self.correct_functions.quickmedian(all_opcodes[op])
                    #new_word = self.quickmedian(all_opcodes[op])
                    # print(d)
                    # if d == None:

                    # print('d:', d)
                    # input()
                    # median_words += d
                    ratio = self.correct_functions.ratio(s1, new_word)
                    #ratio = self.ratio(s1, new_word)
                    # print('ratio: ', ratio, s1)
                    if ratio > max_seq:
                        max_seq = ratio
                        max_pos_words[ind1] = s1
                    if ratio >= 0.7:
                        if ratio == 1.0:
                            all_words[ind1] = [s1]
                            best_ratio = 1
                            break
                        all_words[ind1].append(s1)
                        all_words[ind1].append(new_word)


        all_words = self.correct_functions.concatenate(all_words, max_pos_words)
        #all_words = self.concatenate(all_words, max_pos_words)

        if flag == 0:
            possible_word = '.'.join(all_words)
        else:
            possible_word = '.'.join(all_words)

        # if
        return possible_word



    def divide_word(self, word, words):
        all_words = []
        for i in range(0, len(word)):
            left_word = word[i:len(word)]
            if len(left_word) >= 2:
                if left_word[0:2] in self.alphabet.keys():
                    t = words.copy()
                    t.append(left_word[0:2])
                    all_words += self.divide_word(left_word[2:len(left_word)], t)
            words.append(left_word[0])
        all_words.append(words)
        return all_words

    def sort_blocks_geometry(self, blocks, words):
        i = 0
        words_fam = []
        words_fms = []
        l = len(blocks)
        x_min = self.min_x + (self.max_x - self.min_x) / 5
        x_max = self.max_x - (self.max_x - self.min_x) / 5
        mid_y = (self.max_y + self.min_y) / 2
        self.mid_x = self.mid_x / len(blocks)
        self.mid_y = self.mid_y / len(blocks)

        middle = self.min_x + (self.max_x - self.min_x) / 2

        #  Можно попробьовать обрезать сначала для фмс, потом по фамилиям: измерять среднее для каждого из них

        while i < l:
            x = blocks[i][1][0] - blocks[i][0][0]
            y = blocks[i][1][1] - blocks[i][0][1]  # За пределы нашего паспорта
            if blocks[i][0][0] > x_max or blocks[i][1][0] < x_min \
                    or x < self.mid_x / 1.35 or y < self.mid_y / 1.35 \
                    or x > self.mid_x * 3 or y > self.mid_y * 3:  # # Если границы блока сильно меньше средних значений и Если границы блока сильно больше средних значений
                del blocks[i]
                del words[i]
                l = len(blocks)
            else:
                if blocks[i][0][1] < mid_y:
                    if self.correct_functions.check_digit(words[i]):
                    #if self.check_digit(words[i]):
                        words_fms.append(words[i])
                else:
                    if blocks[i][1][0] >= middle:
                        if (blocks[i][1][0] - middle) / (blocks[i][1][0] - blocks[i][0][0]) > 0.3:
                            # if (blocks[i][1][0]+ blocks[i][0][0])/2 >= (middle - 0.02*middle):
                            words_fam.append(words[i])
                i += 1

        return [blocks, words_fms, words_fam]

    def clean_words(self, words):
        j = 0
        l = len(words)

        while j < l:
            if len(words[j]) != 0:
                if len(words[j]) < 3 and not words[j][0].isdigit():
                    if j + 1 != len(words):
                        words[j] += words[j + 1]
                        del words[j + 1]
                        l = len(words)
                    else:
                        j += 1
                else:
                    j += 1
            else:
                del words[j]
                l = len(words)
        return words

    def look_geometry(self, new_word, blocks, words):
        # print(new_word)
        for block in new_word:
            if 'geometry' in block.keys():
                if 'value' in block.keys():
                    blocks.append(block['geometry'])
                    x0, x1 = block['geometry'][0][0], block['geometry'][1][0]
                    y0, y1 = block['geometry'][0][1], block['geometry'][1][1]

                    self.mid_x += (x1 - x0)
                    self.mid_y += (y1 - y0)
                    if x1 < self.min_x:
                        self.min_x = x1
                    if x0 > self.max_x:
                        self.max_x = x0
                    if y1 < self.min_y:
                        self.min_y = y1
                    if y0 > self.max_y:
                        self.max_y = y0

                    words.append(block['value'])
            if 'pages' in block.keys():
                res = self.look_geometry(block['pages'], blocks, words)
                blocks = res[0]
                words = res[1]
                # print(block['geometry'])
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

    '''def printstring(self, string, about):
        pass'''
        # print(f"{about}: {string}")

    @staticmethod
    def to_right_size(H, W, blocks):
        new_block = []
        for block in blocks:
            new_block.append([int(block[0][0] * W), int(block[0][1] * H), int(block[1][0] * W), int(block[1][1] * H)])
        return new_block

    def find_contours(self, img, flag=0):
        img_path = 'save1.jpg'
        cv2.imwrite(img_path, img)

        doc = DocumentFile.from_images(img_path)

        result = self.model(doc)
        # Show DocTR result
        #result.show(doc)
        json = result.export()

        blocks = []
        words = []
        H, W = img.shape[:2]
        res = self.look_geometry(json['pages'], blocks, words)
        blocks = res[0]
        words = res[1]
        if flag == 0:
            blocks, words_fms, words_fam = self.sort_blocks_geometry(blocks, words)
        # print('blocks', blocks)
        # for block in json['pages']:
        #    self.look_geometry(block)

        os.system('rm save1.jpg')

        H, W = img.shape[:2]
        blocks = self.to_right_size(H, W, blocks)
        # result.show(doc)
        # print(len(blocks), len(words))

        blocks = [x for n, x in enumerate(blocks) if x not in blocks[:n]]
        # words = [x for n, x in enumerate(words) if x not in blocks[:n]]
        # print(blocks)
        # print(words)
        if flag == 1:
            return [blocks, words]
        else:
            return [blocks, words_fms, words_fam]



