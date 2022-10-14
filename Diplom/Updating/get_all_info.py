import os
import time
import random

import cv2
import imutils
import matplotlib.pyplot as plt

from Optimize.updating import Update, show_img
from Optimize.Rotate_diplom import Rotate
from Optimize.SearchBD.correct_word import *
from Optimize.SearchBD.choose_right_BD import FindWord, create_specialities

try:
    from PIL import Image
except ImportError:
    print("No module Image")
    exit()
    #import Image
import pytesseract as pt


class Diplom:


    def __init__(self, path):
        self.result = dict()
        '''self.result = {
            'Университет:': '',
            'Место:': '',
            'Степень:': '',
            'Номер:': '',
            'Серия:': '',
            'Специальность:': '',
            'Номер специальности:':'',
            'Рег. номер:': '',
            'Дата выдачи:': '',
            'Фамилия:':'',
            'Имя:':'',
            'Отчество:':'',
        }'''
        self.doctr_model = None
        self.tes_model = None
        self.orig_image = cv2.imread(path)

    # 1
    def __modify_image(self, image, flag = 1):
        upd = Update()

        if flag == 1:
            rot = Rotate()

            image = rot.rotate(upd.resize_image(image), image)

        show_img(image, "Rotated")
        #upd.resize_image(image)
        #upd.find_contours(image)

        #upd.check_image(image)
        #show_img(image, 'Before Cropp')

        #upd.cropp_image(image)
        image = upd.corner_images(image)
        #show_img(image, 'Corner cropp')
        #upd.equa_CLAHE_change_hist(image)
        #upd.calc_hist(image)
        #upd.linear_change_hist(image)
        image = upd.unsharp_mask(image, sigma=1.5, threshold=1, amount=2.0, kernel_size=(3,3))
        show_img(image, "Unsharp mask")
        if flag == 2:
            image = upd.black_white(image)
        #show_img(image, 'Unsharp mask')
        return image

    # 2
    def __segmentation(self, image):
        H, W = image.shape[:2]
        self.doctr_model = Doctr(H, W)
        blocks, words = self.doctr_model.detect_image(image)
        print('words: ', words)
        print('blocks: ', blocks)
        print('length: ', len(blocks), len(words))
        return blocks, words

    def __check_digit(self, word):
        word = ''.join([w for w in word if w not in ".,':;"])
        count_digit = 0
        for w in word:
            if w.isdigit():
                count_digit+=1
        if len(word)<4:
            if count_digit == len(word):
                return True
            else:
                return False
        else:
            if count_digit >= len(word)//2:
                return True
        return False

    # 3
    def __symbol_blocks(self, blocks, words, img):
        self.tes_model = Tesseract()
        info = []
        H, W = img.shape[:2]


        left_blocks = []
        right_blocks = []

        max_block = self.doctr_model.max_block_left
        max_w = max_block[2] - max_block[0]

        #plt.imshow(img[max_block[1]:max_block[3], max_block[0]:max_block[2]])
        #plt.show()

        #avg_w = avg_w/len(blocks)
        #avg_h = avg_h/len(blocks)
        count_digit_left = 0
        min_gap = 0
        flag = [0, 0]
        left_max_blocks = [[[0,0,0,0],'',''] for i in range(3)]
        right_blocks = []
        new_words = []
        prev_block = 0
        min_gap_y = 0
        dist = 0
        blocks_len = 0
        specs = ['бакалавр', 'магистр', 'специалитет']
        rate_best = 0
        best_word = specs[0]
        for i in range(2):
            info1 = []
            parts = []
            for ind, block in enumerate(blocks):
                if (i == 0 and block[2] < self.doctr_model.mid_x) or (i == 1 and block[2] > self.doctr_model.mid_x):
                    print('block: ', block)

                    #if (block[3] - block[1]) < avg_h*0.4 or (block[2] - block[0]) < avg_w*0.4:
                    #    continue
                    #if (block[3] - block[1])*(block[2]-block[0]) < 200:
                    #    continue

                    block_img = img[block[1]:block[3], block[0]:block[2]]
                    if i == 1:
                        if flag == 0:
                            if self.rightBD.check_digit(words[ind]) and ind>10:
                                parts += left_max_blocks
                                info1.append(parts.copy())
                                parts.clear()
                                word = self.__text_recognition(block_img, self.tes_model)
                                parts.append([word, words[ind]])
                                info1.append(parts.copy())
                                parts.clear()
                                flag = 1
                                prev_block = block
                                dist = blocks[ind + 1][3] - block[1]
                                min_gap = (blocks[ind + 1][2] - block[0]) / 6
                                min_gap = (block[2] - block[0]) / 6
                                min_gap_y = (block[3] - block[1]) / 2
                                # min_gap_y = (blocks[ind+1][3] - block[1])/3
                                continue

                            y = block[3] - block[1]
                            for j in range(len(left_max_blocks)):
                                gap = left_max_blocks[j][0][3] - left_max_blocks[j][0][1]
                                if y > gap:
                                    for z in range(len(left_max_blocks) - 1, j, -1):
                                        left_max_blocks[z] = left_max_blocks[z - 1].copy()
                                    left_max_blocks[j][0] = block
                                    word = self.__text_recognition(block_img, self.tes_model)
                                    another_word = words[ind]
                                    left_max_blocks[j][1] = word
                                    left_max_blocks[j][2] = another_word
                                    break
                            continue
                        elif flag == 1:
                            if abs(prev_block[3] - block[3]) <= min_gap_y:
                                word = self.__text_recognition(block_img, self.tes_model)
                                parts.append([words[ind], word])
                                prev_block = block
                            else:
                                info1.append(parts.copy())
                                parts.clear()
                                flag += 1
                                prev_block = block
                                blocks_len += 1
                            continue
                        elif flag == 2:
                            block_img = img[prev_block[1]:prev_block[3], prev_block[0]:prev_block[2]]
                            block_img = self.__modify_image(block_img, flag=2)
                            word = self.__text_recognition(block_img, self.tes_model)
                            if word != '':
                                for word_true in specs:
                                    rate = self.rightBD.cmp_rates(word.lower(), word_true)
                                    if rate > rate_best:
                                        best_word = word_true
                                        rate_best = rate
                                if rate_best > 0.8:
                                    parts.append([words[ind - 1], best_word])
                                    # info1.append(parts.copy())
                                    break
                            prev_block = block
                            '''if abs(prev_block[3] - block[3]) < min_gap_y and abs(prev_block[2] - block[0]) < min_gap:
                                blocks_len += 1
                                prev_block = block
                            else:
                                if blocks_len == 1:
                                    block_img = img[prev_block[1]:prev_block[3], prev_block[0]:prev_block[2]]
                                    word = self.__text_recognition(block_img, self.tes_model)
                                    if word != '':
                                        for word_true in specs:
                                            rate = self.rightBD.cmp_rates(word.lower(), word_true)
                                            if rate > rate_best:
                                                best_word = word_true
                                                rate_best = rate
                                        if rate_best > 0.8:
                                            parts.append([words[ind - 1], best_word])
                                            #info1.append(parts.copy())
                                            break
                                    #parts.append([words[ind - 1], word])
                                    #flag +=1
                            prev_block = block
                            blocks_len = 1'''
                            continue

                    elif flag[1] == 1 and i == 0 and self.__compare(block, self.doctr_model.max_block_left):
                        parts.append(words[ind])
                        info1.append(parts.copy())
                        parts.clear()
                        flag[1]+=1
                        continue

                    elif self.__compare(block, self.doctr_model.prev_max_block_left) and i==0:
                        info1.append(parts.copy())
                        parts.clear()
                        parts.append(words[ind])
                        flag[1] += 1
                        continue

                    elif block[3] > max_block[3] and i == 0 and flag[1] == 2:
                        if self.__check_digit(words[ind]) and count_digit_left < 2:
                            if len(words[ind]) < 3:
                                continue
                            count_digit_left +=1
                            if count_digit_left == 1:
                                min_gap = (block[2]-block[0])/6
                            #parts.append(words[ind])
                            block_img = self.__modify_image(block_img, flag=2)

                            word = self.__text_recognition(block_img, self.tes_model)
                            parts.append([words[ind], word])
                            if count_digit_left == 2:
                                info1.append(parts.copy())
                                parts.clear()
                            continue
                        elif self.__check_digit(words[ind]) and count_digit_left >=2:
                            if len(words[ind]) < 2:
                                continue
                            left_gap = block[0] - min_gap
                            right_gap = block[2] + min_gap
                            if blocks[ind-1][0]<=left_gap<=blocks[ind-1][2]:
                                prev_block = img[blocks[ind-1][1]:blocks[ind-1][3],
                                             blocks[ind-1][0]:blocks[ind-1][2]]
                                prev_block = self.__modify_image(prev_block, flag=2)
                                word = self.__text_recognition(prev_block, self.tes_model)
                                for j in range(len(parts)):
                                    if words[ind-1] == parts[j][0]:
                                        break
                                else:
                                #if [words[ind-1], word] not in parts:
                                    parts.append([words[ind-1], word])
                            block_img = self.__modify_image(block_img, flag=2)
                            word = self.__text_recognition(block_img, self.tes_model)
                            parts.append([words[ind], word])
                            if len(blocks) != ind+1:
                                if blocks[ind+1][0]<=right_gap<=blocks[ind+1][2]:
                                    next_block = img[blocks[ind+1][1]:blocks[ind+1][3],
                                                 blocks[ind+1][0]:blocks[ind+1][2]]
                                    next_block = self.__modify_image(next_block, flag=2)
                                    word = self.__text_recognition(next_block, self.tes_model)
                                    for j in range(len(parts)):
                                        if words[ind +1] == parts[j][0]:
                                            break
                                    else:
                                    #if [words[ind + 1], word] not in parts:
                                        parts.append([words[ind+1], word])
                            continue
                        else:
                            continue
                        #plt.imshow(block_img)
                        #plt.show()
                            #parts.append()




                    block_img = self.__modify_image(block_img, flag = 2)
                    #show_img(block_img, 'block_img')

                    word = self.__text_recognition(block_img, self.tes_model)
                    #print("text: ", word)
                    parts.append(word)
                else:
                    new_words.append(words[ind])
                    right_blocks.append(block)
                #print("text: ", tes.detect_text(block_img))
                #block_img = img[block]
                #tes.detect_text(block)

            info1.append(parts.copy())
            info.append(info1.copy())
            #break
            blocks = right_blocks[:(len(right_blocks)*3)//4]
            words = new_words[:(len(new_words)*3)//4]
            info1.clear()
            flag = 0
            #flag[0] = 1
            #flag[1] = 0

        return info


    def __compare(self, block1, block2):
        for i in range(len(block1)):
            if block1[i] != block2[i]:
                return False
        return True

    # 3.5
    def look_digits(self, word):
        digit_count = 0
        for w in word:
            if w.isdigit():
                digit_count+=1
        if digit_count > 4:
            return True
        else:
            return False
    # 4
    def __text_recognition(self, block, model):
        #doc = Doctr()
        #upd = Update()
        #block = upd.unsharp_mask(block, sigma=1.5, threshold=1, amount=2.0, kernel_size=(3, 3))
        #block = upd.bright_contrast(block, bright=0.5, contrast=1.2)
        word = model.detect_text(block)
        """if self.look_digits(word):
            print('model word: ', word)
            doc = Doctr()
            upd = Update()
            block = upd.unsharp_mask(block, sigma=1.5, threshold=1, amount=2.0, kernel_size=(3, 3))
            block = upd.bright_contrast(block, bright = 0.5, contrast=1.2)
            #block =
            #blocks, words = doc.detect_image(block, True)
            for w in words:
                print('Doctr find digits: ', w)"""
        return word

    # 5.1
    def __chooseBD_left(self, info):
        words = []
        #rightBD = FindWord()
        for i in range(len(info)):
            chapter = []
            #for part in info[i]:
            if i == 0:
                new_word = info[i][:-1]
                new_word = [x for x in new_word if x != '']
                new_word = ' '.join(new_word)
                cities = self.rightBD.findCity(info[i][-2:])
                universities = self.rightBD.findUniversity(new_word, cities[0])
                self.result['Места:'] = cities
                self.result['Университеты:'] = universities
            if i == 1:
                word = info[i][-1]
                level = self.rightBD.findLevel(word)
                self.result['Степень:'] = level
            if i == 2:
                number = info[i][0][0]
                series = info[i][1][0]
                number, series = self.rightBD.findNumberSeries(number, series)
                self.result['Серия:'] = series
                self.result['Номер:'] = number
            if i == 3:
                reg_number = info[i][0]
                data_given = info[i][1:]
                reg_number, datas_given = self.rightBD.findRegNumbDataGiven(reg_number, data_given)
                self.result['Рег. номер:'] = reg_number
                self.result['Даты выдачи:'] = datas_given
                print("\n\nРезультат:\n", self.result, "\n\n")
                return
            #new_word = rightBD.compare_strings(new_word)
            if new_word == None:
                print('None word: ', new_word)
            chapter.append(new_word)

            print('clean_string')
            # ДОДДЕЛАТЬ(алгоритм удаления одинаковых подстрок)
            #chapter = rightBD.clean_string(chapter)
        words.append(chapter)

        d = random.randint(0, 100)
        file = open(f'result_{d}.txt','w')
        for chapter in words:
            for word in chapter:
                file.write(word)
                file.write('\n')

    # 5.2
    def __chooseBD_right(self, info):
        words = []
        for i in range(len(info)):
            if i == 0:
                family, name, surname = info[i][0], info[i][0], info[i][0]
                left = info[i][0][0][0]
                up = info[i][0][0][1]
                right = info[i][0][0][2]
                for j in range(len(info[i])):
                    if info[i][j][0][0] < left:
                        left = info[i][j][0][0]
                        name = info[i][j]
                    if info[i][j][0][1] < up:
                        up = info[i][j][0][1]
                        family = info[i][j]
                    if info[i][j][0][2] > right:
                        right = info[i][j][0][2]
                        surname = info[i][j]

                fio = self.rightBD.findFIO([family[1], name[1], surname[1]])
                self.result['Фамилия:'] = fio[0]
                self.result['Имя:'] = fio[1]
                self.result['Отчество:'] = fio[2]


            if i == 1:
                spec_number = info[i][0]
                spec_name = info[i+1]
                spec_number, spec_name = self.rightBD.findSpeciality(spec_number, spec_name)
                self.result['Специальность:'] = spec_name[0]
                self.result['Номер специальности:'] = spec_number

            if i == 3:
                cmp_speciality = info[i][0]
                #level_0 = self.rightBD.findLevel(cmp_speciality[0])
                level_1 = self.rightBD.findCmpLevel(cmp_speciality[1], self.result['Степень:'])
                if self.result['Степень:'] == '' and level_1 != None:
                    self.result['Степень:'] = level_1
                #self.result['Степень:'] = level
                break
        return

    def conveer(self):

        #Trye

        # 1
        self.rightBD = FindWord()
        image = self.__modify_image(self.orig_image)
        show_img(image, 'after modifing')
        # 2
        text_blocks, text_words = self.__segmentation(image)

        # 3
        info = self.__symbol_blocks(text_blocks, text_words, image)

        print('left chapter:', info[0])
        print('rifht chapter:', info[1])
        # 4
        info_0 = self.__chooseBD_left(info[0])
        info_1 = self.__chooseBD_right(info[1])
        print('\nРезультат:\n', self.result)
        return info
