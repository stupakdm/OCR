# import asyncio
#import os
import time
#import random

#import cv2
from imutils import rotate_bound
# from pyinstrument import Profiler

from modify_passport.image_updating_passport.Rotating_ps import show_img
from modify_passport.image_updating_passport.findBD_passport.chooseBD_ps import Search
from modify_passport.image_updating_passport.findBD_passport.correct_word import Doctr
#from passport.Modify_ps.Image_update_ps.red_contour import Passport1
from modify_passport.info_functions_ps import Info_functions, Pytesseract
# from Image_update.updating_image import Update
# from Image_update.red_contour import Passport1
from multiprocessing import Pool, cpu_count

"""from transliterate import translit, get_translit_function"""

try:
    from PIL import Image
except ImportError:
    print("No module Image")
    exit()


class Passport:
    '''
    Класс, реализующий весь процесс OCR для паспорта
    '''

    filePath = False

    new_file_path = False

    person_data = {
        'имя': '',
        'фамилия': '',
        'отчество': '',
        'пол': '',
        'место': '',
        'фмс': '',
        'дата выдачи': '',
        'Код подразделения': '',
        'дата рождения': '',
        'серия': '',
        'номер': ''
    }

    def __init__(self):
        '''
        Инициализация всех основных параметров при первом запуске программы
        '''
        #self.check_flag = None
        self.doctr = Doctr()
        self.doctr.choose_model()
        self.search = Search()
        self.search.start()

        #self.update = Update()

        self.info_functions = Info_functions()
        self.tesseract = Pytesseract()
        #self.passport_rotate1 = Passport1()
        #Doctr.choose_model(self)
        #Search.start(self)
        self.workers = 1
        try:
            self.workers = cpu_count()
        except NotImplementedError:
            self.workers = 1
        if self.workers > 1:
            self.pool = Pool(processes=self.workers)
        #self.check_image = None
        self.check_region = False

    def init_everything(self, filepath):
        '''
        Инициализация переменных при каждой новой фотографии
        :param filepath:
        :return:
        '''
        self.filePath = filepath
        self.doctr.init_sizes()
        # Search


    #   50/50
    # 3 Finding FIO
    def findingBoxes(self, img, key=1):
        '''
        Нахождение текстовых блоков на изображении
        :param img:
        :param key:
        :return:
        '''
        boxes = None

        boxes = self.doctr.find_contours(img)
        boxes = [x for n, x in enumerate(boxes) if x not in boxes[:n]]

        return boxes

    #   50/50
    # 4 Detecting Text


    def cleaning_words(self, img, flag):
        words_fms = 0
        words_fam = 0
        text = 0
        res = self.doctr.find_contours(img, flag)
        # boxes = res[0]
        if flag == 0:
            words_fms = res[1]
            words_fam = res[2]
        else:
            text = res[1]
        # print("text: ", text)
        # all_words = []
        if flag == 1:
            length = len(text)
            i = 0
            while i < length:
                for j in text[i]:
                    if j.isalpha():
                        del text[i]
                        length = len(text)
                        break
                else:
                    i += 1

        else:
            words_fms = self.doctr.correct_functions.clean_words(words_fms)
            words_fms = self.doctr.correct_functions.filter_words(words_fms)
            words_fam = self.doctr.correct_functions.clean_words(words_fam)
            words_fam = self.doctr.correct_functions.filter_words(words_fam)
        return words_fms, words_fam, text

    def detectingText(self, img, flag=0):
        '''
        Функция для нахождения текстовых блоков
        '''

        text = None

        words_fms, words_fam, text = self.cleaning_words(img, flag)


        if flag == 0:
            text_fio = words_fam
            code_ind, code, fms_possible, date_fms = self.search.find_code_ind(words_fms)
            if date_fms is not None:
                self.person_data['дата выдачи'] = date_fms
            if code_ind is not None:
                self.person_data['Код подразделения'] = code

                text_fms = words_fms[0:code_ind]
                text_fms = text_fms[3:7]


            else:
                text_fms = words_fms

            text_fms = self.info_functions.clean_text(text_fms)
            text_fio = self.info_functions.clean_text(text_fio)


            all_words_fio = list(filter(None, map(self.translationText, text_fio)))
            all_words_fms = list(filter(None, map(self.translationText, text_fms)))

            self.person_data['фмс'] = self.search.cpr_spec(all_words_fms)
            # self.person_data = Search.cpr_spec_fms(self, all_words_fms, self.person_dat)
        else:

            all_words_fio = list(filter(None, map(self.translationText, text)))


        return all_words_fio

        # 4.3

    #   30/70
    # 4.5 Translation text

    def translationText(self, text):
        '''
        Перевод текста из английского в русский

        :param text:
        :return:
        '''
        if self.check_region:
            return None
        # print('old word: ', text)
        text, flag = self.doctr.poss_words1(text)


        words = self.doctr.divide_word(text, [])

        # Точка останова
        # input()
        text = self.doctr.translating(words, flag)

        if 'обл' in text.lower():
            self.check_region = True

        if text != '':
            #self.result_list.append(text)
            return text
        else:
            return None

    result_list = []



    #   0/100
    # 5 Check BD

    # 8 Compare words
    def compareWords(self, text):
        '''
        Функция сравнения слов из БД
        :param text:
        :return:
        '''
        pers_data = self.person_data.copy()
        data = self.search.cpr_spec_words(text, pers_data)
        return data


    def pack_data_to_file(self, data):
        self.filePath = self.filePath[1:]
        t = open(
            f"results/passport_results/info/file{self.filePath[self.filePath.find('orig') + len('orig'):self.filePath.find('.')]}.txt",
            'w')

        for i in data.keys():
            if data[i] == None:
                data[i] = ''
            print(i + ":", data[i])
            t.write(i + ': '+data[i])

            t.write('\n')
        t.close()


    def full_process_ocr(self, path='', key_n=3):
        '''
        Конвейер OCR

        :param path:
        :param key_n:
        :return:
        '''
        t1 = time.time()
        #self.result_list.clear()

        check_image, check_flag, img = self.info_functions.upgrade_image(self.filePath)

        res = self.detectingText(img, flag=0)
        if res == []:
            print("Плохое качество скана документа. Пересканируйте паспорт ещё раз")
            exit()

        text_fio = res
        # 6 Search BD
        print('Check BD')


        # 8 Compare words

        data1 = self.compareWords(text_fio)

        print("Finished!!!")

        # 7 Finding Серия и Номер
        self.check_region = False
        if check_flag:
            img = check_image
        rotate_img = rotate_bound(img, -90)
        res = self.detectingText(rotate_img, flag=1)
        # boxes = res[0]

        print('Check BD')

        data2 = self.search.series_number(res, data1)
        print("pers_data:", data2)
        self.pack_data_to_file(data2)

        t2 = time.time()
        delta_time = t2 - t1
        print("time used for ", delta_time)

        time.sleep(2)
        return delta_time




