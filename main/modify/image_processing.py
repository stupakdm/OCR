# import asyncio
import os
import time
import random

import cv2
import imutils
# from pyinstrument import Profiler

from Image_update.Rotating import RotateImage, show_img, New_Rotate
from Image_update.findBD.chooseBD import Search
from Image_update.findBD.correct_word import Doctr
from Image_update.red_contour import Passport1
from Image_update.updating_image import Update

# from Image_update.updating_image import Update
# from Image_update.red_contour import Passport1
from multiprocessing import Pool, cpu_count

"""from transliterate import translit, get_translit_function"""

try:
    from PIL import Image
except ImportError:
    print("No module Image")
    exit()
import pytesseract as pt


class Passport(RotateImage, New_Rotate, Update, Doctr, Passport1, Search):
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
        self.check_flag = None
        Doctr.choose_model(self)
        Search.start(self)
        self.workers = 1
        try:
            self.workers = cpu_count()
        except NotImplementedError:
            self.workers = 1
        if self.workers > 1:
            self.pool = Pool(processes=self.workers)
        self.check_image = None
        self.check_region = False

    def init_everything(self, filepath):
        self.filePath = filepath
        Doctr.init_sizes(self)
        # Search

    # Tesseract лучше работает, когда высота букв примерно 30 пикселей, dpi не особо влияем

    #   Okay
    # 0.5 Cropp
    def cropp(self, img):
        cropped = Update.corner_images(self, img)
        return cropped

    #     Check orientation
    # 0.8 Check
    def check_orientation(self, img):
        self.check_image, self.check_flag = Passport1.makeCorrectOrientation(self, img)
        if not self.check_flag:
            print('Not found')
        else:
            print('Found')
        return None

    #   50/50
    # 1 Rotate
    def rotation(self, img, key=1):
        # key = 2
        rotated = New_Rotate.rotate(self, img, key=key)
        return rotated

    #   50/50
    # 1.5 Rotate for boxes
    def rotation_boxes(self, img):
        new_img = Update.box_update(self, img)

        rotated = New_Rotate.another_rotate(self, img, new_img)
        return rotated

    #   50/50
    # 2 PreProcess
    def preProcess(self, img, key=1):
        # 2.1 DPI
        # image_orig = img.copy()
        show_img(img, "Before prePorcess")
        img = Update.change_dpi(self, img.copy(), dpi=450)
        show_img(img, 'Image after dpi')
        img = Update.change_dpi(self, img.copy(), dpi=600)
        show_img(img, 'Image after dpi')
        # 2.2 Unsharp
        img = Update.unsharp_mask(self, img.copy(), sigma=1.5, threshold=1, amount=2.0, kernel_size=(3, 3))
        # img = Update.unsharp_mask(self, img.copy(), sigma=1.5, threshold=1, amount=4.0, kernel_size=(2, 2))
        show_img(img, 'Image after unsharp_mask')
        # 2.3 CLACHE
        # img = Update.try_contrast_CLACHE(self, img.copy())
        # show_img(img, 'Image after CLACHE')

        # 2.4 Gausian
        # img  = Update.gaussian_blur(self, img.copy(), kernel=(3,3))
        # show_img(Update.sharp_image(self, img.copy()), 'Sharp_image before bright_contrast')

        # Плохо
        # show_img(Update.true_sharp(self, img.copy(), image_orig), 'true_Sharp_image before bright_contrast')

        # 2.5 Contrast   (Нужно подобрать пар-ры)
        img = Update.bright_contrast(self, img.copy(), contrast=1.1, bright=0)
        show_img(img, "Image after bright-contrast")
        # Не надо,скорее всего
        # show_img(Update.sharp_image(self, img.copy()), 'Sharp_image after bright_contrast')
        # show_img(Update.true_sharp(self, img.copy(), image_orig), 'true_Sharp_image after bright_contrast')

        show_img(img, "After prePorcess")
        # 2.6 Mask_contrast
        # if key != 1:
        #    img = Update.get_to_norm_contrast(self, img.copy())

        return img

    #   50/50
    # 3 Finding FIO
    def findingBoxes(self, img, key=1):

        resizedImage = imutils.resize(img.copy(), height=self.RESIZED_IMAGE_HEIGHT)
        boxes = None
        # 3.1 Using DNN
        if key == 1:
            boxes = Update.dnn_using(self, resizedImage.copy())

        # 3.2 Using Algorithm
        if key == 2:
            Passport1.processFullNameInternal(self, self.orig_img)
            boxes = Passport1.processFullName(self, resizedImage)
            print('boxes', boxes)

        # 3.3 Doctr
        if key == 3:
            boxes = Doctr.find_contours(self, img)
            boxes = [x for n, x in enumerate(boxes) if x not in boxes[:n]]
            # print(boxes)

        # 3.4 Tesseract
        if key == 4:
            boxes = self.ocr_core_boxes(img)

        return boxes

    #   50/50
    # 4 Detecting Text
    def detectingText(self, img, key=1, flag=0):

        # 4.1 Tesseract
        if key == 1:
            try:
                text = self.ocr_core_text(img, params=1)
                return text
            except:
                return ''

        # 4.2 Doctr
        if key == 3:
            # tex = Doctr.divide_word(self, 'TOJIIATTI', [])
            # print(tex)
            # boxes = 0
            words_fms = 0
            words_fam = 0
            text = 0
            res = Doctr.find_contours(self, img, flag)
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
                words_fms = Doctr.clean_words(self, words_fms)
                words_fms = Doctr.filter_words(self, words_fms)
                words_fam = Doctr.clean_words(self, words_fam)
                words_fam = Doctr.filter_words(self, words_fam)

            # text = ['EAE', 'MMJIMUMM', 'PAMOHA', 'MCK', 'ASI', '-CC', '24', 'OTIEJIOM', 'HEBCKOTO',
            #         'spa', 'Awenun', 'KOA', 'TPAHb', '-CEPTEM', 'BAIMMOBHY', 'JUN', '-Oroctso', 'HoA', 'Mecto', 'els',
            #         '15.07.1986', 'JIEHMHIPAL', 'MTMYK.', 'POOM', 'ATOP-']
            # all_words = []
            boxes = []
            if flag == 0:
                text_fio = words_fam
                code_ind, code, fms_possible, date_fms = Search.find_code_ind(self, words_fms)
                if date_fms is not None:
                    self.person_data['дата выдачи'] = date_fms
                if code_ind is not None:
                    self.person_data['Код подразделения'] = code

                    text_fms = words_fms[0:code_ind]
                    text_fms = text_fms[3:7]

                # self.person_data['фмс'] = fms
                # В случае некскольких вариантов на один код подразделения можно проверить ,например, первые 4 слова
                # И выбрать наиболее подходящее, чтобы не прокручивать всю строку сразу
                # text_fms = text[0:code_ind]
                # text_fms = text[0:len(text)//2]

                # text_fio = text[code_ind+3:len(text)]
                # j = 0
                else:
                    text_fms = words_fms
                    text_fio = words_fam
                j = 0
                length = len(text_fms)
                while j < length:
                    if '<' in text_fms[j] or text_fms[j] == '':
                        del text_fms[j]
                        length = len(text_fms)
                    else:
                        j += 1

                j = 0
                length = len(text_fio)
                while j < length:
                    if '<' in text_fio[j] or text_fio[j] == '':
                        del text_fio[j]
                        length = len(text_fio)
                    else:
                        j += 1

                # if we have more than one CPU or Core
                #if self.workers > 1:
                #all_words_fio = list(filter(None, self.pool.map(self.translationText, text_fio)))
                #all_words_fms = list(filter(None, self.pool.map(self.translationText, text_fms)))
                #else:
                all_words_fio = list(filter(None, map(self.translationText, text_fio)))
                all_words_fms = list(filter(None, map(self.translationText, text_fms)))

                self.person_data['фмс'] = Search.cpr_spec(self, all_words_fms)
                # self.person_data = Search.cpr_spec_fms(self, all_words_fms, self.person_dat)
            else:
                #if self.workers > 1:
                #    all_words_fio = list(filter(None, self.pool.map(self.translationText, text)))
                #else:
                all_words_fio = list(filter(None, map(self.translationText, text)))

                # all_words_fio = list(filter(None, map(self.translationText, text)))

            return [boxes, all_words_fio]

        # 4.3

    #   30/70
    # 4.5 Translation text

    def translationText(self, text):
        if self.check_region:
            return None
        # print('old word: ', text)
        text, flag = Doctr.poss_words1(self, text)

        # Точка останова
        # input()
        # t = random.randint(0, 3)
        # time.sleep(t)
        # await asyncio.sleep((random.uniform(1, 3)))
        words = Doctr.divide_word(self, text, [])

        # Точка останова
        # input()
        text = Doctr.translating(self, words, flag)

        if 'обл' in text.lower():
            self.check_region = True

        if text != '':
            self.result_list.append(text)
            return text
        else:
            return None

    result_list = []

    def collect_result(self, result):
        self.result_list.append(result)

    #   0/100
    # 5 Check BD
    def checkingBD(self, text):
        return text

    # 8 Compare words
    def compareWords(self, text):
        pers_data = self.person_data.copy()
        data = Search.cpr_spec_words(self, text, pers_data)
        return data

    def full_process_ocr(self, path='', key_n=3):
        t1 = time.time()
        orig_image = self.__read_img()
        self.result_list.clear()

        # 0.5 Cropp
        # print('Cropp')
        show_img(orig_image, 'Origin')
        img = self.cropp(orig_image.copy())
        show_img(img, 'Crop before rotate')
        # print()

        # 0.8 Check for right orientation
        # print('Check for right orientation')
        self.check_orientation(orig_image)
        # print()

        # img = orig_image.copy()
        # 1 Rotate
        # print('Rotate')
        # print('orig.shape', img.shape)

        # Закомменчено: если оно под правильным углом, то не нужно поворачивать изображение(ПРОТЕСТИРОВАТЬ для других)
        # if not self.check_flag:
        img = self.rotation(img, key=1)
        # show_img(img, 'Rotated')

        # else:
        # img = self.check_image

        # print('Cropp')
        # img = self.cropp(img.copy())
        # show_img(img, 'Crop after rotate')
        # print()

        # 2 PreProcess (Resize, DPI, Unsharp, contrast, bright, filter, and more ..)
        # print('PreProcess')
        try:
            img = self.preProcess(img)
        except:
            pass
        # print()

        # 2.5 Detecting Text on full image
        key_nn = key_n
        # print("Detect Text on full image")
        if key_nn == 3:
            res = self.detectingText(img, key=key_nn, flag=0)
            if res[0] == [] and res[1] == []:
                print("Плохое качество скана документа. Пересканируйте паспорт ещё раз")
                exit()
            # boxes = res[0]
            # text_fms = res[1]
            text_fio = res[1]
            # 6 Search BD
            print('Check BD')

            # pers_data = self.person_data.copy()

            # 8 Compare words

            data1 = self.compareWords(text_fio)

            print("Finished!!!")

            # 7 Finding Серия и Номер
            self.check_region = False
            if self.check_flag:
                img = self.check_image
            rotate_img = imutils.rotate_bound(img, -90)
            show_img(rotate_img, "rotated on -90 degree")
            res = self.detectingText(rotate_img, key=key_nn, flag=1)
            # boxes = res[0]
            text2 = res[1]

            # Используем всего 6 распознанных значений(так как всего 6 самых первых и составляют серию и номер)
            # text2 = text2[:6]
            # 6 Search BD
            print('Check BD')

            data2 = Search.series_number(self, text2, data1)
            print("pers_data:", data2)
            t = open(f"file{self.filePath[self.filePath.find('orig') + len('orig'):self.filePath.find('.')]}.txt", 'w')

            for i in data2.keys():
                if data2[i] == None:
                    data2[i] = ''
                print(i + ":", data2[i])
                t.write(i + ': ')
                if len(data2[i]) != 0:
                    if i in ['имя', 'фамилия', 'отчество', 'место']:
                        t.write('[')
                        for date in data2[i]:
                            if date[0] != None:
                                t.write(date[0])
                                t.write(', ')
                        t.write(']')
                    else:
                        if i == 'фмс':
                            t.write('[')
                            for date in data2[i]:
                                if date != None:
                                    t.write(date)
                                    t.write(', ')
                            t.write(']')
                        elif data2[i] != None:
                            t.write(data2[i])

                t.write('\n')
            t.close()
            t2 = time.time()
            delta_time = t2 - t1
            print("time used for ", delta_time)
            # file1 = open(f'text+{path}.txt', 'w')
            # file1.write(str(delta_time))
            # file1.close()
            time.sleep(5)
            return delta_time

            # text = text1+text2
        else:
            text = self.detectingText(img, key=key_nn)

        print()

        # 3 Finding FIO, date, FMS, code, снизу машинная строка (Algorithm, dnn, doctr, ...)
        print('FindingBoxes')
        if key_nn != 3:
            boxes = self.findingBoxes(img, key=key_nn)
            # img[boxes[0]]
        print()

        # 4 Preprocessing (Rotate, Resize, DPI, Unsharp, ...)
        words = []
        print('PreProccesing text boxes')

        for box in boxes:
            # 4.1 Rotate
            print('Rotate Boxes')
            box_img = img[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
            # show_img(box_img, '')
            box_img = self.rotation_boxes(box_img)
            show_img(box_img, 'Rotated_box')
            print()

            # 4.2 PreProcess
            print('PreProcces boxes')
            try:
                box_img = self.preProcess(box_img, key=2)
                show_img(box_img, 'Preprocess box')
            except:
                pass
            print()

            # 5 Detecting text (Tesseract, Doctr, dnn, ...)
            print('Detecting Text')
            try:
                text = self.detectingText(box_img, key=key_nn)
                print(text)
            except:
                pass
            print()

            # 5.5 При использовании doctr, dnn or tesseract на английском перевести(перетранслировать) слова на русский
            if key_nn > 1:
                print('Translate text')
                right_text = []
                for word in text:
                    word = self.translationText(word)
                    right_text.append(word)
                text = right_text
                print()

            # 6 Сверить значения с БД, сравнить и посчитать ошибки распознавания и детекции текстов
            print('Check BD')
            text = self.checkingBD(text)
            print(text)
            words.append(text)
            print()

        return words

    def test_quality2(self, flag=0, path=''):
        orig_image = self.__read_img()
        filepath = self.filePath + str(flag)
        print(filepath + '.txt')
        if flag == 0:
            # text = self.full_process_ocr(key_n=1)
            # print(text)
            f = open(filepath + '.txt', 'w')
            text = self.ocr_core_text(orig_image, params=1)
            print(self.divide_on_words(text))
            f.write(text)
            f.close()
        else:
            f = open(filepath + '.txt', 'w')
            cropp_image = Update.corner_images(self, orig_image.copy())
            rotated = New_Rotate.rotate(self, cropp_image, key=1)
            img = rotated
            h, w = img.shape[0:2]
            HEIGHT_CONST = 600
            WIDTH_CONST = 600
            if h < HEIGHT_CONST:
                img = Update.resize_img(self, img, HEIGHT_CONST / h, HEIGHT_CONST / h, interpolation=cv2.INTER_AREA)
            if w < WIDTH_CONST:
                img = Update.resize_img(self, img, WIDTH_CONST / w, WIDTH_CONST / w, interpolation=cv2.INTER_AREA)
            rotated_img = img.copy()
            Doctr.find_contours(self, rotated_img)
            rotated_img = Update.try_contrast_CLACHE(self, rotated_img)
            # show_img(Update.try_contrast_CLACHE(self, rotated_img), 'CLACHE')
            # Update.corner_images(self, rotated_img.copy())
            # norm_contr = Update.get_to_norm_contrast(self, rotated_img)

            # gausian = Update.gaussian_blur(self, norm_contr)
            # show_img(gausian, 'Gausian')

            # cv2.imwrite('save_1.jpg', rotated_img)
            # os.system('mogrify -set density 400 save_1.jpg')
            # Doctr.find_contours(self, 'save_1.jpg')
            # os.system('rm save_1.jpg')
            # Update.dnn_using(self, orig_image)
            # Update.add_alpha_channel(self, rotated_img)
            # img = Update.change_hsv(self, rotated_img, hue = -30, satur = 0, value = 0)
            # Update.using_mask(self, rotated_img)

            boxes = Update.dnn_using(self, rotated_img)
            for box in boxes:
                origImageCut = rotated_img[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
                show_img(origImageCut, 'Cuts')
                unsharp = Update.unsharp_mask(self, origImageCut)
                gaus = Update.gaussian_blur(self, unsharp, (5, 5))
                img = gaus
                # img = Update.get_to_norm_contrast(self, gaus)

                show_img(img, 'Contrast')

                # heightKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 5))
                # dilate = cv2.dilate(img, heightKernel)

                # dilate = Update.dilation(self, dilate)
                # widthKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 2))
                # erose = cv2.erode(img, widthKernel, borderType=cv2.BORDER_REFLECT)
                widthKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 2))
                img = cv2.dilate(img, widthKernel, borderType=cv2.BORDER_REFLECT)
                heightKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 5))
                dilate = cv2.dilate(img, heightKernel)
                # show_img(dilate, 'dilate+erose')

                rotated_cut = New_Rotate.another_rotate(self, origImageCut, dilate)
                show_img(rotated_cut, 'Rotated_cut')
                img = rotated_cut
                cv2.imwrite('save_1.jpg', img)
                os.system('mogrify -set density 300 save_1.jpg')
                img = cv2.imread('save_1.jpg')
                os.system('rm save_1.jpg')
                # show_img(Update.erosion(self, Update.black_white(self, img), (10,2), board='CONSTANT'), 'black_white')
                show_img(img, 'Changes')

                text = self.ocr_core_text(gaus)
                print(self.divide_on_words(text))
            # show_img(rotated_img, 'Creating mask')
            # img = Update.change_hsv(self, rotated_img, satur = 256)
            # show_img(img, 'After saturation')

            # img = Update.sobel_operator(self, img)
            # show_img(img, 'Sobel')
            # img = Update.unsharp_mask(self, img)

            # img = Update.sharp_image(self, img, t = 12, n = 2)
            # show_img(img, 'Sharp')
            # img = Update.aprox_poly(self, img, rotated_img)
            # Update.blob_detector(self, img)
            # img = Update.change_hsv(self, img, value = 30)
            # show_img(img, 'HSV')
            # resized = Update.resize_img(self, rotated, 1.0, 1.0, interpolation=cv2.INTER_AREA)
            # unsharp = Update.unsharp_mask(self, rotated)
            # gray = Update.black_white(self, unsharp)
            # img = Update.getContours(self, gray, unsharp)
            # show_img(img, 'Check')
            # subtract = Update.subtract(self, img)
            text = self.ocr_core_text(img)
            print(self.divide_on_words(text))
            # img = Update.sobel_operator(self, orig_image)
            # img = Update.draw_contours(self, img, orig_image)
            # show_img(img, 'Afterfilling contours')

            # bl_wh = Update.black_white(self, orig_image)
            # cols, rows = orig_image.shape[0:2]
            # img = Update.getEdges(self, orig_image)
            # img = Update.dilation(self, img, k=(2,2))
            # img = Update.getContours(self, img, orig_image)
            # show_img(img, 'After contours')
            # sub = Update.subtract(self, img)
            # img = Update.unsharp_mask(self, sub)
            # img = New_Rotate.rotate(self, img)
            # img = Update.resize_img(self, img, int(cols*1.5), int(rows*1.5), interpolation=cv2.INTER_AREA)
            # show_img(sub, 'subtract')
            # text = self.ocr_core(img)
            # print(self.divide_on_words(text))
            '''hsv = (-30, 0 , 30)
            cols, rows = orig_image.shape[0:2]
            img = orig_image.copy()
            Update.getEdges(self, img)
            #img = New_Rotate.rotate(self, img)
            img = Update.resize_img(self, img, int(cols*1.5), int(rows*1.5), interpolation=cv2.INTER_AREA)
            img = Update.change_hsv(self, img, hue=hsv[0], value=hsv[0], satur=hsv[0])
            img = Update.unsharp_mask(self, img)
            #img = Update.bright_contrast(self, img, contrast=2.0)
            show_img(img, 'after_changing')
            text = self.ocr_core(img)
            print(self.divide_on_words(text))
            f.write(text)
            f.close()'''
        return None

    def __read_img(self):
        self.orig_img = cv2.imread(self.filePath)
        print('shape orig', self.orig_img.shape)
        # print(self.orig_img.shape)
        # show_img(self.orig_img, 'Original')
        # print('shape orig', self.orig_img.shape)
        # upd = Update.resize_img(self, self.orig_img, cols = self.orig_img.shape[0]*2, rows = self.orig_img.shape[1]*2)
        # print('shape resize', upd.shape)
        '''
        print("colors")
        for i in Update.colors:
            show_img(Update.change_color(self, upd, choice=i), i)

        print("erose")
        for i in Update.boards:
            erose = Update.erosion(self, upd, k = (5, 5), board=i)
            show_img(erose, i)
            print(f"Erose {i}\n", Passport.ocr_core(self, erose))

        print("dilation")
        for i in Update.boards:
            dilate = Update.dilation(self, upd, k = (5, 5), board=i)
            show_img(dilate, i)
            print(f"Dilate {i}\n", Passport.ocr_core(self, dilate))

        print("Morphology")
        for i in Update.ex:
            morph = Update.morphology(self, upd, k = (5, 5), choice=i)
            show_img(morph, i)
            print(f"Morph {i}\n", Passport.ocr_core(self, morph))
            '''
        # show_img(upd, 'Resize')
        # show_img(Update.black_white(self, self.orig_img), 'black')
        # hsv = (-30, 0, 30)
        # change_contours = Update.change_hsv(self, upd, hue=-30, value=-30, satur=-30)
        # show_img(change_contours, 'HSV')
        # self.chng = Update.extract_contours_text(self, change_contours)
        # show_img(self.chng, 'New_image')
        # cv2.imwrite('templates/new.jpg', self.chng)
        '''for h in hsv:
            for s in hsv:
                for v in hsv:
                    show_img(Update.change_hsv(self, upd, hue = h, value = v, satur=s), f'hue={h}, satur={s}, value={v}')
        '''
        return self.orig_img

    def ocr_core_text(self, img, params=1):
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

    def divide_on_words(self, text):
        text = text.split('\n')
        return list(map(str.strip, text))
