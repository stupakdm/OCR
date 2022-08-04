import asyncio
import os
import time
import random

import cv2
import imutils
from pyinstrument import Profiler

from Image_update.Rotating import RotateImage, show_img, New_Rotate
from Image_update.findBD.chooseBD import Search
from Image_update.findBD.correct_word import Doctr
from Image_update.red_contour import Passport1
from Image_update.updating_image import Update

# from Image_update.updating_image import Update
# from Image_update.red_contour import Passport1
from multiprocessing.dummy import Pool as ThreadPool
"""from transliterate import translit, get_translit_function"""

try:
    from PIL import Image
except ImportError:
    print("No module Image")
    exit()
    #import Image
import pytesseract as pt


class Passport(RotateImage, New_Rotate,  Update, Doctr, Passport1, Search):
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

    def __init__(self, filepath):
        self.filePath = filepath
        Doctr.choose_model(self)
        Search.start(self)
        self.check_image = None
        self.check_region = False
        # Doctr.add_files(self, self.filePath)
        # Doctr.find_contours(self)


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
        #key = 2
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
    def preProcess(self, img, key = 1):
        # 2.1 DPI
        image_orig = img.copy()
        show_img(img, "Before prePorcess")
        img = Update.change_dpi(self, img.copy(), dpi = 450)
        show_img(img, 'Image after dpi')
        img = Update.change_dpi(self, img.copy(), dpi = 600)
        show_img(img, 'Image after dpi')
        # 2.2 Unsharp
        img = Update.unsharp_mask(self, img.copy(), sigma=1.5, threshold=1, amount=2.0, kernel_size=(3,3))
        #img = Update.unsharp_mask(self, img.copy(), sigma=1.5, threshold=1, amount=4.0, kernel_size=(2, 2))
        show_img(img, 'Image after unsharp_mask')
        # 2.3 CLACHE
        #img = Update.try_contrast_CLACHE(self, img.copy())
        #show_img(img, 'Image after CLACHE')

        # 2.4 Gausian
        #img  = Update.gaussian_blur(self, img.copy(), kernel=(3,3))
        #show_img(Update.sharp_image(self, img.copy()), 'Sharp_image before bright_contrast')

        # Плохо
        #show_img(Update.true_sharp(self, img.copy(), image_orig), 'true_Sharp_image before bright_contrast')

        # 2.5 Contrast   (Нужно подобрать пар-ры)
        img = Update.bright_contrast(self, img.copy(), contrast=1.1, bright = 0)
        show_img(img, "Image after bright-contrast")
        # Не надо,скорее всего
        #show_img(Update.sharp_image(self, img.copy()), 'Sharp_image after bright_contrast')
        #show_img(Update.true_sharp(self, img.copy(), image_orig), 'true_Sharp_image after bright_contrast')


        show_img(img, "After prePorcess")
        # 2.6 Mask_contrast
        #if key != 1:
        #    img = Update.get_to_norm_contrast(self, img.copy())

        return img

    #   50/50
    # 3 Finding FIO
    def findingBoxes(self, img, key = 1):

        RESIZED_IMAGE_HEIGHT = 600
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
            boxes = [x for n,x in enumerate(boxes) if x not in boxes[:n]]
            #print(boxes)

        # 3.4 Tesseract
        if key == 4:
            boxes = self.ocr_core_boxes(img)

        return boxes

    #   50/50
    # 4 Detecting Text
    def detectingText(self, img, key = 1, flag = 0):

        # 4.1 Tesseract
        if key == 1:
            try:
                text = self.ocr_core_text(img, params=1)
                return text
            except:
                return ''

        # 4.2 Doctr
        if key == 3:
            #tex = Doctr.divide_word(self, 'TOJIIATTI', [])
            #print(tex)
            res = Doctr.find_contours(self, img)
            boxes = res[0]
            text = res[1]
            print("text: ", text)
            all_words = []
            if flag == 1:
                l = len(text)
                i = 0
                while i < l:
                    for j in text[i]:
                        if j.isalpha():
                            del text[i]
                            l = len(text)
                            break
                    else:
                        i += 1

            else:
                j = 0
                l = len(text)
                while j < l:
                    if len(text[j]) != 0:
                        if len(text[j]) < 3 and not text[j][0].isdigit():
                            if j+1 != len(text):
                                text[j] += text[j + 1]
                                del text[j + 1]
                                l = len(text)
                            else:
                                j+=1
                        else:
                            j += 1
                    else:
                        del text[j]
                        l = len(text)
            # print("After adding: ", text)
            if flag == 0:
                #print("text:", text)
                text = Doctr.filter_words(self, text)
            # print("After Doctr filter:", text)
            #text = text[10:]
            #text = ['EAE', 'MMJIMUMM', 'PAMOHA', 'MCK', 'ASI', '-CC', '24', 'OTIEJIOM', 'HEBCKOTO',
            #         'llactopy', 'BHA', 'CAHKT', 'TETEPBYPTA', '782-024', '04.10.2006', 'Koa', 'noApaaseNeNnts', 'Aara',
            #         'spa', 'Awenun', 'KOA', 'TPAHb', '-CEPTEM', 'BAIMMOBHY', 'JUN', '-Oroctso', 'HoA', 'Mecto', 'els',
            #         '15.07.1986', 'JIEHMHIPAL', 'MTMYK.', 'POOM', 'ATOP-']
            all_words = []
            boxes = []
            if flag == 0:
                code_ind, code, fms_possible, date_fms = Search.find_code_ind(self, text)
                if date_fms != None:
                    self.person_data['дата выдачи'] = date_fms
                if code_ind != None:
                    self.person_data['Код подразделения'] = code

                    text_fms = text[0:code_ind]
                    text_fms = text_fms[3:7]

                #self.person_data['фмс'] = fms
                #В случае некскольких вариантов на один код подразделения можно проверить ,например, первые 4 слова
                #И выбрать наиболее подходящее, чтобы не прокручивать всю строку сразу
                #text_fms = text[0:code_ind]
                #text_fms = text[0:len(text)//2]

                    text_fio = text[code_ind+3:len(text)]
                    j = 0
                else:
                    text_fms = text[0:len(text)//2]
                    text_fio = text[(len(text)//2)+1:]
                j = 0
                l = len(text_fms)
                while j<l:
                    if '<' in text_fms[j] or text_fms[j] == '':
                        del text_fms[j]
                        l = len(text_fms)
                    else:
                        j+=1

                j = 0
                l = len(text_fio)
                while j < l:
                    if '<' in text_fio[j] or text_fio[j] == '':
                        del text_fio[j]
                        l = len(text_fio)
                    else:
                        j += 1

                # print('2 text: ', text)
                #print("text_fms: ", text_fms)
                # print("pers_data: ", self.person_data)
                # print("text_fio: ", text_fio)
                #threads= []
                #manager = multiprocessing.Manager()
                #self.result_list = manager.list()

                #all_words_fio = []
                #all_words_fms = []
                #pool_size = 1

                #proc = [Process(target=self.translationText, args=(word,)) for word in text_fio]
                #pool = ThreadPool(pool_size)
                #pool.map(self.proc, proc)
                #pool.close()
                #pool.join()

                #for word in text_fio:
                #    w = Process(target=self.translationText, args=(word,))
                #    w.start()
                #    w.join()
                    #all_words_fio.append(w)

                #p = Process(target=self.translationText, args=text)

                # Parallel with ThreadPool
                """pool_size = 5
                with ThreadPool(pool_size) as p:
                    all_words_fio = p.map(self.translationText, text_fio)"""
                #pool.map_async(self.translationText, text_fio, callback=self.collect_result)
                #pool.close()
                #pool.join()

                # Parallel with asyncio
                """p = Profiler(async_mode='disabled')
                with p:
                    loop = asyncio.get_event_loop()

                    tasks = [self.translationText(word) for word in text_fio]
                    main_task = asyncio.gather(*tasks)

                    all_words_fio = loop.run_until_complete(main_task)
                    #loop.close()
                p.print()"""

                # Single with map
                # print('2 text_fio: ', text_fio)
                all_words_fio = list(filter(None, map(self.translationText, text_fio)))

                # print('2 all_words_fio:', all_words_fio)
                """p = Profiler(async_mode='disabled')
                
                with p:
                    all_words_fio = list(map(self.translationText, text_fio))
                p.print()"""
                #print("all_words_fio:", all_words_fio)
                #all_words_fio = self.result_list.copy()
                #self.result_list.clear()
                #all_words_fio = [pool.apply_async(self.translationText, x) for x in text_fio]
                #pool = Pool(processes=pool_size)
                #results_fio = pool.map_async(self.translationText, text_fio)
                #all_words_fio += results_fio.get()
                #pool.close()
                #pool.join()

                #pool = Pool(processes=pool_size)
                #results_fms = pool.map_async(self.translationText, text_fms)
                #all_words_fms += results_fms.get()
                #pool.close()
                #pool.join()
                #all_words_fio = list(map(self.translationText, text_fio))
                #loop = asyncio.get_event_loop()

                """asks = [self.translationText(word) for word in text_fms]
                main_task = asyncio.gather(*tasks)"""

                #all_words_fms = loop.run_until_complete(main_task)
                #loop.close()

                #p = Profiler(async_mode='disabled')
                #with p:
                #all_words_fms = list(map(self.translationText, text_fms))
                #p.print()
                """p = Profiler(async_mode='disabled')
                with p:
                    loop = asyncio.get_event_loop()

                    tasks = [self.translationText(word) for word in text_fms]
                    main_task = asyncio.gather(*tasks)

                    all_words_fms = loop.run_until_complete(main_task)
                    # loop.close()
                p.print()"""

                # Parallel with ThreadPool
                """pool_size = 5
                with ThreadPool(pool_size) as p:
                    all_words_fms = p.map(self.translationText, text_fms)"""

                # Single with map
                all_words_fms = list(filter(None, map(self.translationText, text_fms)))
                # print('all_words_fms:', all_words_fms)
                # print(f"all_words_fms: {all_words_fms}")
                #self.person_data['фмс'] = pool.map_async(Search.cpr_spec,)
                self.person_data['фмс'] = Search.cpr_spec(self, all_words_fms)
                #self.person_data = Search.cpr_spec_fms(self, all_words_fms, self.person_dat)
            else:
                """loop = asyncio.get_event_loop()

                tasks = [self.translationText(word) for word in text]
                main_task = asyncio.gather(*tasks)

                all_words_fio = loop.run_until_complete(main_task)
                loop.close()"""
                #pool_size = 5
                #pool = Pool(processes=pool_size)
                #results_fio = pool.map_async(self.translationText, text)
                #all_words_fio = results_fio.get()
                #pool.close()
                #pool.join()
                """p = Profiler(async_mode='disabled')
                with p:
                    loop = asyncio.get_event_loop()

                    tasks = [self.translationText(word) for word in text]
                    main_task = asyncio.gather(*tasks)

                    all_words_fio = loop.run_until_complete(main_task)
                    # loop.close()
                p.print()"""

                # Parallel with ThreadPool
                """ pool_size = 5
                with ThreadPool(pool_size) as p:
                    try:
                        all_words_fio = p.map(self.translationText, text)
                    except:
                        exit(0)"""

                # Single with map
                all_words_fio = list(filter(None, map(self.translationText, text)))

            #pool_size = 5
            #pool = Pool(processes=pool_size)
            #results = pool.map_async(self.translationText, text)
            #pool.close()
            #pool.join()
            #all_words_fms = list(map(self.translationText, text_fms))
            #all_words_fio = list(map(self.translationText, text_fio))
            #all_words = all_words_fio+all_words_fms
            #all_words = results.get()
            #all_words = list(map(self.translationText, text))
            #for word in text:
            #    pool.apply_async(self.translationText, word)
            #    all_words.append(self.translationText(word))

            #print(all_words)
            return [boxes, all_words_fio]

        # 4.3

    #   30/70
    # 4.5 Translation text

    def translationText(self, text):
        if self.check_region:
            return None
        # print('old word: ', text)
        text, flag = Doctr.poss_words1(self, text)
        # print('text: ', len(text))
        # print('new word: ',text, flag)
        # Точка останова
        # input()
        #t = random.randint(0, 3)
        #time.sleep(t)
        #await asyncio.sleep((random.uniform(1, 3)))
        #print('text after poss_words1: ', text)
        words = Doctr.divide_word(self, text, [])

        #print('words after divide_word: ', words)
        # print('words: ', len(words))
        # Точка останова
        # input()
        text = Doctr.translating(self, words, flag)

        print('text after translating: ', text)
        if 'обл' in text.lower():
            self.check_region = True

        if text != '':
            print(text)
            self.result_list.append(text)
        # print('result_list: ', self.result_list)
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
        orig_image = self.__read_img()
        self.result_list.clear()

        # 0.5 Cropp
        print('Cropp')
        show_img(orig_image, 'Origin')
        img = self.cropp(orig_image.copy())
        # show_img(img, 'Crop before rotate')
        print()

        # 0.8 Check for right orientation
        print('Check for right orientation')
        self.check_orientation(orig_image)
        print()

        #img = orig_image.copy()
        # 1 Rotate
        print('Rotate')
        print('orig.shape', img.shape)

        # Закомменчено: если оно под правильным углом, то не нужно поворачивать изображение(ПРОТЕСТИРОВАТЬ для других)
        #if not self.check_flag:
        img = self.rotation(img, key = 1)
        show_img(img, 'Rotated')
        print('img.shape', img.shape)
        print()
        #else:
        #img = self.check_image

        print('Cropp')
        #img = self.cropp(img.copy())
        show_img(img, 'Crop after rotate')
        print()

        # 2 PreProcess (Resize, DPI, Unsharp, contrast, bright, filter, and more ..)
        print('PreProcess')
        try:
            img = self.preProcess(img)
        except:
            pass
        print()

        # 2.5 Detecting Text on full image
        key_nn = key_n
        print("Detect Text on full image")
        if key_nn == 3:
            res = self.detectingText(img, key=key_nn, flag = 0)
            boxes = res[0]
            #text_fms = res[1]
            text_fio = res[1]
            #print('final text: ', text1)
            #input("Enter: ")
            # 6 Search BD
            print('Check BD')
            #final_text = ['ЯДЕ', 'АЕР', 'АИИ', 'МИЛИЦИИ', 'РАМОНА', 'ИМC', 'КЛЯЛ', 'РОCC', '24', 'ОТДЕЛОМ', 'НЕБCКОГО', 'ббастоот', 'спАаи', 'CАДКП-ГПБЕГБРЕБРРГ', '782-024', '04.10.2006', 'Кох', 'подразделеия', 'Иза', 'спИ', 'Лпенни', 'КОЛ', 'Чар', '0', '6', '', 'ГРЯНь', 'СЕРГЕИ', 'ВЛЯИМОВНЧ', 'ЯЛУ', 'ОМест', 'Поз', 'Место', 'РОДИРИИК', '15.07.1986', 'Иуд.', 'МПс', 'рорспесия', 'Тор.', 'ЛЕНИНГРАД']
            #final_text =  ['Еaе', 'Миялинии', 'Рдионд', 'Мcк', 'Ляд', '', '24', 'Отпеялом', 'Небcкого', 'Ббастору', 'Внд',
            #       'Cдикт', 'Петербургя', '782-024', '04.10.2006', 'Кра', 'Подраассйсиитя', 'Дата', 'Яра', 'Лисинй',
            #       'Коя', 'Гряно', '-cергеи', 'Влдимовнч', 'Яни', '-отостсо', 'Ирд', 'ИМссто', 'Сбя', '15.07.1986',
            #       'Ленинирдц', 'Мпичк.', 'Роои', 'Дпор-']
            pers_data = self.person_data.copy()
            #final_text1 = final_text[len(final_text)//2:len(final_text)]
            #pers_data = Search.cpr_spec_fms(self, text_fms, pers_data)
            print("pers_data: ", pers_data)
            # 8 Compare words
            #pool_size = 5
            #pool = Pool(processes=pool_size)
            #results = pool.map_async(self.compareWords, text_fio)
            #data1 = results.get()
            #pool.close()
            #pool.join()

            data1 = self.compareWords(text_fio)
            #data1 = Search.cpr_spec_words(self, text_fio, pers_data)
            print("pers_data: ", pers_data)
            #print("data_fms: ", data_fms)
            print("data: ", data1)
            #print(data1)
            #time.sleep(15)
            print("Finished!!!")
            #input()

            #7 Finding Серия и Номер
            self.check_region = False
            if self.check_flag:
                img = self.check_image
            rotate_img = imutils.rotate_bound(img, -90)
            show_img(rotate_img, "rotated on -90 degree")
            res = self.detectingText(rotate_img, key=key_nn, flag=1)
            boxes = res[0]
            text2 = res[1]

            print('final text: ', text2)
            # 6 Search BD
            print('Check BD')

            data2 = Search.series_number(self, text2, data1)
            print(data2)
            t = open(f"file{random.randint(1,100)}.txt", 'w')
            for i in data2.keys():
                if data2[i] == None:
                    data2[i] = ''
                t.write(data2[i])
                t.write('\n')
            time.sleep(5)
            print()

            #text = text1+text2
        else:
            text = self.detectingText(img, key=key_nn)

        #print(text)
        print()

        # 3 Finding FIO, date, FMS, code, снизу машинная строка (Algorithm, dnn, doctr, ...)
        print('FindingBoxes')
        if key_nn != 3:
            boxes = self.findingBoxes(img, key=key_nn)
            #img[boxes[0]]
        print()

        # 4 Preprocessing (Rotate, Resize, DPI, Unsharp, ...)
        words = []
        print('PreProccesing text boxes')

        for box in boxes:
            # 4.1 Rotate
            print('Rotate Boxes')
            box_img = img[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
            #show_img(box_img, '')
            box_img  = self.rotation_boxes(box_img)
            show_img(box_img, 'Rotated_box')
            print()

            # 4.2 PreProcess
            print('PreProcces boxes')
            try:
                box_img = self.preProcess(box_img, key = 2)
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

    def test_quality2(self, flag = 0, path=''):
        orig_image = self.__read_img()
        filepath = self.filePath+str(flag)
        print(filepath+'.txt')
        if flag == 0:
            #text = self.full_process_ocr(key_n=1)
            #print(text)
            f = open(filepath + '.txt', 'w')
            text = self.ocr_core_text(orig_image, params = 1)
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
            if h<HEIGHT_CONST:
                img = Update.resize_img(self, img, HEIGHT_CONST/h, HEIGHT_CONST/h, interpolation=cv2.INTER_AREA)
            if w<WIDTH_CONST:
                img = Update.resize_img(self, img, WIDTH_CONST / w, WIDTH_CONST / w, interpolation=cv2.INTER_AREA)
            rotated_img = img.copy()
            Doctr.find_contours(self, rotated_img)
            rotated_img = Update.try_contrast_CLACHE(self, rotated_img)
            #show_img(Update.try_contrast_CLACHE(self, rotated_img), 'CLACHE')
            #Update.corner_images(self, rotated_img.copy())
            #norm_contr = Update.get_to_norm_contrast(self, rotated_img)

            #gausian = Update.gaussian_blur(self, norm_contr)
            #show_img(gausian, 'Gausian')

            #cv2.imwrite('save_1.jpg', rotated_img)
            #os.system('mogrify -set density 400 save_1.jpg')
            #Doctr.find_contours(self, 'save_1.jpg')
            #os.system('rm save_1.jpg')
            #Update.dnn_using(self, orig_image)
            #Update.add_alpha_channel(self, rotated_img)
            #img = Update.change_hsv(self, rotated_img, hue = -30, satur = 0, value = 0)
            #Update.using_mask(self, rotated_img)

            boxes = Update.dnn_using(self, rotated_img)
            for box in boxes:
                origImageCut = rotated_img[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
                show_img(origImageCut, 'Cuts')
                unsharp = Update.unsharp_mask(self, origImageCut)
                gaus = Update.gaussian_blur(self, unsharp, (5,5))
                img = gaus
                #img = Update.get_to_norm_contrast(self, gaus)

                show_img(img, 'Contrast')

                #heightKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 5))
                #dilate = cv2.dilate(img, heightKernel)

                #dilate = Update.dilation(self, dilate)
                #widthKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 2))
                #erose = cv2.erode(img, widthKernel, borderType=cv2.BORDER_REFLECT)
                widthKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 2))
                img = cv2.dilate(img, widthKernel, borderType=cv2.BORDER_REFLECT)
                heightKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 5))
                dilate = cv2.dilate(img, heightKernel)
                #show_img(dilate, 'dilate+erose')

                rotated_cut = New_Rotate.another_rotate(self, origImageCut, dilate)
                show_img(rotated_cut, 'Rotated_cut')
                img = rotated_cut
                cv2.imwrite('save_1.jpg', img)
                os.system('mogrify -set density 300 save_1.jpg')
                img = cv2.imread('save_1.jpg')
                os.system('rm save_1.jpg')
                #show_img(Update.erosion(self, Update.black_white(self, img), (10,2), board='CONSTANT'), 'black_white')
                show_img(img, 'Changes')

                text = self.ocr_core_text(gaus)
                print(self.divide_on_words(text))
            #show_img(rotated_img, 'Creating mask')
            #img = Update.change_hsv(self, rotated_img, satur = 256)
            #show_img(img, 'After saturation')

            #img = Update.sobel_operator(self, img)
            #show_img(img, 'Sobel')
            #img = Update.unsharp_mask(self, img)

            #img = Update.sharp_image(self, img, t = 12, n = 2)
            #show_img(img, 'Sharp')
            #img = Update.aprox_poly(self, img, rotated_img)
            #Update.blob_detector(self, img)
            #img = Update.change_hsv(self, img, value = 30)
            #show_img(img, 'HSV')
            #resized = Update.resize_img(self, rotated, 1.0, 1.0, interpolation=cv2.INTER_AREA)
            #unsharp = Update.unsharp_mask(self, rotated)
            #gray = Update.black_white(self, unsharp)
            #img = Update.getContours(self, gray, unsharp)
            #show_img(img, 'Check')
            #subtract = Update.subtract(self, img)
            text = self.ocr_core_text(img)
            print(self.divide_on_words(text))
            #img = Update.sobel_operator(self, orig_image)
            #img = Update.draw_contours(self, img, orig_image)
            #show_img(img, 'Afterfilling contours')

            #bl_wh = Update.black_white(self, orig_image)
            #cols, rows = orig_image.shape[0:2]
            #img = Update.getEdges(self, orig_image)
            #img = Update.dilation(self, img, k=(2,2))
            #img = Update.getContours(self, img, orig_image)
            #show_img(img, 'After contours')
            #sub = Update.subtract(self, img)
            #img = Update.unsharp_mask(self, sub)
            #img = New_Rotate.rotate(self, img)
            #img = Update.resize_img(self, img, int(cols*1.5), int(rows*1.5), interpolation=cv2.INTER_AREA)
            #show_img(sub, 'subtract')
            #text = self.ocr_core(img)
            #print(self.divide_on_words(text))
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
        #print('shape orig', self.orig_img.shape)
        #upd = Update.resize_img(self, self.orig_img, cols = self.orig_img.shape[0]*2, rows = self.orig_img.shape[1]*2)
        #print('shape resize', upd.shape)
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
        #show_img(Update.black_white(self, self.orig_img), 'black')
        #hsv = (-30, 0, 30)
        #change_contours = Update.change_hsv(self, upd, hue=-30, value=-30, satur=-30)
        #show_img(change_contours, 'HSV')
        #self.chng = Update.extract_contours_text(self, change_contours)
        #show_img(self.chng, 'New_image')
        #cv2.imwrite('templates/new.jpg', self.chng)
        '''for h in hsv:
            for s in hsv:
                for v in hsv:
                    show_img(Update.change_hsv(self, upd, hue = h, value = v, satur=s), f'hue={h}, satur={s}, value={v}')
        '''
        return self.orig_img

    def ocr_core_text(self, img, params=1):
        """This function will handle the core OCR"""
        #rus = 'абвгдеёжзийклмнопрстуфхцчшщЪыьэюя'
        if params == 1:
            text = pt.pytesseract.image_to_string(img, lang='rus')
        else:
            text = pt.pytesseract.image_to_string(img, config='-l rus --psm 5 --oem 1 --dpi 300')
        #text = pt.pytesseract.image_to_string(img, lang='rus', config = f'--psm 12 --oem 3 -c tessedit_char_whitelist={rus}{rus.upper()}ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
        return text.strip().lower().capitalize()

    def ocr_core_boxes(self, img):
        boxes = pt.pytesseract.image_to_boxes(img)
        return boxes

    def divide_on_words(self, text):
        text = text.split('\n')
        return list(map(str.strip, text))