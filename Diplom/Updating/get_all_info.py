
from Optimize.updating import Update, show_img
from Optimize.Rotate_diplom import Rotate
from Optimize.SearchBD.correct_word import *
from Optimize.SearchBD.choose_right_BD import FindWord

from get_all_module import GetFunctions

try:
    from PIL import Image
except ImportError:
    print("No module Image")
    exit()
    #import Image
#import pytesseract as pt


class Diplom(GetFunctions):

    #0
    #begin
    def __init__(self):
        '''
        Конструктор
        Запускается в самом начале
        Здесь передаются все изначальные параметры, которые впоследствии не изменяются
        '''
        self.upd = Update()
        self.rot = Rotate()
        self.doctr_model = Doctr()
        self.tes_model = Tesseract()
        self.rightBD = FindWord()
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

    #0.5
    #Preparing
    def initialize(self, path):
        '''
        Запускается при каждой новой фотографии
        Обнуляет параметры снизу
        path: путь по которому хранится фото
        '''
        self.city_possible = None
        self.orig_image = cv2.imread(path)
        self.result = dict()

    # 1
    def __modify_image(self, image, flag = 1):
        '''
        Улучшение качества изображения
        image: 3 мерное изображение
        flag: принимает значения 0 и 1
        :return: Возвращает улучшенное изображение
        '''
        try:
            if flag == 1:

                image = self.rot.rotate(self.upd.resize_image(image), image)
        except:
            image = image



        image = self.upd.corner_images(image)

        image = self.upd.unsharp_mask(image, sigma=1.5, threshold=1, amount=2.0, kernel_size=(3,3))
        if flag == 2:
            image = self.upd.black_white(image)

        return image

    # 2
    def __segmentation(self, image):
        '''
        Разбиение картинки по текстовым блокам
        image: 3 мерное изображение
        :return: left_block, right_block - два массива, размерностью 2
        Каждый массив хранит 2 массива: первый - координаты текстовых блоков, второй - текст в этом блоке
        '''
        H, W = image.shape[:2]
        self.doctr_model.initialize(H, W)
        left_block, right_block = self.doctr_model.detect_image(image)
        #print('words: ', words)
        #print('blocks: ', blocks)
        #print('length: ', len(blocks), len(words))
        return left_block, right_block



    # 3
    def __symbol_blocks(self, left_block, right_block, img):
        '''
        Распознавание текстовой информмации на текстовых блоках
        left_block, right_block - два массива, размерностью 2
        Каждый массив хранит 2 массива: первый - координаты текстовых блоков, второй - текст в этом блоке
        img: 3 мерное изображение
        :return: массив, размерностью 2, первый - текст на левой стороне, второй - текст на правой стороне
        '''
        info = []

        #specs = ['бакалавр', 'магистр', 'специалитет']
        #max_block = self.doctr_model.max_block_left

        #plt.imshow(img[max_block[1]:max_block[3], max_block[0]:max_block[2]])
        #plt.show()

        #avg_w = avg_w/len(blocks)
        #avg_h = avg_h/len(blocks)
        count_digit_left, left_count_strings = 0, 0
        min_gap_x, min_gap_y = 0, 0
        flag = [0, 0]
        left_max_blocks = [[[0,0,0,0],'',''] for i in range(3)]
        #right_blocks = []
        #new_words = []
        prev_block, blocks_len = 0, 0
        #min_gap_y = 0
        #blocks_len = 0

        specs = ['бакалавр', 'магистр', 'специалитет']
        best_word = specs[0]
        rate_best = 0
        y_prev= 0
        prev_string = ['']
        current_string = []
        #left_count_strings = 0
        prev_y = 0
        #save_prev_word_ind = None
        all_blocks = (left_block, right_block)
        #self.city_possible = [words[0], words[0]]
        for i in range(len(all_blocks)):
            info1 = []
            parts = []
            blocks = all_blocks[i][0].copy()
            words = all_blocks[i][1].copy()
            if i == 0:
                self.city_possible = [words[0], words[0]]
            for ind, block in enumerate(blocks):
                #if (i == 0 and block[2] < self.doctr_model.mid_x) or (i == 1 and block[2] > self.doctr_model.mid_x):
                    #print('block: ', block)

                    block_img = img[block[1]:block[3], block[0]:block[2]]
                    if i == 1:
                        if flag == 0:
                            if self.rightBD.check_digit(words[ind], i) and ind>10 and len(words[ind]) > 5:
                                parts += left_max_blocks
                                info1.append(parts.copy())
                                parts.clear()
                                index_word = self.check_for_letter(words[ind])
                                word = self.text_recognition(block_img, self.tes_model)
                                if index_word == -1:
                                    parts.append([word, words[ind]])
                                else:
                                    parts.append([word, words[ind][0:index_word]])
                                info1.append(parts.copy())

                                parts.clear()
                                if index_word != -1:
                                    parts.append([words[ind][index_word+1:], word])
                                flag = 1
                                prev_block = block
                                #dist = blocks[ind + 1][3] - block[1]
                                #min_gap = (blocks[ind + 1][2] - block[0]) / 6
                                min_gap_x = (block[2] - block[0]) / 6
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
                                    word = self.text_recognition(block_img, self.tes_model)
                                    another_word = words[ind]
                                    left_max_blocks[j][1] = word
                                    left_max_blocks[j][2] = another_word
                                    break
                            continue
                        elif flag == 1:
                            if abs(prev_block[3] - block[3]) <= min_gap_y:
                                word = self.text_recognition(block_img, self.tes_model)
                                parts.append([words[ind], word])
                                prev_block = block
                            else:
                                info1.append(parts.copy())
                                parts.clear()
                                flag += 1
                                prev_block = block
                                y_prev = block[1]
                                #blocks_len += 1
                                current_string = [word]
                            continue
                        elif flag == 2:
                            #block_img = img[prev_block[1]:prev_block[3], prev_block[0]:prev_block[2]]
                            block_img = self.__modify_image(block_img, flag=2)
                            word = self.text_recognition(block_img, self.tes_model)
                            t = False
                            if word != '':
                                #best_word = specs[0]
                                #rate_best = 0
                                for word_true in specs:
                                    rate = self.rightBD.cmp_rates(word.lower(), word_true)
                                    if rate > rate_best:
                                        best_word = word_true
                                        rate_best = rate
                                if rate_best > 0.8:
                                    parts.append([words[ind], best_word])
                                    break
                            for w in word:
                                if w.isdigit():
                                    parts = prev_string
                                    t = True
                                    break
                            if t:
                                break
                            if abs(block[1] - y_prev) > 5:
                                prev_string = current_string
                                current_string = [[words[ind], word]]
                            else:
                                current_string.append([words[ind], word])
                            y_prev = block[1]
                            #print('y_prev: ', word, y_prev)

                            prev_block = block

                            continue

                    elif flag[1] == 1 and i == 0 and self.compare(block, self.doctr_model.max_block_left):
                        parts.append(words[ind])
                        info1.append(parts.copy())
                        parts.clear()
                        flag[1]+=1
                        continue

                    elif self.compare(block, self.doctr_model.prev_max_block_left) and i==0:
                        #parts.append(words[save_prev_word_ind])
                        info1.append(parts.copy())
                        #prev_y = block[1]
                        parts.clear()
                        parts.append(words[ind])
                        flag[1] += 1
                        continue

                    elif block[3] > self.doctr_model.max_block_left[3] and i == 0 and flag[1] == 2:
                        if self.check_digit(words[ind]) and count_digit_left < 2:
                            if len(words[ind]) < 3:
                                continue
                            count_digit_left +=1
                            if count_digit_left == 1:
                                min_gap_x = (block[2]-block[0])/6
                            #parts.append(words[ind])
                            block_img = self.__modify_image(block_img, flag=2)

                            word = self.text_recognition(block_img, self.tes_model)
                            parts.append([words[ind], word])
                            if count_digit_left == 2:
                                info1.append(parts.copy())
                                parts.clear()
                                prev_y = block[1]
                            continue
                        elif self.check_digit(words[ind]) and count_digit_left >=2:
                            if abs(block[1] - prev_y) >5:
                                left_count_strings+=1
                            prev_y = block[1]

                            if left_count_strings != 3 and left_count_strings != 5:
                                continue
                            if 'S' in words[ind]:
                                words[ind] = words[ind].replace('S', '5')
                            if len(words[ind]) < 2:
                                continue
                            if left_count_strings == 3:
                                block_img = self.__modify_image(block_img, flag=2)
                                word = self.text_recognition(block_img, self.tes_model)
                                parts.append([words[ind], word])
                                continue
                            left_gap = block[0] - min_gap_x
                            right_gap = block[2] + min_gap_x
                            block_img = self.__modify_image(block_img, flag=2)
                            word = self.text_recognition(block_img, self.tes_model)
                            if [words[ind], word] in parts:
                                continue
                            if blocks[ind-1][0]<=left_gap<=blocks[ind-1][2]:
                                prev_block = img[blocks[ind-1][1]:blocks[ind-1][3],
                                             blocks[ind-1][0]:blocks[ind-1][2]]
                                prev_block = self.__modify_image(prev_block, flag=2)
                                word = self.text_recognition(prev_block, self.tes_model)
                                for j in range(len(parts)):
                                    if words[ind-1] == parts[j][0]:
                                        break
                                else:
                                #if [words[ind-1], word] not in parts:
                                    parts.append([words[ind-1], word])
                            #block_img = self.__modify_image(block_img, flag=2)
                            word = self.text_recognition(block_img, self.tes_model)
                            parts.append([words[ind], word])
                            if len(blocks) != ind+1:
                                if blocks[ind+1][0]<=right_gap<=blocks[ind+1][2]:
                                    next_block = img[blocks[ind+1][1]:blocks[ind+1][3],
                                                 blocks[ind+1][0]:blocks[ind+1][2]]
                                    next_block = self.__modify_image(next_block, flag=2)
                                    word = self.text_recognition(next_block, self.tes_model)
                                    for j in range(len(parts)):
                                        if words[ind +1] == parts[j][0]:
                                            break
                                    else:
                                    #if [words[ind + 1], word] not in parts:
                                        parts.append([words[ind+1], word])
                            continue
                        else:
                            if count_digit_left >=2:
                                #if left_count_strings == 5:
                                    #block_img = self.__modify_image(block_img, flag=2)
                                    #save_prev_word_ind = ind
                                    #word = self.text_recognition(block_img, self.tes_model)
                                    #parts.append([words[ind], word])
                                if abs(block[1]-prev_y) > 5:
                                    left_count_strings +=1
                                    prev_y = block[1]
                            continue




                    if i==0 and flag[1] == 0:
                        self.city_possible[0]  = self.city_possible[1]
                        self.city_possible[1] = words[ind]
                    if i == 0 and flag[1] != 2:
                        block_img = self.__modify_image(block_img, flag = 2)
                        #print(words[ind])
                        #save_prev_word_ind = ind
                        word = self.text_recognition(block_img, self.tes_model)
                        parts.append(word)

                #    new_words.append(words[ind])
                #    right_blocks.append(block)


            info1.append(parts.copy())
            info.append(info1.copy())
            #break
            #blocks = right_blocks[:(len(right_blocks)*3)//4]
            #words = new_words[:(len(new_words)*3)//4]
            info1.clear()
            flag = 0




        return info


    '''def __compare(self, block1, block2):
        for i in range(len(block1)):
            if block1[i] != block2[i]:
                return False
        return True

    

    # 4
    def __text_recognition(self, block, model):
        word = model.detect_text(block)
        return word
    '''

    # 5.1
    def __chooseBD_left(self, info):
        '''
        Сопоставление текстовой информации на лвой стороне изображения с базами данных
        info: массив, содержащий информацию на левой стороне
        :return: заполненный словарь для левой стороны изображения
        '''
        words = []
        new_word = ''
        chapter = []
        #rightBD = FindWord()
        for i in range(len(info)):
            chapter = []
            #for part in info[i]:
            if i == 0:
                #probable_city = info[i][-1]
                new_word = info[i][:-1]
                new_word = [x for x in new_word if x != '']
                new_word = ' '.join(new_word)
                new_word = self.rightBD.remove_signs(new_word)
                new_word = new_word.replace('\n', ' ')
                try:
                    cities = self.rightBD.findCity(info[i][-2:], self.city_possible)
                except:
                    cities = ['']
                try:
                    universities = self.rightBD.findUniversity(new_word, cities[0])
                except:
                    universities = ['']
                self.result['Места:'] = cities
                self.result['Университеты:'] = universities
            if i == 1:
                word = info[i][-1]
                try:
                    level = self.rightBD.findLevel(word)
                except:
                    level = ''
                self.result['Степень:'] = level
            if i == 2:
                series = info[i][0][0]
                number = info[i][1][0]
                try:
                    series, number = self.rightBD.findNumberSeries(series, number)
                except:
                    series = '0'*6
                    number = '0'*7
                self.result['Серия:'] = series
                self.result['Номер:'] = number
            if i == 3:
                try:
                    reg_number = info[i][0]
                    data_given = info[i][1:]
                    reg_number, datas_given = self.rightBD.findRegNumbDataGiven(reg_number, data_given)
                except:
                    reg_number = '0'*3
                    datas_given = ['__ ______ 20__ года']
                self.result['Рег. номер:'] = reg_number
                self.result['Даты выдачи:'] = datas_given
                print("\n\nРезультат:\n", self.result, "\n\n")
                return
            #new_word = rightBD.compare_strings(new_word)
            if new_word == None:
                pass
                #print('None word: ', new_word)
            chapter.append(new_word)

            #print('clean_string')

        words.append(chapter)

        #d = random.randint(0, 100)
        #file = open(f'result_{d}.txt','w')
        #for chapter in words:
        #    for word in chapter:
        #        file.write(word)
        #        file.write('\n')

    # 5.2
    def __chooseBD_right(self, info):
        '''
        Сопоставление текстовой информации на правой стороне изображения с базами данных
        info: массив, содержащий информацию на левой стороне
        :return: заполненный словарь для правой стороны изображения
        '''
        confidence = True
        try:
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

                    try:
                        fio = self.rightBD.findFIO([family, name, surname])
                    except:
                        fio = ['','','']
                    self.result['Фамилия:'] = fio[0]
                    self.result['Имя:'] = fio[1]
                    self.result['Отчество:'] = fio[2]


                if i == 1:
                    spec_number = info[i][0]
                    if not self.check_digit(spec_number):
                        spec_number = info[i][1]
                    spec_name = info[i+1]
                    try:
                        spec_number, spec_name, confidence = self.rightBD.findSpeciality(spec_number, spec_name, self.result['Степень:'])
                    except:
                        confidence = False
                        spec_number = ''
                        spec_name = ''
                    if spec_number != '' and confidence:
                        self.result['Специальность:'] = spec_name[0]
                        self.result['Номер специальности:'] = spec_number
                    else:
                        self.result['Специальность:'] = spec_name
                        self.result['Номер специальности:'] = spec_number

                if i == 3:
                    if self.result['Степень:'] == 'специалитет':
                        special_number = self.result['Номер специальности:']
                        if confidence == False:
                            special_number = ''
                        cmp_speciality = info[i][0]
                        try:
                            quality = self.rightBD.findQuality(cmp_speciality, special_number)
                        except:
                            quality = ''
                        self.result['Квалификация:'] = quality
                    else:
                        if self.result['Степень:'] != '':
                            self.result.pop('Квалификация:')
                            break
                        cmp_speciality = info[i][0]
                        try:
                            level_1 = self.rightBD.findCmpLevel(cmp_speciality, self.result['Степень:'])
                        except:
                            level_1 = ''
                        if self.result['Степень:'] == '' and level_1 != None:
                            self.result['Степень:'] = level_1
                    break
        except:
            return
        return

    def form_result(self):
        self.result['Места:'] = ''
        self.result['Университеты:'] = ['']
        self.result['Степень:'] = ''
        self.result['Серия:'] = '0'*6
        self.result['Номер:'] = '0'*7
        self.result['Рег. номер:'] = '00'
        self.result['Даты выдачи:'] = 'xx месяца 20xx года'
        self.result['Фамилия:'] = ''
        self.result['Имя:'] = ''
        self.result['Отчество:'] = ''
        self.result['Специальность:'] = ''
        self.result['Номер специальности:'] = '00.00.00'
        self.result['Квалификация:'] = ''

    def conveer(self):
        '''
        Ход работы всего процесса OCR
        :return: заполненный словарь информации для диплома
        '''
        #Trye

        # 1
        self.form_result()
        self.rightBD.initialize()
        image = self.orig_image.copy()
        try:
            image = self.__modify_image(image)
        except:
            pass
        #show_img(image, 'after modifing')
        # 2
        left_block, right_block = self.__segmentation(image)

        # 3
        info = self.__symbol_blocks(left_block, right_block, image)

        print('left chapter:', info[0])
        print('rifht chapter:', info[1])
        # 4
        self.__chooseBD_left(info[0])
        self.__chooseBD_right(info[1])
        print('\nРезультат:\n', self.result)
        return self.result
