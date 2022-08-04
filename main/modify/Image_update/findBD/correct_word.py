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

from chooseBD import Search


class Doctr(Search):
    delimiters = '.', ',' , ';'

    comp = re.compile('[^аеоиыуяэюАЕОИЫУЯЭЮ]+[^ьъЬЪ]+')

    letters_to_num = {
        'A':'4',
        'O':'0',
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
        'IL': ['П', 'Ц', 'Щ'],
        'I': ['П', 'Д', 'Г', 'Л', 'И'],
        'J': ['У','Л', 'Я'],
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
        'T': ['Т', 'Г', 'П'],
        'U': ['И', 'Н', 'Ц'],
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
        '1': ['1', 'I'],
        '2': ['2'],
        '5': ['5'],
        '6': ['6'],
        '7': ['7'],
        '8': ['8'],
        '9': ['9', 'Ч'],
        '0': ['0'],
        '#': ['Ж', 'Ф'],
        '*': ['Ж', 'Ф']
    }
    prob_words = []

    filenames = []
    def choose_model(self, straight_page = False, predict = False):
        #if predict == True:
            #self.model = crnn_vgg16_bn(pretrained=True)
        #else:
            #self.model = linknet_resnet18(pretrained=True)
        self.model = ocr_predictor(det_arch='db_resnet50', reco_arch='crnn_vgg16_bn',
                                   pretrained=True, export_as_straight_boxes=True) #'db_resnet50'
        self.model.compiled_metrics = None
        #self.doc = DocumentFile.from_images(filenames[0])
        self.morph = pymorphy2.MorphAnalyzer(lang='ru')
        self.ban = 'ыьйъ'
        self.ban +=self.ban.upper()
        sogl1 = 'йцкгшщзхфвпрджчмьб'
        sogl2 = 'нслт'
        eng_sogl = 'qwrtpsdfghjklzxcvbnm'
        eng_gl = 'aeyuio'
        self.eng_sogl = eng_sogl+eng_sogl.upper()
        self.eng_gl = eng_gl+eng_gl.upper()
        self.gl = 'уеёаоэяию'
        self.sogl = sogl1+sogl1.upper()+sogl2+sogl2.upper()
        self.gl += self.gl.upper()

    #def add_files(self, filename):
    #    self.filenames.append(filename)
    def poss_words1(self, word):
        count_num = 0
        count_let = 0
        #word = word.lower()
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
                    word = word[0:j] + word[j+1:]
                    l = len(word)
                    continue
            else:
                if word[j].isalpha():
                    word = word[0:j] + word[j + 1:]
                    l = len(word)
                    continue
            j+=1
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
        l = len(words)
        if count_num > count_let:
            flag = 0
        else:
            flag = 1

        while i < l:
            if flag == 0:
                x = [j for j in words[i] if not j.isalpha()]
                #x = [0 for j in words[i] if j.isalpha()]
                #count_let += len(x)
                #if count_let > 0:
                #    del words[i]
                #    l = len(words)
                #else:

                i += 1
                #count_let = 0
            else:
                x= [0 for j in words[i] if j.isdigit()]
                count_num += len(x)
                if count_num > 0:
                    del words[i]
                    l = len(words)
                else:
                    """for t in self.delimiters:
                        if t in words[i]:
                            c = words[i].split(t)
                            words += c
                            del words[i]
                            l = len(words)
                            i-=1
                            break
                        """
                    i += 1
                count_num = 0

        return words, flag

    def translate_word(self, word, words):
        all_words = []
        for (ind, letter) in enumerate(word):
            if letter in self.alphabet.keys():
                if len(self.alphabet[letter]) > 1:
                    for j in range(0, min(3, len(self.alphabet[letter]))):     #Органичим на всего 3 буквы можно заменять
                        t = words.copy()
                        t.append(self.alphabet[letter][j])
                        all_words += self.translate_word(word[ind+1:len(word)], t)
                words.append(self.alphabet[letter][0])
            else:
                words.append(letter)

        #joint = ''.join(words).lower().capitalize()
        joint_orig = ''.join(words)
        #regex = '|'.join(map(re.escape, self.delimiters))
        #joint = re.split(regex, joint)
        #all_words += joint
        #print('joint: ', joint, self.morph.word_is_known(joint))
        #if self.morph.word_is_known(joint):
        all_words.append(joint_orig)
        #else:
        self.prob_words.append(joint_orig)
        #f (p[0].score > 0.5):
        #print(joint_orig)
        #all_words.append(joint)

        return all_words

    def filter_words(self, text):
        symbs = '<*^_'
        ind = 0
        l = len(text)
        while ind < l:
            for j in symbs:
                if j in text[ind]:
                    text[ind] = text[ind].replace(j, '')
            #if text[ind][0] == '-':
            #    text[ind] = text[ind].replace('-','')
            if text[ind][0] == '.':
                text[ind] = text[ind].replace('.','')

            if text[ind] == '':
                del text[ind]
                l = len(text)
            else:
                bad_symb = 0
                good_symb =0
                for i in text[ind]:
                    if i.isdigit() or i.isalpha():
                        good_symb +=1
                    else:
                        bad_symb +=1
                if good_symb < bad_symb:
                    del text[ind]
                    l = len(text)
                else:
                    ind+=1
        return text

    def filt_names(self, word):
        return not '^' in word

    def gramma(self, word):
        if len(word)!=0:
            if word[0] not in self.ban:
                if len(word) >= 2:
                    if word[0] in self.gl and word[1] in self.gl:      #Если первые две буквы слова гласные - убираем из списка
                        return None
                    for i in range(len(word)-2):
                        if (word[i] == word[i+1] and (word[i] in self.sogl[:len(self.sogl)-8] or word[i] in self.gl)):
                            return None
                        if (word[i] in self.gl and word[i+1] in 'ьЬъЪ') or (word[i+1] in self.gl and word[i] in 'ьЬъЪ'):
                            return None
                    if len(word) >= 4:
                        for i in range(len(word) - 4):
                            if (word[i] in self.sogl and word[i + 1] in self.sogl and word[i + 2] in self.sogl and word[i + 3] in self.sogl) or (word[i] in self.gl and word[i + 1] in self.gl and word[i + 2] in self.gl and word[i + 3] in self.gl):
                                return None

                if len(word) < 4:
                    for i in word:
                        if i in '-,:|@!^':
                            return None
                if len(word) == 3:
                    if word[0] in self.sogl and word[1] in self.sogl and word[2] in self.sogl:
                        return None
                    if word[0] in self.sogl:
                        if word[1] in self.gl and word[2] in self.gl and word[1] == word[2]:
                            return None
                    if word[2] in self.sogl:
                        if word[0] in self.gl and word[1] in self.gl and word[0] == word[1]:
                            return None


                if 2 < len(word) < 5:       # Фильтрует такие слова: стр здт апрс цвка и тд
                    for i in range(0, len(word)-2):
                        if (word[i:i+3][0] in self.sogl) and (word[i:i+3][1] in self.sogl) and (word[i:i+3][2] in self.sogl):
                            return None
                if len(word) >= 5:          # Если у нас на входе поступают слова типа слкудреватых, то оно преобразет в кудреватых, но оно также странник преобразует в транник
                    if (word[0] in self.sogl) and (word[1] in self.sogl) and (word[2] in self.sogl):
                        word = word[2:]

                upper_case = 0
                lower_case = 0
                pos_word = [i for i in word]
                for i in pos_word:
                    if i.isalpha():
                        if i.isupper():
                            upper_case+=1
                        else:
                            lower_case+=1
                if upper_case >= lower_case:
                    for i in range(len(pos_word)):
                        if pos_word[i].isalpha():
                            pos_word[i] = pos_word[i].upper()
                else:
                    for i in range(len(pos_word)):
                        if pos_word[i].isalpha():
                            pos_word[i] = pos_word[i].lower()

                return ''.join(pos_word)

            else:
                return None

        else:
            return None

    def check_digit(self, word):
        for j in word:
            if j.isdigit():
                return True
        return False

    def eng_gramma(self, word):
        # print('Filter word:', word)
        is_digit = self.check_digit(word)
        if not is_digit:
            word = self.remove_signs(word)
            count_upper = 0

            for symb in word:
                if symb.isupper():
                    count_upper += 1
            if count_upper >= len(word) // 2:
                word = [i for i in ''.join(word).upper()]
            else:
                return None

        if len(word) < 3 and not is_digit:
            return None
        else:
            if is_digit:
                return word
            elif len(word)>=5:
                for i in range(len(word)-5):
                    count_sogl = 0
                    count_gl = 0
                    for z in word[i:i+5]:
                        if z in self.eng_sogl:
                            count_sogl += 1

                    for z in word[i:i+4]:
                        if z in self.eng_gl:
                            count_gl += 1
                    if count_sogl == 5 or count_gl == 4:
                        return None
        return word

    def remove_signs(self, word):
        if '.' in word:
            word = word[word.index('.')+1:len(word)]
        if ',' in word:
            word = word[word.index(',')+1:len(word)]
        if ':' in word:
            word = word[word.index(':') + 1:len(word)]
        if '(' in word:
            word = word[word.index('(') + 1:len(word)]
        if ')' in word:
            word = word[word.index(')') + 1:len(word)]
        return word

    def filt_words(self, words):
        st = ' '.join(words)
        s1 = re.findall(r'[аеоиыуяэюАЕОИЫУЯЭЮ]+[ьъЬЪыЫйЙ]+', st, re.MULTILINE)
        s2 = re.findall(r'[ьъЬЪйЙыЫ]+[аеоиыуяэюАЕОИЫУЯЭЮ]+', st, re.MULTILINE)
        s = s1+s2

        k = [[i, '^'] for i in s]
        for r in k:
            st = st.replace(*r)

        words = st.split(' ')
        words = list(filter(self.filt_names, words))
        words = list(set(filter(None, (map(self.gramma, words)))))
        # print(words)
        #words = list(set(list(filter(self.gramma, words))))
        #print(words)
        words = list(map(self.remove_signs, words))
        if len(words) >0:
            if len(words[0]) < 2:
                words = []
        # print('filter words', words)
        return words

    def translating(self, words, flag):

        possible_words = []
        dictionary = {}

        # Вывод кол-ва слов для оценки производительности
        # print("words: ", words)
        #if len(words) != 0:
        #
        # print('len(words[0]):', len(words[0]))

        # Filter english words before filter
        # print('words before filter:', words)
        if len(words) != 0:
            min_l = len(words[0])
            words_copy = [words[0]]
            if len(words) > 1:
                for word in words[1:]:
                    if len(word) <= min_l:
                        words_copy.append(word)
            words = words_copy.copy()
            del words_copy

        print('words: ', words)
        #input()
        words = list(filter(None, map(self.eng_gramma, words)))
        # print('words after filter:', words)
        if len(words)==0:
            return ''
        t1 = time.time()
        limit = 55000      # Лимит по кол-ву слов на обработку
        if len(words[0])> 11:
            limit = 55000
        if len(words[0]) > 20:
            return ''
        # print('len(words):', len(words))
        #input()
        for word in words:
            all_words_poss = np.array(self.translate_word(word, []))
            print('all_words_poss: ', len(all_words_poss))
            if len(all_words_poss) > limit or len(all_words_poss[0])>11:
                #limit = 100000                                      # Лимит по кол-ву слов на обработку
                all_words_poss = np.array(all_words_poss)
                all_words_poss = np.random.choice(all_words_poss, size = limit)  # Ограничиваем размер по кол-ву слов
            print('all_words_poss1:', all_words_poss)
            #fil = np.array(list(map(self.morph.word_is_known, all_words_poss)))
            #all_words_poss = list(all_words_poss[fil])
            #print('all_words_poss2:', all_words_poss)
            #if len(all_words_poss)!=0:
            #possible_words += all_words_poss    #filter((map(self.morph.word_is_known, all_words_poss)))
        if len(possible_words) == 0:
            possible_words = self.prob_words.copy()
        self.prob_words = []

        # Time for this operation
        print('Time for translate and word is known is ', time.time()-t1)
        #print('poss_word1: ', possible_words) 3PNRUSOLEGNIKOVCGRUSLANCBAHTIBROVI3
        #possible_words, flag = self.poss_word(possible_words)
        # print('len: ', len(possible_words))
        #input()
        limit = 30000
        if len(possible_words)>limit:
            possible_words = np.array(possible_words)
            possible_words = list(np.random.choice(possible_words, size=limit))
            #possible_words = possible_words[:75000]

        # print('poss_words2: ', len(possible_words))
        if flag == 0:
            return ''.join(possible_words[0])
        dict = enchant.Dict('ru_RU')

        max_seq = -1
        all_words = [[]]
        max_pos_words = ['']

        t1 = time.time()
        if flag == 1:
            if len(possible_words[0]) >=15:
                possible_word = Search.median(self, possible_words)
                return possible_word
            if len(possible_words) > limit:
                possible_words = np.array(possible_words)
                possible_words = list(np.random.choice(possible_words, size=limit))    # Ограничиваем кол-во слов после обработки
            # print('poss words: ', possible_words)
            possible_words = self.filt_words(possible_words)

            #possible_words = list(map(self.remove_signs, possible_words))
            # print('after filter: ', len(possible_words))
            if len(possible_words) !=0:
                if '.' in possible_words[0]:
                    all_words = [[] for i in possible_words[0].split('.')]
                    max_pos_words = ['' for i in all_words]
            else:
                return ''

        best_ratio = 0
        median_words = []

        limit_poss_words = 60
        if len(possible_words) > limit_poss_words:
            possible_words = np.array(possible_words)
            possible_words = list(np.random.choice(possible_words, size=limit_poss_words))   # Ограничиваем кол-во слов после обработки
        # print('Amount of possible words after filtering is ', len(possible_words))

        # Для тестирования одной итерации цикла
        #flag = 0
        #ind = 0
        #t2 = 0
        #l = len(possible_words)
        #while ind < l:
        for (ind, word) in enumerate(possible_words):
            #all_words = []
            word = possible_words[ind]
            """if flag == 0:
                t2 = time.time()"""
            if best_ratio == 1:
                break
            w = [word]
            if len(all_words) > 1:
                w = word.split('.')
                w = [i.lower().capitalize() for i in w]
                #print(w)
            print('Test w: ', w)
            for (ind1, s1) in enumerate(w):
                if s1 == '':
                    max_pos_words[ind1] = ''
                    all_words[ind1].append(s1)
                    continue
                print('Test s1:', s1)
                d = dict.suggest(s1)
                print('Dict suggest: ', d)
                if len(d) != 0:
                    opcodes = []
                    all_opcodes = {}
                    for d_word in d:
                        op = Search.opcodes(self, d_word, s1)
                        opcodes.append(op)
                        if op in all_opcodes.keys():
                            all_opcodes[op].append(d_word)
                        else:
                            all_opcodes[op] = [d_word]
                    op = min(opcodes)
                    new_word = Search.quickmedian(self, all_opcodes[op])
                    #print(d)
                    #if d == None:

                    #print('d:', d)
                    #input()
                    #median_words += d
                    ratio = Search.ratio(self, s1, new_word)
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
                '''if len(d) > 0:
                    ratio = Search.ratio(self, s1, d[0])
                    #print('ratio: ', ratio, s1)
                    if ratio > max_seq:
                        max_seq = ratio
                        max_pos_words[ind1] = s1
                    if ratio >= 0.7:
                        if ratio == 1.0:
                            all_words[ind1] = [s1]
                            best_ratio = 1
                            break
                        all_words[ind1].append(s1)'''
            """if flag == 0:
                flag = 1
                delta = time.time() - t2
                print('delta ', delta)
                print('length of possible words', len(possible_words))
                #input()
                if delta*len(possible_words) > 4:
                    possible_words = np.array(possible_words[1:])
                    possible_words = list(np.random.choice(possible_words[1:], size=len(possible_words[1:])//2))
                    l = len(possible_words)
            ind+=1"""

        print("Time after finding possible words from LV is", time.time() - t1)
        #med_word = Search.median(self, median_words)
        #print(f'median word: {med_word}')
        #quickmed = Search.quickmedian(self, median_words)
        #print(f'quickmedian word: {quickmed}')
        #setmedian = Search.setmedian(self, median_words)
        #print(f'setmedian word: {setmedian}')

        for i in range(len(all_words)):
            if type(all_words[i]) == list:
                #if len(all_words[i]) == 1:
                #    possible_word = all
                all_words[i] = [x for x in all_words[i] if x is not None]
                possible_word = Search.quickmedian(self, all_words[i])
                if possible_word == None:
                    print('max_pos_word: ', max_pos_words[i])
                    possible_word = max_pos_words[i]
                all_words[i] = possible_word


        if flag == 0:
            possible_word = '.'.join(all_words)
        else:
            possible_word = '.'.join(all_words)

        #if
        return possible_word


        #d = list(set(list(dict.suggest(word))))
        """if len(d) < 10:
                d = d*10
                d = d[:10]"""
        #print('word: ', word, d)
        #input()
        """if len(d) > 0:
                ratio  = Search.ratio(self, word, d[0])
                print('ratio: ', ratio)
                if ratio >= 0.7:
                    if ratio > max_seq:
                        max_seq = ratio
                        all_words = []
                        all_words.append(word)
                        all_words.append(d[0])
                        max_pos_word = word
                    elif ratio == max_seq:
                        all_words.append(word)
                        all_words.append(d[0])"""


        """seq = Search.seqratio(self, d, [word]*len(d))
            #print('seq: ', seq)
            if seq >= 0.7 and len(d) > 0:
                #print(word)
                if seq > max_seq:
                    max_seq = seq
                    max_pos_word = word
                all_words.append(word)
            #seq = max
            d = [x for x in d if Search.ratio(self, x, word)>seq]
            if len(d) < 10:
                d = d*10
                d = d[:10]

            all_words += d"""
        #print(all_words)
        #possible_words[ind] = Search.median(self, all_words)
        """all_words = [x for x in all_words if x is not None]
        possible_word = Search.median(self, all_words)
        if possible_word == None:
            possible_word = max_pos_word
        print(possible_word)
        print(max_pos_word)
        return possible_word"""

        #Г
        #T O JI IATTI
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

    def look_geometry(self, new_word, blocks, words):
        #print(new_word)
        for block in new_word:
            if 'value' in block.keys():
                words.append(block['value'])
            if 'pages' in block.keys():
                res = self.look_geometry(block['pages'], blocks, words)
                blocks = res[0]
                words = res[1]
            if 'geometry' in block.keys():
                blocks.append(block['geometry'])
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

    @staticmethod
    def to_right_size(H, W, blocks):
        new_block = []
        for block in blocks:
            new_block.append([int(block[0][0]*W), int(block[0][1]*H), int(block[1][0]*W), int(block[1][1]*H)])
        return new_block

    def find_contours(self, img):
        img_path = 'save1.jpg'
        cv2.imwrite(img_path, img)

        doc = DocumentFile.from_images(img_path)

        result = self.model(doc)
        # Show DocTR result
        result.show(doc)
        json = result.export()

        blocks = []
        words = []
        res = self.look_geometry(json['pages'], blocks, words)
        blocks = res[0]
        words = res[1]
        #print('blocks', blocks)
        #for block in json['pages']:
        #    self.look_geometry(block)

        os.system('rm save1.jpg')

        H, W = img.shape[:2]
        blocks = self.to_right_size(H, W, blocks)
        #result.show(doc)
        #print(len(blocks), len(words))

        blocks = [x for n, x in enumerate(blocks) if x not in blocks[:n]]
        words = [x for n, x in enumerate(words) if x not in blocks[:n]]
        #print(blocks)
        #print(words)

        return [blocks, words]

    def find_text(self, img):
        img_path = 'save1.jpg'
        cv2.imwrite(img_path, img)
        doc = DocumentFile.from_images(img_path)

        result = self.model(doc)

        json = result.export()
        os.system('rm save1.jpg')
        print(json)
        return json

    def __synthetic_pages__(self, result):
        synthetic_pages = result.synthesize()
        plt.imshow(synthetic_pages[0])
        plt.axis('off')
        plt.show()