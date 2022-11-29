import numpy as np
import Levenshtein as lv
import re
import random

class BD_Functions:
    '''
    Класс полезных функций для chooseBD
    '''
    def __init__(self):
        pass

    def find_regions(self, regions, places):
        obl = dict()
        krai = dict()
        for i in range(len(regions)):
            reg = regions[i].split(' ')
            if 'область' in reg:
                reg = list(filter(lambda a: a!= 'область', reg))
                reg = ' '.join(reg)
                all_places = places[np.where(places == regions[i])[0], 0]
                for j in range(len(all_places)):
                    place = all_places[j].split(' ')
                    place = list(filter(self.filter_mesta, place))
                    place = ' '.join(place)
                    all_places[j] = place
                obl[reg] = all_places #places[np.where(places == regions[i])[0], 0]
            if 'край' in reg:
                reg = list(filter(lambda a: a != 'край', reg))
                reg = ' '.join(reg)
                all_places = places[np.where(places == regions[i])[0], 0]
                for j in range(len(all_places)):
                    place = all_places[j].split(' ')
                    place = list(filter(self.filter_mesta, place))
                    place = ' '.join(place)
                    all_places[j] = place
                krai[reg] = all_places #places[np.where(places == regions[i])[0], 0]
            #region[regions[i]] = places[np.where(places == regions[i])[0], 0]
        return obl, krai

    def region_filter(self, word):
        w = word.split(' ')
        sec_word = w[1].lower()
        if sec_word == 'республика':
            sec_word = 'респ.'
        elif sec_word == 'край':
            pass
        elif sec_word == 'федерального':
            sec_word = w[-1]
        else:
            if w[0] == 'Республика':
                return 'Респ. ' + ' '.join(w[1:])
        return w[0] + ' ' + sec_word

    def name_filter(self, word):
        w = word.split()
        f = w[0]
        if f == 'село':
            f = 'с'
        elif f == 'станция' or 'железнодорож' in f or 'насел' in f or 'разъезд' == f:
            return ' '.join(w)
        else:
            f = w[0][0:3] + ''
        return f + '. ' + ' '.join(w[1:])


    def filter_mesta(self, word):
        word = word.lower()
        if word == 'город' or word == 'деревня' or word == 'хутор' or word == 'село':
            return None
        else:
            return word

    def filter_names(self, word):
        return None if 'і' in word or 'ї' in word else word

    def filter_city(self, city):
        city = city.lower()
        if 'респ' in city or 'обл' in city or 'гор' in city:
            return None
        else:
            return city.capitalize()

    def print(self, about, string):
        pass

    def sort_data(self, data):
        try:
            if len(data)<3:
                for j in range(3-len(data)):
                    data.append(data[0])
            data.sort(key=lambda x: x[3])
            k = 0
            new_data = [0,1,2]
            for j in range(-1, -4, -1):
                new_data[k] = data[j][0:]
                k += 1
            return new_data
        except:
            return [[None, None, 0] for i in range(3)]

    def delete_symbs(self, s1):
        new_s1 = []
        for i in range(len(s1)):
            if s1[i] in s1[:i]+s1[i+1:] and (len(s1[i]) == 2 or len(s1[i]) > 5):
                new_s1.append(s1[i])
        return new_s1

    def ratio(self, s1, s2):
        rate1 = lv.ratio(s1, s2)
        rate2 = lv.jaro(s1, s2)
        rate = (rate1 + rate2) / 2
        return rate

    def distance(self, s1, s2):
        return lv.distance(s1, s2)


class Correct_Word_Fuctions:
    '''
    Класс функиий для correct_word
    '''
    def __init__(self):
        self.gl = 'уеёаоэяию'
        self.gl += self.gl.upper()

        sogl1 = 'йцкгшщзхфвпрджчмьб'
        sogl2 = 'нслт'
        self.sogl = sogl1 + sogl1.upper() + sogl2 + sogl2.upper()

        self.ban = 'ыьйъ'
        self.ban += self.ban.upper()

        eng_sogl = 'qwrtpsdfghjklzxcvbnm'
        eng_gl = 'aeyuio'
        self.eng_sogl = eng_sogl + eng_sogl.upper()
        self.eng_gl = eng_gl + eng_gl.upper()

    def filter_words(self, text):
        symbs = '<*^_'
        ind = 0
        length = len(text)
        while ind < length:
            for j in symbs:
                if j in text[ind]:
                    text[ind] = text[ind].replace(j, '')
            # if text[ind][0] == '-':
            #    text[ind] = text[ind].replace('-','')
            if text[ind] == '':
                del text[ind]
                length = len(text)
                continue
            if text[ind][0] == '.':
                text[ind] = text[ind].replace('.', '')

            if text[ind] == '':
                del text[ind]
                length = len(text)
            else:
                bad_symb = 0
                good_symb = 0
                for i in text[ind]:
                    if i.isdigit() or i.isalpha():
                        good_symb += 1
                    else:
                        bad_symb += 1
                if good_symb < bad_symb:
                    del text[ind]
                    length = len(text)
                else:
                    ind += 1
        return text


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


    def filt_names(self, word):
        return not '^' in word

    def gramma(self, word):
        if len(word) != 0:
            if word[0] not in self.ban:
                if len(word) >= 2:
                    if word[0] in self.gl and word[1] in self.gl:  # Если первые две буквы слова гласные - убираем из списка
                        return None
                    for i in range(len(word) - 2):
                        if (word[i] == word[i + 1] and (
                                word[i] in self.sogl[:len(self.sogl) - 8] or word[i] in self.gl)):
                            return None
                        if (word[i] in self.gl and word[i + 1] in 'ьЬъЪ') or (
                                word[i + 1] in self.gl and word[i] in 'ьЬъЪ'):
                            return None
                    if len(word) >= 4:
                        for i in range(len(word) - 4):
                            if (word[i] in self.sogl and word[i + 1] in self.sogl and word[i + 2] in self.sogl and word[i + 3] in self.sogl) or (
                                    word[i] in self.gl and word[i + 1] in self.gl and word[i + 2] in self.gl and word[
                                i + 3] in self.gl):
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

                if 2 < len(word) < 5:  # Фильтрует такие слова: стр здт апрс цвка и тд
                    for i in range(0, len(word) - 2):
                        if (word[i:i + 3][0] in self.sogl) and (word[i:i + 3][1] in self.sogl) and (
                                word[i:i + 3][2] in self.sogl):
                            return None
                if len(word) >= 5:  # Если у нас на входе поступают слова типа слкудреватых, то оно преобразет в кудреватых, но оно также странник преобразует в транник
                    if (word[0] in self.sogl) and (word[1] in self.sogl) and (word[2] in self.sogl):
                        word = word[2:]

                upper_case = 0
                lower_case = 0
                pos_word = [i for i in word]
                for i in pos_word:
                    if i.isalpha():
                        if i.isupper():
                            upper_case += 1
                        else:
                            lower_case += 1
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
                word = [i.upper() for i in word]
            else:
                return None

        if len(word) < 3 and not is_digit:
            return None
        else:
            if is_digit:
                return word
            elif len(word) >= 5:
                all_gl = []
                all_sogl = []
                for i in range(len(word) - 5):
                    count_sogl = 0
                    count_gl = 0
                    """for z in word[i:i+5]:
                        if z in self.eng_sogl:
                            all_sogl.append(z)
                            count_sogl += 1

                    for z in word[i:i+4]:
                        if z in self.eng_gl:
                            all_gl.append(z)
                            count_gl += 1"""
                    all_gl = list(set(all_gl))
                    all_sogl = list(set(all_sogl))
                    if (count_sogl == 5 and len(all_sogl) < 4) or (count_gl == 4 and len(all_gl) < 3) or \
                            (count_sogl == 4 and len(all_sogl) == 1) or (count_gl == 3 and len(all_gl) == 1):
                        return None
                    all_gl = []
                    all_sogl = []
        return word


    def remove_signs(self, word):
        if '.' in word:
            word = word[word.index('.') + 1:len(word)]
        if ',' in word:
            word = word[word.index(',') + 1:len(word)]
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
        s = s1 + s2

        k = [[i, '^'] for i in s]
        for r in k:
            st = st.replace(*r)

        words = st.split(' ')
        words = list(filter(self.filt_names, words))
        words = list(set(filter(None, (map(self.gramma, words)))))
        # print(words)
        # words = list(set(list(filter(self.gramma, words))))
        # print(words)
        words = list(map(self.remove_signs, words))
        if len(words) > 0:
            if len(words[0]) < 2:
                words = []
        # print('filter words', words)
        return words


    def limit_words(self, limit, words):
        if len(words) > limit:
            words = np.array(words)
            words = list(
                np.random.choice(words, size=limit))  # Ограничиваем кол-во слов после обработки
        # print('Amount of possible words after filtering is ', len(possible_words))
        if len(words) < limit and len(words) != 0:
            num_iter = limit // len(words)
            another_poss_words = words.copy()
            for i in range(num_iter):
                words += another_poss_words
            #for i in range(limit_poss_words-len(possible_words)):
            #    z = random.choice(possible_words)
            #    possible_words.append(z)
            random.shuffle(words)
            words = words[:limit]

        random.shuffle(words)
        return words

    def check_digit(self, word):
        for j in word:
            if j.isdigit():
                return True
        return False

    def clean_translating_words(self, words):
        if len(words) != 0:
            min_l = len(words[0])
            words_copy = [words[0]]
            if len(words) > 1:
                for word in words[1:]:
                    if len(word) <= min_l:
                        words_copy.append(word)
            words = words_copy.copy()
            del words_copy

        words = list(filter(None, map(self.eng_gramma, words)))
        return words

    def concatenate(self, all_words, max_pos_words):
        for i in range(len(all_words)):
            if type(all_words[i]) == list:
                # if len(all_words[i]) == 1:
                #    possible_word = all
                all_words[i] = [x for x in all_words[i] if x is not None]
                possible_word = self.quickmedian(all_words[i])
                if possible_word == None:
                    # self.printstring(max_pos_words[i], 'max_pos_word')
                    # print('max_pos_word: ', max_pos_words[i])
                    possible_word = max_pos_words[i]
                all_words[i] = possible_word
        return all_words


    def ratio(self, s1, s2):
        rate1 = lv.ratio(s1, s2)
        rate2 = lv.jaro(s1, s2)
        rate = (rate1 + rate2) / 2
        return rate

    def opcodes(self, s1, s2):
        return len(lv.opcodes(s1, s2))

    def median(self, sequence):
        return lv.quickmedian(sequence)

    def median_improve(self, word, sequence):
        return lv.median_improve(word, sequence)

    def quickmedian(self, sequence):
        return lv.quickmedian(sequence)

    def setmedian(self, sequence):
        return lv.setmedian(sequence)
