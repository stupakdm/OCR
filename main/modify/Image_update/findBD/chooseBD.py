from difflib import SequenceMatcher

import Levenshtein as lv
import numpy as np
import pandas as pd
import time

class Search:
    special_words = ('дата', 'рождения', 'место', 'гор', 'код', 'обл', 'муж', 'пос', 'пол', 'россия', 'федерация')
    # 'имя', 'фамилия', 'отчество'
    """person_data = {
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
    }"""

    def start(self):
        # self.name_filepath = 'templates/Names and surnames.csv'
        # self.fams_filepath = 'templates/families2.csv'
        curr_dir = 'modify/Image_update/findBD/data/'
        self.df_name = pd.read_csv(f'{curr_dir}Names_and_surnames2.csv', encoding='utf-8')
        self.names_surnames = np.array(self.df_name)
        self.names_surnames = self.names_surnames[np.where(self.names_surnames[:,2] >150)[0]]
        #self.names = np.array(self.df_name['name'])
        #self.surnames = np.array(self.df_name['surname'])
        #self.priority = np.array(self.df_name['priority'])

        self.df_families = pd.read_csv(f'{curr_dir}families3.csv', encoding='utf-8')
        self.families = np.array(self.df_families['family'])

        self.df_cities = pd.read_csv(f'{curr_dir}RussianCities.tsv', sep='\t', index_col=False, encoding='utf-8')
        self.df_places = np.array(self.df_cities[['name', 'region_name']])
        self.cities = None #self.find_something(self.df_places, 'город')
        self.derevni = None #self.find_something(self.df_places, 'деревня')
        self.sela = None #self.find_something(self.df_places, 'село')
        self.poselki = None# self.find_something(self.df_places, 'посёлок')
        self.hutor = None #self.find_something(self.df_places, 'хутор')
        self.find_something(self.df_places, 'city')

        arr_name = np.array(self.df_cities['name'])
        arr_region = np.array(self.df_cities['region_name'])
        arr_region = np.unique(arr_region)
        self.obl, self.krai = self.find_regions(arr_region, self.df_places)
        self.arr_region = np.array(list(map(self.region_filter, arr_region)))
        self.arr_name = np.array(list(filter(self.filter_names, map(self.name_filter, arr_name))))

        self.df_fms = pd.read_csv(f'{curr_dir}fms_unit.csv', sep=',', index_col=False, encoding='utf-8')
        self.codes = {}
        places = np.array(self.df_fms['name'])
        code = np.array(self.df_fms['code'])
        for i in range(len(code)):
            if code[i] not in self.codes.keys():
                self.codes[code[i]] = [places[i], ]
            else:
                self.codes[code[i]].append(places[i])

        self.possible_fms = None

        del self.df_cities
        del self.df_families
        del self.df_name
        del self.df_fms

    def find_something(self, arrays, word, flag= False):
        array = np.array(['Москва'])
        self.cities = array.copy()
        self.derevni = array.copy()
        self.sela = array.copy()
        self.poselki = array.copy()
        self.hutor = array.copy()
        for i in range(arrays.shape[0]):
            new_line = arrays[i][0].split(' ')
            if 'город' in new_line:
                new_line = list(filter(lambda a: a != 'город', new_line))
                self.cities = np.append(self.cities, np.array([' '.join(new_line)]), axis = 0)
            elif 'село' in new_line:
                new_line = list(filter(lambda a: a != 'село', new_line))
                self.sela = np.append(self.sela, np.array([' '.join(new_line)]), axis=0)
            elif 'посёлок' in new_line or 'поселок' in new_line:
                new_line = list(filter(lambda a: a != 'посёлок', new_line))
                self.poselki = np.append(self.poselki, np.array([' '.join(new_line)]), axis=0)
            elif 'деревня' in new_line:
                new_line = list(filter(lambda a: a != 'деревня', new_line))
                self.derevni = np.append(self.derevni, np.array([' '.join(new_line)]), axis=0)
            elif 'хутор' in new_line:
                new_line = list(filter(lambda a: a != 'хутор', new_line))
                self.hutor = np.append(self.hutor, np.array([' '.join(new_line)]), axis=0)

            """    new_line = list
            if word in new_line:
                self.cities = None  # self.find_something(self.df_places, 'город')
                self.derevni = None  # self.find_something(self.df_places, 'деревня')
                self.sela = None  # self.find_something(self.df_places, 'село')
                self.poselki = None  # self.find_something(self.df_places, 'посёлок')
                self.hutor
                new_line = list(filter(lambda a: a != word, new_line))
                arrays[i][0] = ' '.join(new_line)
                array = np.append(array, np.array([arrays[i][0]]), axis = 0)"""
        self.cities = np.delete(self.cities, 0, axis = 0)
        self.sela = np.delete(self.sela, 0, axis=0)
        self.derevni = np.delete(self.derevni, 0, axis=0)
        self.poselki = np.delete(self.poselki, 0, axis=0)
        self.hutor = np.delete(self.hutor, 0, axis=0)
        #array = np.delete(array, 0, axis = 0)
        return array

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

    def city_comparing(self, prob_cities, fms):
        if len(prob_cities) == 0:
            return None
        max_word = prob_cities[0]
        max_rate = 0

        if fms != None:
            w = fms.split(' ')
            # w = word[0].split(' ')
            if 'ПО' in w:
                w = w[w.index('ПО') + 1:]
            elif 'МВД' in w:
                w = w[w.index('МВД') + 1:]
            elif 'ОВД' in w:
                w = w[w.index('ОВД') + 1:]
            elif 'РОВД' in w:
                w = w[w.index('РОВД') + 1:]
            elif 'ОМВД' in w:
                w = w[w.index('ОМВД') + 1:]
            elif 'УВД' in w:
                w = w[w.index('УВД') + 1:]
            elif 'ФМС' in w:
                w = w[w.index('ФМС') + 1:]
            w = list(filter(None, map(self.filter_city, w)))
            if len(w) == 0:
                return None
            for prob_city in prob_cities:
                #for word in list(self.codes.values()):
                for corr_word in w:
                    rate = lv.jaro(prob_city, corr_word)
                    if rate > max_rate and (max(len(prob_city), len(corr_word)) / min(len(prob_city), len(corr_word))) < 1.5:
                        max_word = prob_city
                        max_rate = rate
                        self.print('max_word_city', [prob_city, corr_word])
                        #print('max_word_city: ', prob_city, corr_word)
        else:
            for prob_city in prob_cities:
                for word in list(self.codes.values()):
                    w = word[0].split(' ')
                    if 'ПО' in w:
                        w = w[w.index('ПО') + 1:]
                    elif 'МВД' in w:
                        w = w[w.index('МВД') + 1:]
                    elif 'ОВД' in w:
                        w = w[w.index('ОВД') + 1:]
                    elif 'РОВД' in w:
                        w = w[w.index('РОВД') + 1:]
                    elif 'ОМВД' in w:
                        w = w[w.index('ОМВД') + 1:]
                    elif 'УВД' in w:
                        w = w[w.index('УВД') + 1:]
                    elif 'ФМС' in w:
                        w = w[w.index('ФМС') + 1:]
                    w = list(filter(None, map(self.filter_city, w)))
                    if len(w) == 0:
                        continue
                    for corr_word in w:
                        rate = lv.jaro(prob_city, corr_word)
                        if rate > max_rate and (max(len(prob_city), len(corr_word)) / min(len(prob_city), len(corr_word))) < 1.5:
                            max_word = prob_city
                            max_rate = rate
                            self.print('max_word_city', [prob_city, corr_word])
                            #print('max_word_city: ', prob_city, corr_word)
        if max_rate <= 0.5:
            return None
        else:
            return max_word



    def __right_check(self, text, data, flag=0, places_flag = 0):
        if text.lower() == 'имя' or text.lower() == 'код':
            return ('', 0)

        priority = None
        if flag == 0:
            priority = data[:, 1]
            data = data[:, 0]
        city_ind = -1
        city_flag = False
        if places_flag>0:
            city_flag = True
            city_ind = 0
            max_rate = lv.jaro(text, data[0])
            if places_flag == 1:
                max_rate += 0.2*max_rate
        else:
            max_rate = lv.jaro(text, data[0])

        corr = data[0]

        for ind, word in enumerate(data):

            if city_flag:
                if '-' in word:
                    word = word.replace('-', '')
                word = word.lower().capitalize()
                w_copy = word
                rate = lv.jaro(text, word)
                if places_flag == 1:
                    rate += 0.2*rate
                word_check = word
            else:
                w_copy = word
                if '-' in w_copy:
                    w_copy = w_copy.replace('-', '')
                w_copy = w_copy.lower().capitalize()
                rate = lv.jaro(text, w_copy)
                word_check = word
            if rate > max_rate and (max(len(text), len(w_copy)) / min(len(text), len(w_copy))) < 1.5:
                if flag == 0:
                    if priority[ind] > 250: #and rate>0.7:
                        # if word == 'Сергей':
                        #    print('rate ', rate, 'dist: ', lv.distance(text, word), 'ratio:', lv.ratio(text, word))
                        max_rate = rate
                        corr = word_check
                        # Печатать правильно подобранное слово
                        #print('corr:', corr, text)
                    #if 150<priority[ind]<250 and rate
                else:
                    city_ind = ind
                    max_rate = rate
                    corr = word_check
                    # Печатать правильно подобранное слово
                    #print('corr:', corr, text)
        if city_flag:
            return (corr, max_rate, city_ind)
        return (corr, max_rate)

    def __check__(self, text, param):  # lv.ratio - compute similarity of two strings
        lev_dist = lv.distance(text, param[
            0])  # lv.median - делает медианное значение из нескольких вариаций распознанных слов (также median_improve, quickmedian, setmedian)
        corr = param[0]
        for i, word in enumerate(param):
            lev = lv.distance(text, word)
            if lev < lev_dist:
                lev_dist = lev
                corr = word
        if lev_dist <= 2:
            text = corr
        return text

    def __make_data(self, left_word, right_word, spec, pers_data):
        name_l = ''
        name_r = ''
        if left_word != '':
            if spec == 'имя':
                name_l = self.look_name(left_word)
            if spec == 'фамилия':
                name_l = self.look_family(left_word)
            if spec == 'отчество':
                name_l = self.look_surname(left_word)
        if right_word != '':
            if spec == 'имя':
                name_r = self.look_name(right_word)
            if spec == 'фамилия':
                name_r = self.look_family(right_word)
            if spec == 'отчество':
                name_r = self.look_surname(right_word)
        ratio_l = 0
        ratio_r = 0
        print(spec)
        if name_l != '':
            ratio_l = lv.ratio(name_l, left_word)
        if name_r != '':
            ratio_r = lv.ratio(name_r, right_word)
        print('left_word:', name_l)
        print('right_word:', name_r)
        print(pers_data)
        if ratio_l > ratio_r:
            pers_data[spec] = name_l
        else:
            pers_data[spec] = name_r
        return pers_data

    def delete_symbs(self, s1):
        new_s1 = []
        for i in range(len(s1)):
            if s1[i] in s1[:i]+s1[i+1:] and (len(s1[i]) == 2 or len(s1[i]) > 5):
                new_s1.append(s1[i])
        return new_s1

    def series_number(self, s1, pers_data):
        series = ''
        number = ''
        self.print('s1', s1)
        s1 = self.delete_symbs(s1)
        #print(s1)
        for (ind, word) in enumerate(s1):
            if len(word) == 2 and word not in series:
                if ind != 0 and ind != len(s1) - 1:
                    if len(s1[ind - 1]) >= 2 or len(s1[ind + 1]) >= 2:
                        series += word
            if len(word) >= 5 and word not in number:
                number += word
        pers_data['серия'] = series[2:] + series[:2]
        pers_data['номер'] = number
        return pers_data

    def delete_multiple_element(self, list_object, indices):
        indices = sorted(indices, reverse=True)
        for idx in indices:
            if idx < len(list_object):
                list_object.pop(idx)

    def find_code_ind(self, s1):
        self.possible_fms = None
        all_digits = [None]
        s1.append('')
        for (ind, word) in enumerate(s1):
            digit = ''
            digit_count = 0
            char_count = 0
            del_symbs = []
            alpha = []
            for k in word:
                if k.isdigit():
                    digit_count += 1
                    digit += k
                elif k.isalpha():
                    char_count += 1
                    alpha.append(k)
                else:
                    if k != '-' and k != '.':
                        del_symbs.append(k)
            if digit_count >= 5 and digit_count <= 7:
                if '.' not in word:
                    # for j in del_symbs+alpha:
                    #    word = word.replace(j, '')
                    digit = digit[0:3] + '-' + digit[3:6]
                    code_digit = digit

                    try:
                        self.print('possible fms', self.codes[code_digit])
                        #print("possible fms: ", self.codes[code_digit])
                        if len(self.codes[code_digit])>1 and len(self.codes[code_digit])<3:
                            self.possible_fms = self.codes[code_digit][0:2]
                        if len(self.codes[code_digit]) > 2:
                            self.possible_fms = self.codes[code_digit][0:3]
                        else:
                            self.possible_fms = self.codes[code_digit]
                    except:
                        return (ind, code_digit, None, None)

                    if '.' in s1[ind+1]:
                        digit = ''
                        digit_count = 0
                        char_count = 0
                        del_symbs = []
                        alpha = []
                        for k in s1[ind+1]:
                            if k.isdigit():
                                digit_count += 1
                                digit += k
                            elif k.isalpha():
                                char_count += 1
                                alpha.append(k)
                            else:
                                if k != '-' and k != '.':
                                    del_symbs.append(k)
                        if digit_count > char_count:
                            if digit_count >= 6:
                                # for j in del_symbs + alpha:
                                #    digit = digit.replace(j, '')
                                if len(word) < 8:
                                    digit = digit + '0' * (8 - len(digit))
                                elif len(digit) > 8:
                                    digit = digit[0:8]
                                digit = digit[0:2] + '.' + digit[2:4] + '.' + digit[4:8]
                                all_digits[0] = digit
                                ind+=1
                    if len(self.codes[code_digit]) > 1 and len(self.codes[code_digit]) < 3:
                        self.possible_fms = self.codes[code_digit][0:2]
                    if len(self.codes[code_digit]) > 2:
                        self.possible_fms = self.codes[code_digit][0:3]
                    if len(self.codes[code_digit]) == 1:
                        self.possible_fms = self.codes[code_digit]
                    return (ind, code_digit, self.possible_fms, all_digits[0])
            else:
                if digit_count > char_count:
                    if digit_count >= 6:
                        # for j in del_symbs + alpha:
                        #    digit = digit.replace(j, '')
                        if len(word) < 8:
                            digit = digit + '0' * (8 - len(digit))
                        elif len(digit) > 8:
                            digit = digit[0:8]
                        digit = digit[0:2] + '.' + digit[2:4] + '.' + digit[4:8]
                        all_digits[0] = digit
        return (None, None, None, None)
        """for j in del_symbs:
                    word = word.replace(j, '')
                all_digits.append(word)
        print('Digits checking: ', all_digits)
        input()"""

    def cpr_spec(self, s1):
        self.print('After adding', s1)
        #print("After adding: ", s1)
        if self.possible_fms == None:
            return None

        ratio = [0] * len(self.possible_fms)
        for i in range(len(self.possible_fms)):
            cp = self.possible_fms[i].split(' ')[0:4]
            max_rat = 0
            for j in range(len(s1)):
                rat = 0
                ind_max = 0
                for z in range(len(cp)):
                    r = lv.ratio(s1[j], cp[z])
                    if r > rat:
                        ind_max = z
                        rat = r
                del cp[ind_max]
                ratio[i] += rat
            self.possible_fms[i] = [self.possible_fms[i], ratio[i]]

        self.possible_fms = sorted(self.possible_fms, key=lambda x: -x[1])
        for i in range(len(self.possible_fms)):
            self.possible_fms[i] = self.possible_fms[i][0]
        return self.possible_fms
        '''m = max(ratio)
        for i, val in enumerate(ratio):
            if m == val:
                possible_fms = [self.possible_fms[i]]
                del self.possible_fms[i]
                self.possible_fms = possible_fms+self.possible_fms
                n = min(3, len(self.possible_fms))
                return self.possible_fms[:n]'''

    def cpr_spec_fms(self, s1, pers_data):
        #self.print('After adding', s1)
        #print("After adding: ", s1)

        all_digits = []
        # all_words = []
        for (ind, word) in enumerate(s1):
            digit_count = 0
            digits = ''
            # words = ''
            for k in word:
                #self.print('digit checking', word)
                #print('digit checking:', word)
                if k.isdigit():
                    digit_count += 1
                    digits += k

            if digit_count > 0:
                all_digits.append(digits)
            else:
                pass
                """word = word.replace('.', '')
                word = word.replace('-', '')
                word = word.replace(',', '')
                word = word.replace(':', '')
                all_words.append(word.lower().capitalize())"""
        # print('all_digits', all_digits)
        for digit in all_digits:
            if len(digit) > 6 and '-' not in digit:
                if digit[0:2] > '31':
                    digit = "30"+digit[2:len(digit)]
                if digit[2:4] > "12":
                    digit = digit[0:2] + "05"+digit[4:len(digit)]
                if digit[4:8] > "2022":
                    digit = digit[0:4] + "2022"
                pers_data["дата выдачи"] = digit[0:2] + '.' + digit[2:4] + '.' + digit[4:8]
            elif 5 <= len(digit) < 7:
                if len(digit) > 6:
                    digit = digit[:6]
                elif len(digit) < 6:
                    digit += '0' * (6 - len(digit))
                code = digit[0:3] + '-' + digit[3:6]
                pers_data['Код подразделения'] = code
                #print('ФМС: ', self.codes[code])
                pers_data['фмс'] = self.codes[code]
            else:
                pass

        return pers_data


    def cpr_fio(self, words, dicts, ind):
        arr = []
        flag = 1
        #priority = dicts[:,1].copy()
        #dicts = dicts[:,0].copy()
        if ind == 1 or ind == 2:
            flag = 0
        for z in range(3):
            word2, rate = self.__right_check(words[ind], dicts, flag=flag)
            arr.append([word2, words[ind], ind, rate])
            #if ind == 1:
                #priority = priority[np.where(dicts != word2)]
            if ind != 0:
                dicts = dicts[np.where(dicts[:,0] != word2)[0]]
            else:
                dicts = dicts[np.where(dicts != word2)]
            #temp_arr = np.array([word2])
            #dicts = np.setdiff1d(dicts, temp_arr)
        three_arrs = self.sort_data(arr)
        return three_arrs

    def cpr_obls(self, word, data, len_places, places_flag):
        places = []
        obl_flag = True
        for z in range(3):
            word_corr, rate, city_ind = self.__right_check(word, data, flag=1, places_flag=places_flag)
            while [word_corr, word, True, rate] in places or [word_corr, word, False, rate] in places:
                if city_ind > len_places[0]:
                    len_places[1] -=1
                else:
                    len_places[0] -= 1
                data = np.delete(data, city_ind, axis=0)
                word_corr, rate, city_ind = self.__right_check(word, data, flag=1, places_flag=places_flag)
            if city_ind > len_places[0]:
                obl_flag = False
            places.append([word_corr, word, obl_flag, rate])
            obl_flag = True
        three_cities = self.sort_data(places)
        places.clear()
        return three_cities

    def cpr_cities(self, word, data, places_flag):
        cities = []
        for z in range(3):
            word_corr, rate, city_ind = self.__right_check(word, data, flag=1, places_flag=places_flag)
            while [word_corr, word, 0, rate] in cities:
                data = np.delete(data, city_ind, axis=0)
                word_corr, rate, city_ind = self.__right_check(word, data, flag=1, places_flag=places_flag)
            cities.append([word_corr, word, 0, rate])
        three_cities = self.sort_data(cities)
        cities.clear()
        return three_cities

    def cpr_spec_words(self, s1, pers_data):
        self.data_fms = pers_data['фмс']
        self.print('s1', s1)
        # print('s1:', s1)
        all_digits = []
        t1 = time.time()

        all_words = [word.lower().capitalize() for word in s1]

        three_names = self.cpr_fio(all_words, self.names_surnames[:, [0,2]].copy(), 1)

        three_families = self.cpr_fio(all_words, self.families.copy(), 0)

        three_surnames = self.cpr_fio(all_words, self.names_surnames[:, [1, 2]].copy(), 2)
        all_words = all_words[3:]
        new_words = []
        first_ind = -1
        for (ind, word) in enumerate(all_words):
            digit_count = 0
            digits = ''
            for k in word:
                if k.isdigit():
                    digit_count += 1
                    digits += k

            if digit_count > 0:
                if len(digits) > 7:
                    first_ind = ind
                all_digits.append(digits)
            else:
                #if first_ind != -1:
                new_words.append(word)
        all_words = new_words
        for digit in all_digits:
            if len(digit) > 6 and '-' not in digit:
                if digit[0:2] > '31':
                    digit = "30"+digit[2:len(digit)]
                if digit[2:4] > "12":
                    digit = digit[0:2] + "05"+digit[4:len(digit)]
                year = digit[4:8]
                fl = False
                if int(year) > 2006:
                    for j in range(1, len(year)):
                        for d in range(int(year[j]), -1 , -1):
                            year1 = int(year[0:j]+str(d)+year[j+1:])
                            if year1 >=1950 and year1 <= 2006:
                                year = str(year1)
                                fl = True
                                break
                        if fl:
                            break
                        year = year[0:j] + '0' + year[j + 1:]

                elif int(year) < 1950:
                    for j in range(1, len(year)):
                        for d in range(int(year[j]), 10):
                            year1 = int(year[0:j] + str(d) + year[j + 1:])
                            if year1 >= 1950 and year1 <= 2006:
                                year = str(year1)
                                fl = True
                                break
                        if fl:
                            break
                        year = year[0:j] + '9' + year[j + 1:]
                pers_data["дата рождения"] = digit[0:2] + '.' + digit[2:4] + '.' + year
            elif 5 <= len(digit) < 7:
                if len(digit) > 6:
                    digit = digit[:6]
                elif len(digit) < 6:
                    digit += '0' * (6 - len(digit))
            else:
                pass

        pers_data['пол'] = 'муж'
        pers_data['фамилия'] = three_families
        pers_data['имя'] = three_names
        pers_data['отчество'] = three_surnames

        cities = []
        three_cities = None
        three_obls = None
        three_krai = None
        if len(all_words) == 1:
            if ' ' not in all_words[0]:
                three_cities = self.cpr_cities(all_words[0], self.cities.copy(), places_flag = 1)
                """for z in range(3):
                    word_corr, rate, city_ind = self.__right_check(all_words[0], self.cities[:, 0], flag=1, places_flag = 1)
                    while [word_corr, all_words[0], 0, rate] in cities:
                        self.cities = np.delete(self.cities, city_ind, axis = 0)
                        word_corr, rate, city_ind = self.__right_check(all_words[0], self.cities[:, 0], flag=1, places_flag = 1)
                    cities.append([word_corr, all_words[0], 0, rate])
                three_cities = self.sort_data(cities)
                cities.clear()"""
            else:
                all_words = all_words[0].split(' ')
        three_places = None
        if len(all_words) != 1:
            for i in range(len(all_words)-1, -1 , -1):
                word = all_words[i].lower()
                rate_krai = lv.ratio(word, 'край')
                if rate_krai > 0.8:
                    krais = list(self.krai.keys())
                    three_krai = self.cpr_obls(all_words[i-1], list(self.krai.keys()).copy(), [0, len(krais)], places_flag=2)

                rate_obl = max(lv.ratio(word, 'обл.'), lv.ratio(word, 'область'))
                if rate_obl > 0.8:
                    obls = list(self.obl.keys())
                    three_obls = self.cpr_obls(all_words[i-1], list(self.obl.keys()).copy(), [len(obls), 0], places_flag=3)


                rate_city = max(lv.ratio(word, 'город'), lv.ratio(word, 'гор.'))
                if rate_city > 0.8:
                    three_cities = self.cpr_cities(all_words[i+1], self.cities.copy(), places_flag=1)

                rate_der = max(lv.ratio(word, 'деревня'), lv.ratio(word, 'дер.'))
                if rate_der > 0.8:
                    pass
                rate_selo = max(lv.ratio(word, 'село'), lv.ratio(word, 'с.'))
                if rate_selo > 0.8:
                    pass
                rate_posel = max(lv.ratio(word, 'посёлок'), lv.ratio(word, 'пос.'))
                if rate_posel > 0.8:
                    pass
                rate_hutor = max(lv.ratio(word, 'хутор'), lv.ratio(word, 'хут.'))
                if rate_hutor > 0.8:
                    pass

            if three_cities == None:
                new_list_cities = np.array([None])
                if three_obls == None and three_krai == None:
                    for i in range(len(all_words)-1 , 0, -1):
                        all_obls = list(self.obl.keys())+list(self.krai.keys())
                        len_obls = [len(list(self.obl.keys())), len(list(self.krai.keys()))]
                        three_new_places = self.cpr_obls(all_words[i], all_obls.copy(), len_obls, places_flag=2)
                        if three_places == None:
                            three_places = three_new_places
                        else:
                            three_places = three_places + three_new_places
                            three_places.sort(key=lambda x: x[3])
                            three_places.reverse()
                            three_places = three_places[:3]

                        for w in three_places:
                            if w[2] == True:
                                self.obl[w[0]] = self.obl[w[0]].reshape(self.obl[w[0]].shape[0], 1)
                                new_list_cities = np.append(new_list_cities, self.obl[w[0]])
                            else:
                                self.krai[w[0]] = self.krai[w[0]].reshape(self.krai[w[0]].shape[0], 1)
                                new_list_cities = np.append(new_list_cities, self.krai[w[0]])
                        new_list_cities = np.delete(new_list_cities, 0)
                    #three_cities = self.cpr_cities(all_words[0], new_list_cities.copy(), places_flag=1)
                elif three_obls != None:
                    for w in three_obls:
                        if w[2] == True:
                            self.obl[w[0]] = self.obl[w[0]].reshape(self.obl[w[0]].shape[0], 1)
                            new_list_cities = np.append(new_list_cities, self.obl[w[0]])
                        else:
                            self.krai[w[0]] = self.krai[w[0]].reshape(self.krai[w[0]].shape[0], 1)
                            new_list_cities = np.append(new_list_cities, self.krai[w[0]])
                    new_list_cities = np.delete(new_list_cities, 0)
                    three_places = three_obls
                elif three_krai != None:
                    for w in three_krai:
                        if w[2] == True:
                            self.obl[w[0]] = self.obl[w[0]].reshape(self.obl[w[0]].shape[0], 1)
                            new_list_cities = np.append(new_list_cities, self.obl[w[0]])
                        else:
                            self.krai[w[0]] = self.krai[w[0]].reshape(self.krai[w[0]].shape[0], 1)
                            new_list_cities = np.append(new_list_cities, self.krai[w[0]])
                    new_list_cities = np.delete(new_list_cities, 0)
                    three_places = three_krai
                three_cities = self.cpr_cities(all_words[0], new_list_cities.copy(), places_flag=1)

                # Доделать для областей и краев
                #nas_

        if three_places == None:
            pers_data['место'] = three_cities
        else:
            pers_data['место'] = three_cities+three_places
        return pers_data

    def cpr_spec_words1(self, s1, pers_data):
        """j = 0
        l = len(s1)
        while j<l:
            if len(s1[j]) < 3:
                s1[j] +=s1[j+1]
                del s1[j+1]
                l = len(s1)
            else:
                j+=1"""
        # print("After adding: ", s1)
        self.data_fms = pers_data['фмс']
        self.print('s1', s1)
        #print('s1:', s1)
        all_digits = []
        all_words = []
        first_ind = 0
        t1 = time.time()

        all_words = [word.lower().capitalize() for word in s1]
        names = []
        for z in range(3):
            word2, rate = self.__right_check(all_words[1], self.names, flag=0)
            names.append([word2, all_words[1], 1, rate])
            temp_arr = np.array([word2])
            self.names = np.setdiff1d(self.names, temp_arr)
        three_names = self.sort_data(names)
        names.clear()

        families = []
        for z in range(3):
            fam, rate = self.__right_check(all_words[0], self.families, flag=1)
            families.append([fam, all_words[0], 0, rate])
            temp_arr = np.array([fam])
            self.families = np.setdiff1d(self.families, temp_arr)
        three_families = self.sort_data(families)
        families.clear()

        surnames = []
        for z in range(3):
            sur, rate = self.__right_check(all_words[2], self.surnames, flag=0)
            surnames.append([sur, all_words[2], 2, rate])
            temp_arr = np.array([sur])
            self.surnames = np.setdiff1d(self.surnames, temp_arr)
        three_surnames = self.sort_data(surnames)
        surnames.clear()

        all_words = all_words[3:]
        for (ind, word) in enumerate(all_words):
            digit_count = 0
            digits = ''
            for k in word:
                # print('digit checking:', word)
                if k.isdigit():

                    digit_count += 1
                    digits += k

            if digit_count > 0:
                if len(digits) > 7:
                    first_ind = ind
                all_digits.append(digits)
                all_words = all_words[first_ind+1:]
                break

        for (ind, word) in enumerate(s1):
            digit_count = 0
            digits = ''
            words = ''
            for k in word:
                # print('digit checking:', word)
                if k.isdigit():

                    digit_count += 1
                    digits += k

            if digit_count > 0:
                if len(digits) >7:
                    first_ind = ind
                all_digits.append(digits)
            else:
                word = word.replace('.', '')
                word = word.replace('-', '')
                word = word.replace(',', '')
                word = word.replace(':', '')
                all_words.append(word.lower().capitalize())


        # spec_word = ''
        corr_name, family, surname, prev_word = '', '', '', ''
        max_rate, ind_name = 0, 0
        rate_fam, rate_sur = 0, 0
        ind_sur, ind_fam, fl = 0, 0, 0

        # Поиск имени и отчества
        three_names = [[None, None, -1] for i in range(3)]
        three_surnames = three_names.copy()
        three_cities = three_names.copy()
        three_families = three_names.copy()

        names = []
        nams = self.names.copy()
        if first_ind != 0:
            first_ind = max(first_ind, 1)
        else:
            first_ind = len(s1)
        for (ind, word) in enumerate(all_words[:first_ind]):
            self.names = nams.copy()
            for z in range(3):
                word2, rate = self.__right_check(word, self.names, flag=0)
                names.append([word2, word, ind, rate])
                temp_arr = np.array([word2])
                self.names = np.setdiff1d(self.names, temp_arr)
        self.names = nams.copy()
        del nams


        #temp_arr = np.array([word2])
        #self.names = np.setdiff1d(self.names, temp_arr)

        if rate > max_rate:
                max_rate = rate
                corr_name = word2
                three_names[2] = three_names[1]
                three_names[1] = three_names[0]
                three_names[0] = [corr_name, word, ind]
                prev_word = word
                ind_name = ind
        three_names = self.sort_data(names)
        names.clear()
        three_surnames = [[None, None, 0] for i in range(3)]
        three_families = None
        # print("Chosen name: ", corr_name)
        # print(prev_word[-3:].lower())
        # print('ratio вич: ', self.ratio(prev_word[-3:].lower(), 'вич'))
        if self.ratio(three_names[0][1][-3:].lower(), 'вич') > 0.6:
            corr_name, ratio = self.__right_check(three_names[0][1], self.surnames, flag=0)
            three_names[0] = [None, None,0]
            three_surnames[0] = [corr_name, three_names[0][1], three_names[0][2]]
            #pers_data['отчество'] = corr_name
            #ind_sur = ind_name
            corr_name = ''
            ind_name = 0
            fl = 1
        else:
            pass

                #pers_data['имя'] = corr_name
        #print("Correct_name: ", corr_name, max_rate)
        #pers_data['имя'] = corr_name

        # Поиск фамилии и отчества, если найдено имя
        families = []
        if three_names[0][0] != None:
                fams = self.families.copy()
                '''if three_names[0][2] == 0:
                    three_names[0][2] = 1'''
                for i in range(max(0, three_names[0][2] - 3), three_names[0][2]):
                    self.families = fams.copy()
                    for z in range(3):
                        fam, rate = self.__right_check(all_words[i], self.families, flag=1)
                        families.append([fam, all_words[i], i, rate])
                        temp_arr = np.array([fam])
                        self.families = np.setdiff1d(self.families, temp_arr)
                del fams
                '''if rate > rate_fam:
                        rate_fam = rate
                        three_families[2] = three_families[1]
                        three_families[1] = three_families[0]
                        three_families[0] = [fam, all_words[i], ind_fam]
                        family = fam
                        ind_fam = i'''
                three_families = self.sort_data(families)
                families.clear()
                #pers_data['фамилия'] = three_families
                    #pers_data['фамилия'] = family

                surnames = []
                surs = self.surnames.copy()
                '''if  min(three_names[0][2] + 3, first_ind) == three_names[0][2] + 1:
                    three_names[0][2] -=1'''
                for i in range(three_names[0][2] + 1, min(three_names[0][2] + 3, first_ind)):
                    self.surnames = surs.copy()
                    for z in range(3):
                        sur, rate = self.__right_check(all_words[i], self.surnames, flag=0)
                        surnames.append([sur, all_words[i], i, rate])
                        temp_arr = np.array([sur])
                        temp_arr = np.array([sur])
                        self.surnames = np.setdiff1d(self.surnames, temp_arr)

                del surs
                three_surnames = self.sort_data(surnames)
                surnames.clear()
                '''if rate > rate_sur:
                    rate_sur = rate
                    surname = sur
                    ind_sur = i
                pers_data['отчество'] = surname'''
                #pers_data['отчество'] = three_surnames
                # Поиск фамилии и имени, если найдено отчество
        else:
                fams = self.families.copy()
                '''if three_surnames[0][2] == 0:
                    three_surnames[0][2] = 1'''
                for i in range(max(0, three_surnames[0][2] - 3), three_surnames[0][2]):
                    self.families = fams.copy()
                    for z in range(3):
                        fam, rate = self.__right_check(all_words[i], self.families, flag=1)
                        families.append([fam, all_words[i], i, rate])
                        temp_arr = np.array([fam])
                        self.families = np.setdiff1d(self.families, temp_arr)

                del fams
                three_families = self.sort_data(families)
                families.clear()

                '''fam, rate = self.__right_check(all_words[i], self.families, flag=1)
                    if rate > rate_fam:
                        rate_fam = rate
                        family = fam
                        ind_fam = i
                pers_data['фамилия'] = family'''
                #pers_data['фамилия'] = three_families
                #max_rate = 0

                t = 2
                '''if three_families[0][2] == three_surnames[0][2]-1:
                    t = 0
                    three_surnames[0][2] = 4'''
                surs = self.surnames.copy()
                for i in range(max(0, three_surnames[0][2] - 4), three_surnames[0][2]-t):
                    self.surnames = surs.copy()
                    for z in range(3):
                        word, rate = self.__right_check(all_words[i], self.names, flag=0)
                        names.append([word, all_words[i], i, rate])
                        temp_arr = np.array([word])
                        self.names = np.setdiff1d(self.names, temp_arr)

                del surs
                three_names = self.sort_data(names)
                names.clear()
                '''if rate > max_rate:
                        max_rate = rate
                        corr_name = word
                        ind_name = i
                pers_data['имя'] = corr_name'''
                #pers_data['имя'] = three_names
        #self.print('surname', all_words[ind_sur])
        #print("surname: ", all_words[ind_sur])
        three_cities = [[None, None, 0] for i in range(3)]
        delete_indexes = []
        for j in range(len(three_surnames)):
            ind_sur= s1.index(all_words[three_surnames[j][2]].upper())
            three_surnames[j][2] = ind_sur
            city_index = ind_sur
            if first_ind != len(s1)-1:
                city_index = first_ind
            three_cities[j][2] = city_index
            city_word = s1[city_index+1:]
            three_cities[j][1] = city_word
            delete_indexes += [three_surnames[j][2], three_names[j][2], three_families[j][2]]


        delete_indexes = list(set(delete_indexes))
        self.delete_multiple_element(all_words, delete_indexes)

        max_rate_city, max_rate_reg, date_digit = 0,0,0
        corr_city, corr_region = '', ''
        print("Time for finding name and surname in BD is ", time.time()-t1)

        # Поиск даты рождения среди всех чисел
        for digit in all_digits:
            if len(digit) > 6 and '-' not in digit:
                if digit[0:2] > '31':
                    date_digit = digit[0:2]+'.'
                    digit = "30"+digit[2:len(digit)]
                if digit[2:4] > "12":
                    date_digit += digit[2:4]+'.'
                    digit = digit[0:2] + "05"+digit[4:len(digit)]
                year = digit[4:8]
                fl = False
                if int(year) > 2006:
                    for j in range(1, len(year)):
                        for d in range(int(year[j]), -1 , -1):
                            year1 = int(year[0:j]+str(d)+year[j+1:])
                            if year1 >=1950 and year1 <= 2006:
                                year = str(year1)
                                fl = True
                                break
                        if fl:
                            break
                        year = year[0:j] + '0' + year[j + 1:]

                elif int(year) < 1950:
                    for j in range(1, len(year)):
                        for d in range(int(year[j]), 10):
                            year1 = int(year[0:j] + str(d) + year[j + 1:])
                            if year1 >= 1950 and year1 <= 2006:
                                year = str(year1)
                                fl = True
                                break
                        if fl:
                            break
                        year = year[0:j] + '9' + year[j + 1:]
                #date_digit += year
                pers_data["дата рождения"] = digit[0:2] + '.' + digit[2:4] + '.' + year
            elif 5 <= len(digit) < 7:
                if len(digit) > 6:
                    digit = digit[:6]
                elif len(digit) < 6:
                    digit += '0' * (6 - len(digit))
                # code = digit[0:3]+'-'+digit[3:6]
                # pers_data['Код подразделения'] = code
                # pers_data['фмс'] = self.codes[code]
            else:
                pass

        pers_data['пол'] = 'муж'
        pers_data['фамилия'] = three_families
        pers_data['имя'] = three_names
        pers_data['отчество'] = three_surnames
        # Поиск города и региона рождения
        ind_dig = 0
        #self.print('city_word_testing', city_word)
        #print('city_word_testing: ', city_word)
        #self.print('s1 testing', s1)
        #print('s1 testing: ',s1)
        #self.print('pers_data', pers_data["дата рождения"])
        #print('pers_data: ', pers_data["дата рождения"])
        #if date_digit in all_words:
        #    ind_dig = all_words.index(date_digit)
        #self.print('All words', all_words)
        #self.print('date digit', date_digit)
        #self.print('Probable city place', all_words[ind_dig:])
        #self.print('cities', self.arr_name)
        #print('All words: ', all_words)
        #print('date digit: ', date_digit)
        #print('Probable city place: ', all_words[ind_dig:])
        #print('cities: ', self.arr_name)
        cities = []
        cits = self.arr_name.copy()
        for j in range(len(three_cities)):
            cities = []
            for (ind, word) in enumerate(three_cities[j][1]):
                #if '.' in
                for symb in '.,:- ':
                    if symb in word:
                        word = word.replace(symb, '')
                word = word.lower().capitalize()
                word1 = word
                self.arr_name = cits.copy()
                for z in range(3):
                    word_corr, rate,city_ind = self.__right_check(word, self.arr_name, flag=1)
                    while [word_corr, word, ind, rate] in cities:
                        self.arr_name = np.delete(self.arr_name, city_ind)
                        #temp_arr = np.array([word_corr])
                        #print(temp_arr)
                        #self.arr_name = np.setdiff1d(self.arr_name, temp_arr)
                        #print(self.arr_name.shape)
                        word_corr, rate, city_ind = self.__right_check(word, self.arr_name, flag=1)
                    temp_arr = np.array([word_corr])
                    self.arr_name = np.setdiff1d(self.arr_name, temp_arr)
                    cities.append([word_corr, word, ind, rate])

        three_cities = self.sort_data(cities)
        cities.clear()
        del cits

        '''if rate > max_rate_city:
                    print('City: ', word, 'corr word: ', word_corr)
                    max_rate_city = rate
                    corr_city = word_corr
                    cities.append(corr_city)
                    ind_city = ind'''
            #word_corr, rate = self.__right_check(word1, self.arr_region, flag=1)
            #if rate > max_rate_reg:
            #    print('Region: ', word1, 'corr word: ', word_corr)
            #    max_rate_reg = rate
            #    corr_region = word_corr
            #    ind_reg = ind
        # print("Correct_city: ", corr_city, max_rate_city)
        # print("Correct_region: ", corr_region, max_rate_reg)
        #if max_rate_reg < 0.6:
        #    corr_region = '-'
        '''city = self.city_comparing(cities, pers_data['фмс'])
        if city != None:
            corr_city = city
        if max_rate_city < 0.6:
            corr_city = '-' '''
        pers_data['место'] = three_cities #+ ' ' + corr_region

        """for (ind_spec, spec) in enumerate(self.special_words):
            max_ratio = 0
            max_ind = 0
            for (ind,word) in enumerate(all_words):
                if len(spec)-1<=len(word)<=len(spec)+1:
                    ratio = lv.ratio(word.lower(), spec)
                    if ratio > max_ratio:
                        max_ratio = ratio
                        max_ind = ind
                        spec_word = spec

            all_words[max_ind] = spec_word

            left_word = ''
            right_word = ''
            if max_ind != 0:
                left_word = all_words[max_ind-1]
            if max_ind != len(all_words)-1:
                right_word = all_words[max_ind+1]

            if all_words[max_ind].lower()  == 'имя':
                print(all_words[max_ind])
                print(left_word)
                print(right_word)
                pers_data = self.__make_data(left_word, right_word, 'имя', pers_data)
            if all_words[max_ind].lower() == 'фамилия':
                print(all_words[max_ind])
                print(left_word)
                print(right_word)
                pers_data = self.__make_data(left_word, right_word, 'фамилия', pers_data)
            if all_words[max_ind].lower() == 'отчество':
                print(all_words[max_ind])
                print(left_word)
                print(right_word)
                pers_data = self.__make_data(left_word, right_word, 'отчество', pers_data)

            #if all_words[max_ind].lower() == 'пол':
            #    pers_data['пол'] = 'муж'
            #pers_data['пол'] = 'муж'
            if all_words[max_ind].lower() == 'обл':
                pers_data['место'] = 'обл'+pers_data['место']
                left = max_ind - 2
                if left < 0:
                    left = 0
                right = max_ind +3
                if right >= len(all_words):
                    right = len(all_words)


            if all_words[max_ind].lower() == 'гор':
                pers_data['место'] = 'гор'+pers_data['место']
            if all_words[max_ind].lower() == 'пос':
                pers_data['место'] = 'пос'+pers_data['место']
        """

        """
                for spec in self.special_words:
                        ratio = lv.ratio(word.lower(), spec)
                        print('word: ', word.lower(), ratio)
                        if ratio >= 0.7:
                            print(word, spec)
                            s1[ind] = spec
                left_word = ''
                right_word = ''
                if ind != 0:
                    left_word = s1[ind - 1]
                if ind != len(s1) - 1:
                    right_word = s1[ind + 1]
                if s1[ind].lower()  == 'имя':
                    if left_word != '':
                        s1[ind-1] = self.look_name(left_word, 'имя')
                    if right_word != '':
                        s1[ind+1] = self.look_name(right_word, 'имя')

                elif s1[ind].lower() == 'фамилия':
                    if left_word != '':
                        s1[ind-1] = self.look_family(left_word, 'фамилия')
                    if right_word != '':
                        s1[ind+1] = self.look_family(right_word, 'фамилия')

                elif s1[ind].lower() == 'отчество':
                    if left_word != '':
                        s1[ind-1] = self.look_surname(left_word, 'отчество')
                    if right_word != '':
                        s1[ind+1] = self.look_surname(right_word, 'отчество')

                elif s1[ind].lower() == 'пол':
                    if left_word != '':
                        if lv.distance(left_word, 'муж.') <= 2:
                            s1[ind-1] = 'муж.'
                    if right_word != '':
                        if lv.distance(right_word, 'муж.') <= 2:
                            s1[ind+1] = 'муж.'
        """
        # print('all_digits', all_digits)
        """for digit in all_digits:
            if len(digit) > 6 and '-' not in digit:
                if digit[0:2] > '31':
                    digit = "30"+digit[2:len(digit)]
                if digit[2:4] > "12":
                    digit = digit[0:2] + "05"+digit[4:len(digit)]
                if digit[4:8] > "2022":
                    digit = digit[0:4] + "2012"
                pers_data["дата рождения"] = digit[0:2] + '.' + digit[2:4] + '.' + digit[4:8]"""
        """for dig in all_digits:
                    if len(dig) >6 and '-' not in dig and dig != digit:

                        if len(digit)> 8:
                            digit = digit[:8]
                        elif len(digit) < 8:
                            digit += '0' * (8 - len(digit))
                        if len(dig) > 8:
                            dig = dig[:8]
                        elif len(dig) < 8:
                            dig += '0' * (8 - len(dig))
                        new_dig1 = digit[4:8]
                        new_dig2 = dig[4:8]
                        if new_dig1 < new_dig2:
                            d1 = digit
                            d2 = dig
                        else:
                            d1 = dig
                            d2 = digit

                        pers_data["дата рождения"] = d1[0:2]+'.'+d1[2:4]+'.'+d1[4:8]
                        pers_data["дата выдачи"] = d2[0:2]+'.'+d2[2:4]+'.'+d2[4:8]

                        break"""
        """elif 5 <= len(digit) < 7:
                if len(digit) > 6:
                    digit = digit[:6]
                elif len(digit) < 6:
                    digit += '0' * (6 - len(digit))
                # code = digit[0:3]+'-'+digit[3:6]
                # pers_data['Код подразделения'] = code
                # pers_data['фмс'] = self.codes[code]
            else:
                pass"""
                # pers_data['фмс'] += digit

        """elif 'код' in s1[ind].lower():
                for k in range(ind-4, ind+4):
                    if s1[k][0].isdigit() and '-' in s1[k]:
                        self.person_data['код'] = s1[k]
                        break
            elif 'дата' in s1[ind].lower():
                for k in range(ind-4, ind+4):
                    if s1[k][0].isdigit() and '.' in s1[k]:
                        self.person_data["дата рождения"] = s1[k]
                        break"""

        return pers_data

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

    def print(self, about, string):
        pass
        #print(f"{about}: {string}")

    def median(self, sequence):
        return lv.quickmedian(sequence)

    def median_improve(self, word, sequence):
        return lv.median_improve(word, sequence)

    def quickmedian(self, sequence):
        return lv.quickmedian(sequence)

    def setmedian(self, sequence):
        return lv.setmedian(sequence)

    def seqratio(self, s1, s2):
        return lv.seqratio(s1, s2)

    def ratio(self, s1, s2):
        s = SequenceMatcher(None, s1, s2)
        # return s.ratio()
        return lv.ratio(s1, s2)

    def opcodes(self, s1, s2):
        return len(lv.opcodes(s1, s2))

    def look_name(self, name):
        return self.__check__(name, self.names)

    def look_surname(self, surname):
        return self.__check__(surname, self.surnames)

    def look_family(self, family):
        return self.__check__(family, self.families)

    def look_spec_words(self, text):
        return self.__check__(text, self.special_words)
        # ham_dist = lv.hamming(name, 'ham')
