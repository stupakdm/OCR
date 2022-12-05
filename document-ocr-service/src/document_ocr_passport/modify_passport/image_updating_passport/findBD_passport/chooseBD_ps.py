#from difflib import SequenceMatcher

import numpy as np
import pandas as pd
import time


from modify_passport.image_updating_passport.findBD_passport.data_passport.models import Namessurnames, Families, FMSUnit, Cities, get_item

from modify_passport.image_updating_passport.findBD_passport.useful_functions_ps import BD_Functions


class Search:
    '''
    Класс функций для сопоставления распознанных слов с БД
    и формирования итоговой информации
    '''
    special_words = ('дата', 'рождения', 'место', 'гор', 'код', 'обл', 'муж', 'пос', 'пол', 'россия', 'федерация')
    # 'имя', 'фамилия', 'отчество'
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

    def start(self):
        # self.name_filepath = 'templates/Names and surnames.csv'
        # self.fams_filepath = 'templates/families2.csv'
        curr_dir = 'modify/Image_update/findBD/data_passport/'
        #self.df_name = pd.read_csv(f'{curr_dir}Names_and_surnames2.csv', encoding='utf-8')
        df_name = get_item(Namessurnames)
        df_name = pd.DataFrame.from_dict(df_name)
        self.names_surnames = np.array(df_name)
        self.useful_functions = BD_Functions()

        self.pers_data = {
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
        #self.names_surnames = np.array([self.df_name['name'], self.df_name['surname'], self.df_name['priority']])

        self.names_surnames = self.names_surnames[np.where(self.names_surnames[:,2] >150)[0]]

        df_families = get_item(Families)
        df_families = pd.DataFrame.from_dict(df_families)
        self.families = np.array(df_families['family'])

        #self.df_cities = pd.read_csv(f'{curr_dir}RussianCities.tsv', sep='\t', index_col=False, encoding='utf-8')
        df_cities = get_item(Cities)
        df_cities = pd.DataFrame.from_dict(df_cities)
        self.df_places = np.array(df_cities[['name', 'regionname']])
        #self.df_places = np.array([self.df_places['name'], self.df_places['region_name']])
        self.cities = None #self.find_something(self.df_places, 'город')
        self.derevni = None #self.find_something(self.df_places, 'деревня')
        self.sela = None #self.find_something(self.df_places, 'село')
        self.poselki = None# self.find_something(self.df_places, 'посёлок')
        self.hutor = None #self.find_something(self.df_places, 'хутор')
        self.find_place(self.df_places)

        arr_name = np.array(df_cities['name'])
        arr_region = np.array(df_cities['regionname'])
        arr_region = np.unique(arr_region)
        self.obl, self.krai = self.useful_functions.find_regions(arr_region, self.df_places)
        self.arr_region = np.array(list(map(self.useful_functions.region_filter, arr_region)))
        self.arr_name = np.array(list(filter(self.useful_functions.filter_names, map(self.useful_functions.name_filter, arr_name))))


        self.df_fms = get_item(FMSUnit)
        self.df_fms = pd.DataFrame.from_dict(self.df_fms)
        #self.df_fms = pd.read_csv(f'{curr_dir}fms_unit.csv', sep=',', index_col=False, encoding='utf-8')
        self.codes = {}
        places = np.array(self.df_fms['name'])
        code = np.array(self.df_fms['code'])
        for i in range(len(code)):
            if code[i] not in self.codes.keys():
                self.codes[code[i]] = [places[i], ]
            else:
                self.codes[code[i]].append(places[i])

        self.possible_fms = None

        #del self.df_cities
        #del self.df_families
        #del self.df_name
        del self.df_fms

    def find_place(self, arrays):
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

            max_rate = self.useful_functions.ratio(text, data[0])
            if places_flag == 1:
                max_rate += 0.2*max_rate
        else:
            max_rate = self.useful_functions.ratio(text, data[0])

        corr = data[0]

        for ind, word in enumerate(data):

            if city_flag:
                if '-' in word:
                    word = word.replace('-', '')
                word = word.lower().capitalize()
                w_copy = word
                rate = self.useful_functions.ratio(text, word)
                if places_flag == 1:
                    rate += 0.2*rate
                word_check = word
            else:
                w_copy = word
                if '-' in w_copy:
                    w_copy = w_copy.replace('-', '')
                w_copy = w_copy.lower().capitalize()
                rate = self.useful_functions.ratio(text, w_copy)
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


    def series_number(self, s1, pers_data):
        series = ''
        number = ''
        #self.useful_functions.print('s1', s1)
        s1 = self.useful_functions.delete_symbs(s1)
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
                        #self.useful_functions.print('possible fms', self.codes[code_digit])
                        #print("possible fms: ", self.codes[code_digit])
                        '''if len(self.codes[code_digit])>1 and len(self.codes[code_digit])<3:
                            self.possible_fms = self.codes[code_digit][0:2]
                        if len(self.codes[code_digit]) > 2:
                            self.possible_fms = self.codes[code_digit][0:3]
                        else:'''
                        self.possible_fms = self.codes[code_digit][0]
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
                    '''if len(self.codes[code_digit]) > 1 and len(self.codes[code_digit]) < 3:
                        self.possible_fms = self.codes[code_digit][0:2]
                    if len(self.codes[code_digit]) > 2:
                        self.possible_fms = self.codes[code_digit][0:3]
                    if len(self.codes[code_digit]) == 1:
                        self.possible_fms = self.codes[code_digit]'''
                    self.possible_fms = self.codes[code_digit][0]
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

    def cpr_spec(self, s1):
        #self.useful_functions.print('After adding', s1)
        #print("After adding: ", s1)
        if self.possible_fms == None:
            return None
        return self.possible_fms



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
        #for z in range(3):
        word2, rate = self.__right_check(words[ind], dicts, flag=flag)
        arr.append([word2, words[ind], ind, rate])
            #if ind == 1:
                #priority = priority[np.where(dicts != word2)]
        #if ind != 0:
        #        dicts = dicts[np.where(dicts[:,0] != word2)[0]]
        #    else:
        #        dicts = dicts[np.where(dicts != word2)]
            #temp_arr = np.array([word2])
            #dicts = np.setdiff1d(dicts, temp_arr)
        #three_arrs = self.useful_functions.sort_data(arr)
        return word2

    def cpr_obls(self, word, data, len_places, places_flag):
        #places = []
        obl_flag = True
        #for z in range(3):
        word_corr, rate, city_ind = self.__right_check(word, data, flag=1, places_flag=places_flag)
        if city_ind > len_places[0]:
            obl_flag = False

        return (word_corr, obl_flag, rate)


    def cpr_cities(self, word, data, places_flag):

        word_corr, rate, city_ind = self.__right_check(word, data, flag=1, places_flag=places_flag)

        return word_corr


    def find_FIO(self, words):
        name = self.cpr_fio(words, self.names_surnames[:, [0, 2]].copy(), 1)

        family = self.cpr_fio(words, self.families.copy(), 0)

        surname = self.cpr_fio(words, self.names_surnames[:, [1, 2]].copy(), 2)

        return name, family, surname


    def find_date(self, all_digits):
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
                date = digit[0:2] + '.' + digit[2:4] + '.' + year
                return date
            elif 5 <= len(digit) < 7:
                if len(digit) > 6:
                    digit = digit[:6]
                elif len(digit) < 6:
                    digit += '0' * (6 - len(digit))
            else:
                pass
        return ''


    def find_city(self, all_words):
        city = ''
        obl = ''
        krai = ''
        place = ''
        best_flag = None
        best_rate_obl = 0
        max_poss_obl = ''
        if len(all_words) == 1:
            if ' ' not in all_words[0]:
                city = self.cpr_cities(all_words[0], self.cities.copy(), places_flag=1)

            else:
                all_words = all_words[0].split(' ')
        #three_places = None
        if len(all_words) != 1:
            for i in range(len(all_words) - 1, -1, -1):
                word = all_words[i].lower()
                rate_krai = self.useful_functions.ratio(word, 'край')
                if rate_krai > 0.8:
                    krais = list(self.krai.keys())
                    krai, flag, rate  = self.cpr_obls(all_words[i - 1], list(self.krai.keys()).copy(), [0, len(krais)],
                                               places_flag=2)
                    if rate > best_rate_obl:
                        best_rate_obl = rate
                        max_poss_obl = krai
                        best_flag = flag
                rate_obl = max(self.useful_functions.ratio(word, 'обл.'), self.useful_functions.ratio(word, 'область'))
                if rate_obl > 0.8:
                    obls = list(self.obl.keys())
                    obl, flag, rate = self.cpr_obls(all_words[i - 1], list(self.obl.keys()).copy(), [len(obls), 0],
                                               places_flag=3)
                    if rate > best_rate_obl:
                        best_rate_obl = rate
                        max_poss_obl = obl
                        best_flag = flag
                rate_city = max(self.useful_functions.ratio(word, 'город'), self.useful_functions.ratio(word, 'гор.'))
                if rate_city > 0.8:
                    city = self.cpr_cities(all_words[i + 1], self.cities.copy(), places_flag=1)

                rate_der = max(self.useful_functions.ratio(word, 'деревня'), self.useful_functions.ratio(word, 'дер.'))
                if rate_der > 0.8:
                    pass
                rate_selo = max(self.useful_functions.ratio(word, 'село'), self.useful_functions.ratio(word, 'с.'))
                if rate_selo > 0.8:
                    pass
                rate_posel = max(self.useful_functions.ratio(word, 'посёлок'),
                                 self.useful_functions.ratio(word, 'пос.'))
                if rate_posel > 0.8:
                    pass
                rate_hutor = max(self.useful_functions.ratio(word, 'хутор'), self.useful_functions.ratio(word, 'хут.'))
                if rate_hutor > 0.8:
                    pass

            if city == '':
                new_list_cities = np.array([None])
                if obl == '' and krai == '':
                    for i in range(len(all_words) - 1, 0, -1):
                        all_obls = list(self.obl.keys()) + list(self.krai.keys())
                        len_obls = [len(list(self.obl.keys())), len(list(self.krai.keys()))]
                        new_place,flag, rate = self.cpr_obls(all_words[i], all_obls.copy(), len_obls, places_flag=2)
                        if rate>best_rate_obl:
                            best_flag = flag
                            max_poss_obl = new_place
                            best_rate_obl = rate

            new_list_cities = np.array([None])
            if max_poss_obl != '':
                if best_flag == True:
                    self.obl[max_poss_obl] = self.obl[max_poss_obl].reshape(self.obl[max_poss_obl].shape[0], 1)
                    new_list_cities = np.append(new_list_cities, self.obl[max_poss_obl])
                else:
                    self.krai[max_poss_obl] = self.krai[max_poss_obl].reshape(self.krai[max_poss_obl].shape[0], 1)
                    new_list_cities = np.append(new_list_cities, self.krai[max_poss_obl])
                new_list_cities = np.delete(new_list_cities, 0)
                city = self.cpr_cities(all_words[0], new_list_cities.copy(), places_flag=1)

                # Доделать для областей и краев
                # nas_

        if max_poss_obl == '':
            return city
        else:
            possible_end = 'обл.'
            if max_poss_obl in self.krai:
                possible_end = 'край'
            #if flag == True:

            max_poss_obl = max_poss_obl + ' ' + possible_end
            #else:
            #    max_poss_obl = max_poss_obl +' '+'край'
            return city + ',' + max_poss_obl


    def cpr_spec_words(self, s1, pers_data):
        self.data_fms = pers_data['фмс']
        #self.print('s1', s1)
        # print('s1:', s1)
        all_digits = []
        t1 = time.time()

        all_words = [word.lower().capitalize() for word in s1]

        name, family, surname = self.find_FIO(all_words)


        all_words = all_words[3:]
        new_words = []
        #first_ind = -1
        for (ind, word) in enumerate(all_words):
            digit_count = 0
            digits = ''
            for k in word:
                if k.isdigit():
                    digit_count += 1
                    digits += k

            if digit_count > 0:
                #if len(digits) > 7:
                    #first_ind = ind
                all_digits.append(digits)
            else:
                #if first_ind != -1:
                new_words.append(word)
        all_words = new_words


        date = self.find_date(all_digits)
        pers_data['пол'] = 'муж'
        pers_data['фамилия'] = family
        pers_data['имя'] = name
        pers_data['отчество'] = surname
        pers_data["дата рождения"] = date
        pers_data['место'] = self.find_city(all_words)
        cities = []
        return pers_data

