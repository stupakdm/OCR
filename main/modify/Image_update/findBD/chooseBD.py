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
        self.names = np.array(self.df_name['name'])
        self.surnames = np.array(self.df_name['surname'])
        self.priority = np.array(self.df_name['priority'])

        self.df_families = pd.read_csv(f'{curr_dir}families3.csv', encoding='utf-8')
        self.families = np.array(self.df_families['family'])

        self.df_cities = pd.read_csv(f'{curr_dir}RussianCities.tsv', sep='\t', index_col=False, encoding='utf-8')
        arr_name = np.array(self.df_cities['name'])
        arr_region = np.array(self.df_cities['region_name'])
        arr_region = np.unique(arr_region)
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
                        print('max_word_city: ', prob_city, corr_word)
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
                            print('max_word_city: ', prob_city, corr_word)
        if max_rate <= 0.5:
            return None
        else:
            return max_word



    def __right_check(self, text, data, flag=0):
        if text.lower() == 'имя' or text.lower() == 'код':
            return ('', 0)

        city_flag = False
        if np.array_equal(data, self.arr_name):
            city_flag = True
            w = data[0].split(' ')
            max_rate = lv.jaro(text, w[1])
            if w[0] == 'гор.':
                max_rate += 0.2*max_rate
        else:
            max_rate = lv.jaro(text, data[0])

        corr = data[0]

        for ind, word in enumerate(data):

            if city_flag:
                w = word.split(' ')
                w_copy = w[1]
                if '-' in w_copy:
                    w_copy = w_copy.replace('-', '')
                w_copy = w_copy.lower().capitalize()
                rate = lv.jaro(text, w_copy)
                if w[0] == 'гор.':
                    rate += 0.2*rate
                word_check = w[1]
            else:
                w_copy = word
                if '-' in w_copy:
                    w_copy = w_copy.replace('-', '')
                w_copy = w_copy.lower().capitalize()
                rate = lv.jaro(text, w_copy)
                word_check = word

            if rate > max_rate and (max(len(text), len(w_copy)) / min(len(text), len(w_copy))) < 1.5:
                if flag == 0:
                    if self.priority[ind] > 150:
                        # if word == 'Орест':
                        #    print('rate ', rate, 'dist: ', lv.distance(text, word), 'ratio:', lv.ratio(text, word))
                        # if word == 'Сергей':
                        #    print('rate ', rate, 'dist: ', lv.distance(text, word), 'ratio:', lv.ratio(text, word))
                        max_rate = rate
                        corr = word_check
                        # Печатать правильно подобранное слово
                        #print('corr:', corr, text)
                else:
                    max_rate = rate
                    corr = word_check
                    # Печатать правильно подобранное слово
                    #print('corr:', corr, text)
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

    def series_number(self, s1, pers_data):
        series = ''
        number = ''
        print(s1)
        for (ind, word) in enumerate(s1):
            if len(word) == 2 and word not in series:
                if ind != 0 and ind != len(s1) - 1:
                    if len(s1[ind - 1]) >= 2 or len(s1[ind + 1]) >= 2:
                        series += word
            if len(word) >= 5 and word not in number:
                number += word
        pers_data['серия'] = series[2:] + series[0:2]
        pers_data['номер'] = number
        return pers_data

    def delete_multiple_element(self, list_object, indices):
        indices = sorted(indices, reverse=True)
        for idx in indices:
            if idx < len(list_object):
                list_object.pop(idx)

    def find_code_ind(self, s1):
        all_digits = [None]
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
                        print("possible fms: ", self.codes[code_digit])
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

                    return (ind, code_digit, self.codes[code_digit][0], all_digits[0])
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
        print("After adding: ", s1)
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
        m = max(ratio)
        for i, val in enumerate(ratio):
            if m == val:
                return self.possible_fms[i]

    def cpr_spec_fms(self, s1, pers_data):
        print("After adding: ", s1)

        all_digits = []
        # all_words = []
        for (ind, word) in enumerate(s1):
            digit_count = 0
            digits = ''
            # words = ''
            for k in word:
                print('digit checking:', word)
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
                pers_data['фмс'] = self.codes[code]
            else:
                pass

        return pers_data

    def cpr_spec_words(self, s1, pers_data):
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
        print('s1:', s1)
        all_digits = []
        all_words = []
        first_ind = 0
        t1 = time.time()
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

        for (ind, word) in enumerate(all_words[:first_ind]):

            word2, rate = self.__right_check(word, self.names, flag=0)
            if rate > max_rate:
                max_rate = rate
                corr_name = word2
                prev_word = word
                ind_name = ind

        # print("Chosen name: ", corr_name)
        # print(prev_word[-3:].lower())
        # print('ratio вич: ', self.ratio(prev_word[-3:].lower(), 'вич'))
        if self.ratio(prev_word[-3:].lower(), 'вич') > 0.6:
            corr_name, ratio = self.__right_check(prev_word, self.surnames, flag=0)
            pers_data['отчество'] = corr_name
            ind_sur = ind_name
            corr_name = ''
            ind_name = 0
            fl = 1
        else:
            pers_data['имя'] = corr_name
        #print("Correct_name: ", corr_name, max_rate)
        #pers_data['имя'] = corr_name

        # Поиск фамилии и отчества, если найдено имя

        if fl == 0:
            for i in range(max(0, ind_name - 3), ind_name):

                fam, rate = self.__right_check(all_words[i], self.families, flag=1)
                if rate > rate_fam:
                    rate_fam = rate
                    family = fam
                    ind_fam = i
            pers_data['фамилия'] = family

            for i in range(ind_name + 1, min(ind_name + 3, first_ind)):

                sur, rate = self.__right_check(all_words[i], self.surnames, flag=0)
                if rate > rate_sur:
                    rate_sur = rate
                    surname = sur
                    ind_sur = i
            pers_data['отчество'] = surname

            # Поиск фамилии и имени, если найдено отчество
        else:
            for i in range(max(0, ind_sur - 3), ind_sur):

                fam, rate = self.__right_check(all_words[i], self.families, flag=1)
                if rate > rate_fam:
                    rate_fam = rate
                    family = fam
                    ind_fam = i
            pers_data['фамилия'] = family

            max_rate = 0

            t = 2
            if ind_fam == ind_sur-1:
                t = 1
            for i in range(max(0, ind_sur - 4), ind_sur-t):

                word, rate = self.__right_check(all_words[i], self.names, flag=0)
                if rate > max_rate:
                    max_rate = rate
                    corr_name = word
                    ind_name = i
            pers_data['имя'] = corr_name

        print("surname: ", all_words[ind_sur])

        ind_sur= s1.index(all_words[ind_sur].upper())
        city_word = s1[first_ind+1:]

        self.delete_multiple_element(all_words, [ind_name, ind_fam, ind_sur])

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
                if digit[4:8] > "2022":
                    date_digit += digit[4:8]
                    digit = digit[0:4] + "2012"
                pers_data["дата рождения"] = digit[0:2] + '.' + digit[2:4] + '.' + digit[4:8]
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

        # Поиск города и региона рождения
        ind_dig = 0
        print('city_word_testing: ', city_word)
        print('s1 testing: ',s1)
        print('pers_data: ', pers_data["дата рождения"])
        #if date_digit in all_words:
        #    ind_dig = all_words.index(date_digit)

        print('All words: ', all_words)
        print('date digit: ', date_digit)
        print('Probable city place: ', all_words[ind_dig:])
        print('cities: ', self.arr_name)
        cities = []
        for (ind, word) in enumerate(city_word):
            #if '.' in
            for symb in '.,:- ':
                if symb in word:
                    word = word.replace(symb, '')
            word = word.lower().capitalize()
            word1 = word
            word_corr, rate = self.__right_check(word, self.arr_name, flag=1)
            if rate > max_rate_city:
                print('City: ', word, 'corr word: ', word_corr)
                max_rate_city = rate
                corr_city = word_corr
                cities.append(corr_city)
                ind_city = ind
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
        city = self.city_comparing(cities, pers_data['фмс'])
        if city != None:
            corr_city = city
        if max_rate_city < 0.6:
            corr_city = '-'
        pers_data['место'] = corr_city #+ ' ' + corr_region

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
