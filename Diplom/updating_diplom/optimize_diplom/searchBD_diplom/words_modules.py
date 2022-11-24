import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from psycopg2 import Error
import pandas as pd
import numpy as np
import random
import enchant
from Levenshtein import quickmedian


class CorrectFunctions:
    def __init__(self):
        pass

    def divide_left_right_words(self, blocks, words, mid_x):
        '''
        blocks: Координаты блоков слов
        words: Слова, соответствующие этим блокам слов
        mid_x: Середина диплома

        Функция делит слова на те, что на левом листе
        и на те, что на провом листе
        '''
        left_words = []
        right_words = []
        left_blocks = []
        right_blocks = []
        for ind, block in enumerate(blocks):
            if block[2] < mid_x:
                left_words.append(words[ind])
                left_blocks.append(block)
            else:
                right_words.append(words[ind])
                right_blocks.append(block)
        return (left_blocks, left_words), (right_blocks, right_words)

    def sort_by_geometry(self, blocks, words):
        '''
        blocks: Координаты блоков слов
        words: Слова, соответствующие этим блокам слов

        Сортировка слов по их местоположению:
        Слова сортируются сверху вниз, а если они находятся рядом - то есть составляют одно предложение -
        то оно будут вместе идти друг за другом
        '''
        new_words = []
        sort_with_ind = sorted(enumerate(blocks), key=lambda x: x[1][1])
        #i = 0
        #new_blocks = []
        for i in range(1, len(sort_with_ind)):
            #block_copy = sort_with_ind[i]
            for j in range(i-1, -1, -1):
                if abs(sort_with_ind[j+1][1][1] - sort_with_ind[j][1][1])<5:
                    #if abs(sort_with_ind[j+1][1][2] - sort_with_ind[j][1][0])<=5 or sort_with_ind[j][1][0]<=sort_with_ind[j+1][1][2]<= sort_with_ind[j][1][2] or sort_with_ind[j+1][1][0]<=sort_with_ind[j][1][2]<= sort_with_ind[j+1][1][2]:
                        #block_copy = sort_with_ind[j][1]
                        if sort_with_ind[j+1][1][0] < sort_with_ind[j][1][0]:
                            sort_with_ind[j+1], sort_with_ind[j] = sort_with_ind[j], sort_with_ind[j+1]
                else:
                    break

        indexes = [i[0] for i in sort_with_ind]
        new_blocks = [i[1] for i in sort_with_ind]
        sort_with_ind.clear()
        for ind in indexes:
            new_words.append(words[ind])
        return new_blocks, new_words

    def to_right_size(self, H, W, blocks):
        '''
        H: высота изображения
        W: ширина изображения
        blocks: Координаты блоков слов

        Приведение блоков слов к масштабу изображения
        '''
        new_block = []
        for block in blocks:
            new_block.append(self.new_size(H, W, block))
        return new_block

    def new_size(self, H, W, block):
        '''
        H: высота изображения
        W: ширина изображения
        blocks: Координаты блоков слов

        Приведение блока к масштабу изображения
        '''
        return [int(block[0][0] * W), int(block[0][1] * H), int(block[1][0] * W), int(block[1][1] * H)]




class Get_BD:
    '''def optimize_strings(file, name):
        curr_dir = 'Updating/Optimize/SearchBD/data'

        with open(f'{curr_dir}/{file}') as file:
            lines = file.readlines()
            df = pd.DataFrame(columns=['number', 'name'])
            for (i, line) in enumerate(lines):
                line = line.replace('"', '')
                line = line.replace('\n', '')
                # print(f"{i}: {line}")
                if (' ' not in line):
                    continue
                ind = line.find(' ')
                df.loc[i] = [line[:ind], line[ind + 1:]]

            df.to_csv(f'{name}.csv', sep=';', encoding='utf-8', index=False)

    def create_specialities(self):
        files = ['Специальности_Бакалавриат.csv', "Специальности_Магистратура.csv", 'Специальности_Специалитет.csv']
        new_name = ['Bachelor', 'Magistr', 'Speciality']
        for i in range(len(files)):
            self.optimize_strings(files[i], new_name[i])
    '''

    def create_database(self):
        '''
        Создание БД

        '''
        try:
            connection = psycopg2.connect(user="postgres",
                                          password='1111',
                                          host='127.0.0.1',
                                          port="5432")
            # database="postgres_db")
            connection.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
            cursor = connection.cursor()

            sql_screate_database = 'create database prostgres_db'
            cursor.execute(sql_screate_database)
        except (Exception, Error) as error:
            print("Ошибка при работе с PostgreSQL", error)
        finally:
            if connection:
                cursor.close()
                connection.close()
                print("Соединение с PostgreSQL закрыто")

    def create_table(self):
        '''
        Создание таблиц в БД

        '''
        try:
            connection = psycopg2.connect(user="postgres",
                                          # пароль, который указали при установке PostgreSQL
                                          password="1111",
                                          host="127.0.0.1",
                                          port="5432",
                                          database="postgres_db")

            # Курсор для выполнения операций с базой данных
            cursor = connection.cursor()
            create_table_bachelor = '''CREATE TABLE bachelor
                                    (id integer PRIMARY KEY NOT NULL,
                                    number text NOT NULL,
                                    name text NOT NULL);'''

            # Выполнение команды: это создает новую таблицу
            cursor.execute(create_table_bachelor)
            connection.commit()

            create_table_magistr = '''CREATE TABLE magistr
                                    (id integer PRIMARY KEY NOT NULL,
                                    number text NOT NULL,
                                    name text NOT NULL);'''
            cursor.execute(create_table_magistr)
            connection.commit()
            create_table_speciality = '''CREATE TABLE speciality
                                    (id integer PRIMARY KEY NOT NULL,
                                    number text NOT NULL,
                                    name text NOT NULL,
                                    quality text);'''

            cursor.execute(create_table_speciality)
            connection.commit()

            create_table_univerity = '''CREATE TABLE university
                                                (id integer PRIMARY KEY NOT NULL,
                                                full_name text NOT NULL,
                                                adress text NOT NULL);'''

            cursor.execute(create_table_univerity)
            connection.commit()

            create_table_namessurnames = '''CREATE TABLE namessurnames
                                                    (id integer PRIMARY KEY NOT NULL,
                                                    name text NOT NULL,
                                                    surname text NOT NULL,
                                                    priority integer NOT NULL);'''

            cursor.execute(create_table_namessurnames)

            connection.commit()
            create_table_families = '''CREATE TABLE families
                                                    (id integer PRIMARY KEY NOT NULL,
                                                    family text NOT NULL);'''

            cursor.execute(create_table_families)
            connection.commit()
        except (Exception, Error) as error:
            print("Ошибка при работе с PostgreSQL", error)
        finally:
            if connection:
                cursor.close()
                connection.close()
                print("Соединение с PostgreSQL закрыто")

    def insert_values(self):
        '''
        Добавление значений в БД

        '''
        curr_dir = 'Updating/Optimize/SearchBD/data/'   # Местоположение .csv и других файлов, содержащие данные
        try:
            connection = psycopg2.connect(user="postgres",
                                          # пароль, который указали при установке PostgreSQL
                                          password="1111",
                                          host="127.0.0.1",
                                          port="5432",
                                          database="postgres_db")

            # Курсор для выполнения операций с базой данных
            cursor = connection.cursor()

            #Insert Values
            bach = pd.read_csv(f'{curr_dir}/Bachelor.csv', encoding='utf-8', sep=';')
            for i, row in bach.iterrows():
                cursor.execute(f"INSERT INTO bachelor VALUES ({i}, {row['number']}, {row['name']})")
                #print(row['number'], row['name'])
            connection.commit()

            bach.close()

            mag = pd.read_csv(f'{curr_dir}/Magistr.csv', encoding='utf-8', sep=';')
            for i, row in mag.iterrows():
                cursor.execute(f"INSERT INTO magistr VALUES ({i}, {row['number']}, {row['name']})")
                #print(row['number'], row['name'])
            connection.commit()

            spec_all = pd.read_csv(f'{curr_dir}/special_quality.csv', encoding='utf-8', sep=';')
            spec_all['quality'].fillna('', inplace=True)
            for i, row in spec_all.iterrows():
                cursor.execute(f"INSERT INTO speciality VALUES ({i}, {row['number']}, {row['name']}, {row['quality']})")
                #print(row['number'], row['name'])
            connection.commit()

            universities = pd.read_csv(f'{curr_dir}/Университеты России.csv', encoding='utf-8', sep=',')
            for i, row in universities.iterrows():
                cursor.execute(f"INSERT INTO speciality VALUES ({i}, {row['Full Name']}, {row['Adress']})")
                #print(row['number'], row['name'])
            connection.commit()

            df_name = pd.read_csv(f'{curr_dir}Names_and_surnames2.csv', encoding='utf-8', sep=',')
            for i, row in df_name.iterrows():
                cursor.execute(f"INSERT INTO speciality VALUES ({i}, {row['name']}, {row['surname']}, {row['priority']})")
                #print(row['number'], row['name'])
            connection.commit()

            df_families = pd.read_csv(f'{curr_dir}families3.csv', encoding='utf-8')
            for i, row in df_families.iterrows():
                cursor.execute(f"INSERT INTO speciality VALUES ({i}, {row['family']})")
                #print(row['number'], row['name'])
            connection.commit()

        except (Exception, Error) as error:
            print("Ошибка при работе с PostgreSQL", error)
        finally:
            if connection:
                cursor.close()
                connection.close()
                print("Соединение с PostgreSQL закрыто")

class BDFunctions:

    def __init__(self):
        self.dict = enchant.Dict('ru_RU')

    def is_upper_case(self, word):
        '''
        word: Слово
        Проверка букв в слове на верхний регистр
        '''
        count = 0
        for w in word:
            if w == w.upper() and w.isdigit() == False and w not in ':;[]{}@.,?""':
                count+=1
        if count >= len(word)//2:
            return True
        return False

    def proccess_string(self, string):
        '''
        string: Строка
        Обработка каждого слова в строке
        '''
        new_string = string.split(' ')
        string = []
        for (ind, word) in enumerate(new_string):
            if self.is_upper_case(word): # приведение к верхнему регистру
                word = word.upper()
            else:
                word = word.lower()

            for symb in ':;[]{}@':
                if symb in word:
                    word = word.replace(symb, '')   # удаление лишних символов
            if word != '':
                string.append(word)
        return ' '.join(string)

    def filt_names(self, word):
        return not '^' in word

    def checkfordigits(self, word):
        '''
        word: строка
        Разделение строки на строчные подстроки и числовые подстроки
        '''
        if word == '':
            return ['']
        word = word.strip(' ')
        words = []
        w = word[0]
        flag = 0
        if word[0].isalpha():
            flag = 1
        for i in range(1, len(word)):
            if flag == 0 and word[i].isalpha():
                flag = 1
                words.append(w)
                w = word[i]
            elif flag == 1 and word[i].isdigit():
                flag = 0
                words.append(w)
                w = word[i]
            else:
                w+=word[i]
        words.append(w)
        return words

    def check_digit(self, word, ind=0):
        '''
        word: слово, состоящее из чисел
        ind: флаг = 0(по умолчанию)
        Проверяет, состоит ли строка преимущественно из чисел или нет
        '''
        digit_count = 0
        for w in word:
            if digit_count > len(word)//2:
                return True
            if w.isdigit():
                digit_count+=1
        if digit_count > len(word) // 2:
            return True
        if ind == 1:
            if digit_count >=4:
                return True
        return False

    def clean_number(self, number):
        '''
        number: Строка, состоящая из чисел
        Приведение строки к формату даты xx.xx.xx

        '''
        new_n = ''
        fl = 0
        count_d = 0
        for w in number:
            if w.isdigit():
                new_n += w
                fl+=1
            if fl == 2:
                count_d +=1
                if count_d ==3:
                    break
                new_n += '.'
                fl = 0
        return new_n

    def limit_words(self, words):
        limit = 200
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

        h = int(limit // 10)
        all_words = []
        for i in range(0, len(words), h):
            word_cmp = quickmedian(words[i:i+h])
            d = self.dict.suggest(word_cmp)
            all_words.append(d)
        return all_words

    def sort_by_priority(self, words):
        words = sorted(words, key=lambda x: x[2])
        return words

    def check_rate(self, word, poss_word,rate, min_rate = 0.8):
        if rate > min_rate:
            return poss_word
        else:
            return word

