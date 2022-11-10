import enchant
import Levenshtein as lv
import numpy as np
import pandas as pd
import re
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from psycopg2 import Error



def optimize_strings(file, name):
    curr_dir = 'Updating/Optimize/SearchBD/data'

    with open(f'{curr_dir}/{file}') as file:
        lines = file.readlines()
        df = pd.DataFrame(columns=['number', 'name'])
        for (i,line) in enumerate(lines):
            line = line.replace('"', '')
            line = line.replace('\n', '')
            #print(f"{i}: {line}")
            if (' ' not in line):
                continue
            ind = line.find(' ')
            df.loc[i] = [line[:ind], line[ind+1:]]

        df.to_csv(f'{name}.csv', sep=';', encoding = 'utf-8', index=False)


def create_specialities():
    files  = ['Специальности_Бакалавриат.csv', "Специальности_Магистратура.csv", 'Специальности_Специалитет.csv']
    new_name = ['Bachelor', 'Magistr', 'Speciality']
    for i in range(len(files)):
        optimize_strings(files[i], new_name[i])


def create_database():
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

def create_table():
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

def insert_values():
    curr_dir = 'Updating/Optimize/SearchBD/data/'
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
        #f = open('Bachelor.csv')
class FindWord:
    rus_alphabet = 'ёабвгдежзийклмнопрстуфхцчшщъыьэюя'
    rus_alphabet_upper =  rus_alphabet.upper()
    spec_symbols = '.,()""«»'
    special_words = [
        'диплом',
        'профессиональном',
        'среднем',
        'образованиии',
        'квалификация',
        'квалификации',
        'дата',
        'документ',
        'свидетельствует',
        'программу',
        'среднего',
        'успешно',
        'прошел',
        'государственную',
        'итоговую',
        'освоил',
        'специальности',
        'Решением',
        'государтсвенной',
        'комисии',
    ]

    alphabet = {
        'A': ['Д', 'А', 'Л', 'Я', 'И'],
        'B': ['В', 'Б', 'Ы', 'Ь'],
        'B|': ['Ы'],
        'C': ['С'],
        'DK': ['Ж'],
        'D': ['О', 'Ф'],
        'E': ['Е', 'В', 'Я', 'Б'],
        'F': ['Г'],
        'G': ['О'],
        'Hl': ['П'],
        'H': ['Н', 'И', 'П', 'Ч'],
        '/I': ['Д'],
        'IB': ['Е'],
        'II': ['П'],
        'IJ': ['Ц'],
        'IO': ['Ю'],
        'JI': ['Л', 'П'],
        'JL': ['Л'],
        'IL': ['П', 'Ц', 'Н', 'Щ'],
        'I': ['П', 'Д', 'Г', 'Н', 'Л', 'И'],
        'J': ['У', 'Л', 'Я'],
        'K': ['К', 'Ж'],
        'L': ['Ь', 'Ц'],
        'M': ['М', 'И', 'Й'],
        'N': ['И', 'Й', 'Н'],
        'O': ['О', 'Ф'],
        'P': ['Р'],
        'Q': ['О'],
        'R': ['Я', 'К'],
        'SI': ['Я'],
        'S': ['Я'],
        'TI': ['П'],
        'T': ['Т', 'Л', 'П', 'Г'],
        'U': ['И', 'Н', 'Ц', 'О'],
        'V': ['У', 'Х', 'Ж', 'И'],
        'W': ['У', 'Х'],
        'X': ['Х', 'У', 'Ж', 'Д'],
        'Y': ['У', 'Х', 'Ж', 'Ч'],
        'Z': ['И', 'Й'],
        '10': ['Ю'],
        '0': ['О'],
        '1': ['И'],
    }
    """     
            try:
                connection = psycopg2.connect(user="postgres",
                                              password = '1111',
                                              host='127.0.0.1',
                                              port = "5432")
                                              #database="postgres_db")
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
            #Подключение к БД и создание таблицы
            try:
                connection = psycopg2.connect(user="postgres",
                                              # пароль, который указали при установке PostgreSQL
                                              password="1111",
                                              host="127.0.0.1",
                                              port="5432",
                                              database="postgres_db")

                # Курсор для выполнения операций с базой данных
                cursor = connection.cursor()
                create_table_bachelor = '''CREATE TABLE BACHELOR
                                        (id integer PRIMARY KEY NOT NULL,
                                        number text NOT NULL,
                                        name text NOT NULL);'''

                # Выполнение команды: это создает новую таблицу
                cursor.execute(create_table_bachelor)
                connection.commit()

                create_table_magistr = '''CREATE TABLE MAGISTR
                                        (id integer PRIMARY KEY NOT NULL,
                                        number text NOT NULL,
                                        name text NOT NULL);'''
                cursor.execute(create_table_magistr)
                connection.commit()
                create_table_speciality ='''CREATE TABLE SPECIALITY
                                        (id integer PRIMARY KEY NOT NULL,
                                        number text NOT NULL,
                                        name text NOT NULL,
                                        quality text);'''


                cursor.execute(create_table_speciality)
                connection.commit()

                bach = pd.read_csv(f'{curr_dir}/Bachelor.csv', encoding='utf-8', sep=';')
                for number, name in bach.items():
                    print(number, name)



            except (Exception, Error) as error:
                print("Ошибка при работе с PostgreSQL", error)
            finally:
                if connection:
                    cursor.close()
                    connection.close()
                    print("Соединение с PostgreSQL закрыто")
            """
    def __init__(self):
        #Create database
        curr_dir = 'Updating/Optimize/SearchBD/data/'
        """try:
            connection = psycopg2.connect(user="postgres",
                                            password = '1111',
                                            host='127.0.0.1',
                                            port = "5432")
                                          #database="postgres_db")
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

        #Подключение к БД и создание таблицы
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
            create_table_speciality ='''CREATE TABLE speciality
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
        """

        #Get data by SELECT
        try:
            connection = psycopg2.connect(user="postgres",
                                          # пароль, который указали при установке PostgreSQL
                                          password="1111",
                                          host="127.0.0.1",
                                          port="5432",
                                          database="postgres_db")
            cursor = connection.cursor()
            sql_get_bachelor = 'SELECT * FROM bachelor'
            cursor.execute(sql_get_bachelor)
            bach = cursor.fetchall()

            sql_get_magistr = 'SELECT * FROM magistr'
            cursor.execute(sql_get_magistr)
            mag = cursor.fetchall()

            sql_get_speciality = 'SELECT * FROM speciality'
            cursor.execute(sql_get_speciality)
            spec = cursor.fetchall()

            sql_get_university = 'SELECT * FROM university'
            cursor.execute(sql_get_university)
            university = cursor.fetchall()

            sql_get_namessurnames = 'SELECT * FROM namessurnames'
            cursor.execute(sql_get_namessurnames)
            namessurnames = cursor.fetchall()

            sql_get_families = 'SELECT * FROM families'
            cursor.execute(sql_get_families)
            families = cursor.fetchall()

        except (Exception, Error) as error:
            print("Ошибка при работе с PostgreSQL", error)
        finally:
            if connection:
                cursor.close()
                connection.close()
                print("Соединение с PostgreSQL закрыто")

        sogl1 = 'йцкгшщзхфвпрджчмьб'
        sogl2 = 'нслт'
        eng_sogl = 'qwrtpsdfghjklzxcvbnm'
        eng_gl = 'aeyuio'
        self.eng_sogl = eng_sogl + eng_sogl.upper()
        self.eng_gl = eng_gl + eng_gl.upper()
        self.gl = 'уеёаоэяию'
        self.sogl = sogl1 + sogl1.upper() + sogl2 + sogl2.upper()
        self.gl += self.gl.upper()
        self.ban = 'ыьйъ'
        self.ban += self.ban.upper()

        self.engl_alphabet = dict()

        for key in self.alphabet.keys():
            for letter in self.alphabet[key]:
                if letter not in self.engl_alphabet.keys():
                    self.engl_alphabet[letter] = [key]
                else:
                    self.engl_alphabet[letter].append(key)

        curr_dir = 'Updating/Optimize/SearchBD/data/'
        bach = pd.read_csv(f'{curr_dir}/Bachelor.csv', encoding='utf-8', sep=';')

        mag = pd.read_csv(f'{curr_dir}/Magistr.csv', encoding='utf-8', sep=';')
        # spec = pd.read_csv(f'{curr_dir}/Speciality.csv', encoding='utf-8', sep=';')
        spec_all = pd.read_csv(f'{curr_dir}/special_quality.csv', encoding='utf-8', sep=';')
        spec_all['quality'].fillna('', inplace=True)
        self.spec_number = np.array(spec_all['number'])
        self.spec_name = np.array(spec_all['name'])
        self.spec_quality = np.array(spec_all['quality'])
        self.mag_number = np.array(mag['number'])
        self.mag_name = np.array(mag['name'])
        self.bach_number = np.array(bach['number'])
        self.bach_name = np.array(bach['name'])
        universities = pd.read_csv(f'{curr_dir}/Университеты России.csv', encoding='utf-8', sep=',')
        self.universities = np.array(universities['Full Name'])
        self.cities_university = np.array(universities['Adress'])
        for i in range(self.cities_university.shape[0]):
            city = self.cities_university[i]
            city = city.lower()
            substr = ''
            if 'г.' in city:
                substr = 'г.'
            elif 'с.' in city:
                substr = 'с.'
            elif 'пгт.' in city:
                substr = 'пгт.'
            elif 'пос.' in city:
                substr = 'пос.'
            elif 'д.' in city:
                substr = 'дер.'
            elif 'хут.' in city:
                substr = 'хут.'
            f = city.find(substr)
            n = f + len(substr)
            if city[n] == ' ':
                n += 1
            name = city[n:]
            name = name[:name.find(' ') - 1]
            self.cities_university[i] = substr + ' ' + name.capitalize()
        self.universities_by_cities = dict()
        for i in range(self.cities_university.shape[0]):
            if self.cities_university[i] not in self.universities_by_cities.keys():
                self.universities_by_cities[self.cities_university[i]] = [self.universities[i]]
            else:
                self.universities_by_cities[self.cities_university[i]].append(self.universities[i])

        df_name = pd.read_csv(f'{curr_dir}Names_and_surnames2.csv', encoding='utf-8')
        self.names = np.array(df_name['name'])
        self.surnames = np.array(df_name['surname'])
        self.priority = np.array(df_name['priority'])

        df_families = pd.read_csv(f'{curr_dir}families3.csv', encoding='utf-8')
        self.families = np.array(df_families['family'])

        self.begin_fio_bd = dict()
        for w in self.rus_alphabet_upper:
            self.begin_fio_bd[w] = []
            self.begin_fio_bd[w].append([0, None])
            self.begin_fio_bd[w].append([0, None])
            self.begin_fio_bd[w].append([0, None])

        begin = self.families[0][0]
        for i in range(self.families.shape[0]):
            if self.families[i][0] != begin:
                self.begin_fio_bd[begin][0][1] = i
                begin = self.families[i][0]
                self.begin_fio_bd[begin][0][0] = i

        begin = self.names[0][0]
        for i in range(self.names.shape[0]):
            if self.names[i][0] != begin:
                self.begin_fio_bd[begin][1][1] = i
                self.begin_fio_bd[begin][2][1] = i
                begin = self.names[i][0]
                self.begin_fio_bd[begin][1][0] = i
                self.begin_fio_bd[begin][2][0] = i

        # begin = self.surnames[0][0]
        # for i in range(self.surnames.shape[0]):
        #    if self.surnames[i][0] != begin:
        #        self.begin_fio_bd[begin][2][1] = i
        #        begin = self.surnames[i][0]
        #        self.begin_fio_bd[begin][2][0] = i

        self.begin_fio_bd['Я'][0][1] = self.families.shape[0]
        self.begin_fio_bd['Я'][1][1] = self.names.shape[0]
        self.begin_fio_bd['Я'][2][1] = self.surnames.shape[0]

        self.dict = enchant.Dict('ru_RU')
        self.my_bd = [self.bach_number, self.bach_name, self.mag_number, self.mag_name, self.spec_number,
                      self.spec_name, self.universities, self.names, self.surnames, self.families]

    def initialize(self):
        self.prob_words = []
        self.best_month_rate = 0
        self.count_month = [0] * 12
        self.best_month = [[None, 0] for i in range(3)]

    '''def __init__(self):
        self.prob_words = []
        sogl1 = 'йцкгшщзхфвпрджчмьб'
        sogl2 = 'нслт'
        eng_sogl = 'qwrtpsdfghjklzxcvbnm'
        eng_gl = 'aeyuio'
        self.eng_sogl = eng_sogl + eng_sogl.upper()
        self.eng_gl = eng_gl + eng_gl.upper()
        self.gl = 'уеёаоэяию'
        self.sogl = sogl1 + sogl1.upper() + sogl2 + sogl2.upper()
        self.gl += self.gl.upper()
        self.ban = 'ыьйъ'
        self.ban += self.ban.upper()

        self.engl_alphabet = dict()

        for key in self.alphabet.keys():
            for letter in self.alphabet[key]:
                if letter not in self.engl_alphabet.keys():
                    self.engl_alphabet[letter] = [key]
                else:
                    self.engl_alphabet[letter].append(key)
        self.count_month = [0]*12
        self.best_month_rate = 0

        self.best_month = [[None, 0] for i in range(3)]
        curr_dir = 'Updating/Optimize/SearchBD/data/'
        bach = pd.read_csv(f'{curr_dir}/Bachelor.csv', encoding='utf-8', sep=';')
        mag = pd.read_csv(f'{curr_dir}/Magistr.csv', encoding='utf-8', sep=';')
        #spec = pd.read_csv(f'{curr_dir}/Speciality.csv', encoding='utf-8', sep=';')
        spec_all = pd.read_csv(f'{curr_dir}/special_quality.csv', encoding='utf-8', sep=';')
        spec_all['quality'].fillna('', inplace=True)
        self.spec_number = np.array(spec_all['number'])
        self.spec_name = np.array(spec_all['name'])
        self.spec_quality = np.array(spec_all['quality'])
        self.mag_number = np.array(mag['number'])
        self.mag_name = np.array(mag['name'])
        self.bach_number = np.array(bach['number'])
        self.bach_name = np.array(bach['name'])
        universities = pd.read_csv(f'{curr_dir}/Университеты России.csv', encoding='utf-8', sep=',')
        self.universities = np.array(universities['Full Name'])
        self.cities_university = np.array(universities['Adress'])
        for i in range(self.cities_university.shape[0]):
            city = self.cities_university[i]
            city = city.lower()
            substr = ''
            if 'г.' in city:
                substr = 'г.'
            elif 'с.' in city:
                substr=  'с.'
            elif 'пгт.' in city:
                substr = 'пгт.'
            elif 'пос.' in city:
                substr = 'пос.'
            elif 'д.' in city:
                substr = 'дер.'
            elif 'хут.' in city:
                substr = 'хут.'
            f = city.find(substr)
            n = f+len(substr)
            if city[n] == ' ':
                n+=1
            name = city[n:]
            name = name[:name.find(' ')-1]
            self.cities_university[i] = substr+' '+name.capitalize()
        self.universities_by_cities = dict()
        for i in range(self.cities_university.shape[0]):
            if self.cities_university[i] not in self.universities_by_cities.keys():
                self.universities_by_cities[self.cities_university[i]] = [self.universities[i]]
            else:
                self.universities_by_cities[self.cities_university[i]].append(self.universities[i])

        df_name = pd.read_csv(f'{curr_dir}Names_and_surnames2.csv', encoding='utf-8')
        self.names = np.array(df_name['name'])
        self.surnames = np.array(df_name['surname'])
        self.priority = np.array(df_name['priority'])

        df_families = pd.read_csv(f'{curr_dir}families3.csv', encoding='utf-8')
        self.families = np.array(df_families['family'])

        self.begin_fio_bd = dict()
        for w in self.rus_alphabet_upper:
            self.begin_fio_bd[w] = []
            self.begin_fio_bd[w].append([0, None])
            self.begin_fio_bd[w].append([0, None])
            self.begin_fio_bd[w].append([0, None])


        begin = self.families[0][0]
        for i in range(self.families.shape[0]):
            if self.families[i][0] != begin:
                self.begin_fio_bd[begin][0][1] = i
                begin = self.families[i][0]
                self.begin_fio_bd[begin][0][0] = i

        begin = self.names[0][0]
        for i in range(self.names.shape[0]):
            if self.names[i][0] != begin:
                self.begin_fio_bd[begin][1][1] = i
                self.begin_fio_bd[begin][2][1] = i
                begin = self.names[i][0]
                self.begin_fio_bd[begin][1][0] = i
                self.begin_fio_bd[begin][2][0] = i

        #begin = self.surnames[0][0]
        #for i in range(self.surnames.shape[0]):
        #    if self.surnames[i][0] != begin:
        #        self.begin_fio_bd[begin][2][1] = i
        #        begin = self.surnames[i][0]
        #        self.begin_fio_bd[begin][2][0] = i

        self.begin_fio_bd['Я'][0][1] = self.families.shape[0]
        self.begin_fio_bd['Я'][1][1] = self.names.shape[0]
        self.begin_fio_bd['Я'][2][1] = self.surnames.shape[0]

        self.dict = enchant.Dict('ru_RU')
        self.my_bd = [self.bach_number, self.bach_name, self.mag_number, self.mag_name, self.spec_number,
                      self.spec_name, self.universities, self.names, self.surnames, self.families]
    '''

    def find_bd_word(self, word, bd, max_score, result_bd, result_word):
        for word_bd in bd:
            score = lv.ratio(word, word_bd)
            if score > max_score:
                max_score = score
                result_bd = bd
                result_word = word_bd
        if max_score < 0.4:
            return None, None, 0
        return result_bd, result_word, max_score


    def is_upper_case(self, word):
        count = 0
        for w in word:
            if w == w.upper() and w.isdigit() == False and w not in ':;[]{}@.,?""':
                count+=1
        if count >= len(word)//2:
            return True
        return False

    def proccess_string(self, string):
        new_string = string.split(' ')
        string = []
        for (ind, word) in enumerate(new_string):
            if self.is_upper_case(word):
                word = word.upper()
            else:
                word = word.lower()

            for symb in ':;[]{}@':
                if symb in word:
                    word = word.replace(symb, '')
            if word != '':
                string.append(word)
        return ' '.join(string)


    def cmp_rates(self, word_test, word_valid):
        rate1 = lv.ratio(word_test, word_valid)
        rate2 = lv.jaro(word_test, word_valid)
        rate = (rate1+rate2)/2
        return rate

    def __check_best_word(self, data, word_cmp):
        data = list(set(data))
        best_rate = self.cmp_rates(word_cmp, data[0])
        best_word = data[0]
        best_three = [[None, 0] for i in range(0, 3)]
        for word in data:
            rate = self.cmp_rates(word_cmp, word)
            for j in range(len(best_three)):
                if rate > best_three[j][1]:
                    for z in range(len(best_three)-1, j):
                        best_three[z] = best_three[z-1].copy()
                    best_three[j] = [word, rate]
                    best_rate = rate
                    best_word = word
                    break
            #if rate > best_rate:
            #    best_rate = rate
            #    best_word = word
        best_three = [word[0] for word in best_three if word[0] is not None]
        return best_three

    def findCity(self, word, possible_cities):
        city = word[1]
        if self.cmp_rates(city.lower().capitalize(), 'Дубликат') > 0.8:
            city = word[0]
        if city == '':
            city = possible_cities[1]
            if '.' in city:
                city = city[city.find('.')+1:]
            if 'r' in city:
                city = city.replace('r', '')
            city = city.upper()
            city = self.filterForRegNum(city)

            #city = self.translate_word(city, [])
            all_words = self.check_before_translate(city, size=len(city))

            city = lv.quickmedian(all_words)
            city = city.lower().capitalize()
            city = 'г. '+city
        best_cities = self.__check_best_word(self.cities_university, city)
        return best_cities


    def findUniversity(self, string, city):
        data = None
        if city != '':
            data = self.universities_by_cities[city]
        else:
            data = self.universities_by_cities.values()
            full_data = []
            for d in data:
                full_data += d
            data = full_data.copy()
            full_data.clear()
        best_universities = self.__check_best_word(data, string)
        return best_universities


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

    def remove_signs(self, word):
        symbs = ['.', ',', ':', '(', ')', "'", '‚','‘', '”']
        for symb in symbs:
            while symb in word:
                word = word[:word.index(symb)]+word[word.index(symb) + 1:len(word)]
        '''if ',' in word:
            word = word[:word.index('.')]+word[word.index('.') + 1:len(word)]
        if ':' in word:
            word = word[:word.index('.')]+word[word.index('.') + 1:len(word)]
        if '(' in word:
            word = word[:word.index('.')]+word[word.index('.') + 1:len(word)]
        if ')' in word:
            word = word[:word.index('.')]+word[word.index('.') + 1:len(word)]'''
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

    def check_for_letter(self, word):
        for w in word:
            if w.isalpha():
                return word.find(w)
        return -1

    def check_before_translate(self, word, size):
        if len(word) > size:
            word = word[:size]
        return self.translate_word(word, [])

    def translate_word(self, word, words):
        all_words = []
        #if len(word) < 12:

        for (ind, letter) in enumerate(word):
            if letter in self.alphabet.keys():
                if len(self.alphabet[letter]) > 1:
                    for j in range(0, min(3, len(self.alphabet[letter]))):  # Органичим на всего 3 буквы можно заменять
                        t = words.copy()
                        t.append(self.alphabet[letter][j])
                        all_words += self.translate_word(word[ind + 1:len(word)], t)
                words.append(self.alphabet[letter][0])
            else:
                words.append(letter)

        # joint = ''.join(words).lower().capitalize()
        joint_orig = ''.join(words)
        # regex = '|'.join(map(re.escape, self.delimiters))
        # joint = re.split(regex, joint)
        # all_words += joint
        # print('joint: ', joint, self.morph.word_is_known(joint))
        # if self.morph.word_is_known(joint):
        all_words.append(joint_orig)
        # else:
        self.prob_words.append(joint_orig)
        # f (p[0].score > 0.5):
        # print(joint_orig)
        # all_words.append(joint)

        return all_words


    def find_string(self, word):
        all_words = self.check_before_translate(word, size=3)
        #all_words = self.translate_word(word, [])
        all_words = self.filt_words(all_words)
        dict = enchant.Dict('ru_RU')
        all_enchant_words = []
        levels = [0, 0, 0]
        levels_begin = ['БАК', 'МАГ', 'СПЕ']
        for word in all_words:
            for ind, level in enumerate(levels_begin):
                rate = self.cmp_rates(word, level)
                levels[ind] += rate

        '''for word in all_words:
            true_w = dict.suggest(word)
            all_enchant_words += true_w
            # all_enchant_words.append(true_w)
        first_string = lv.quickmedian(all_enchant_words)'''
        if max(levels) == levels[0]:
            return 'бакалавр'
        if max(levels) == levels[1]:
            return 'магистр'
        if max(levels) == levels[2]:
            return 'специалитет'
        #return first_string

    def findLevel(self, word):
        first_string = self.find_string(word)
        return first_string
        """rate_bach = self.cmp_rates(first_string, 'БАКАЛАВРА')
        rate_mag = self.cmp_rates(first_string, 'МАГИСТРА')
        rate_spec = self.cmp_rates(first_string, 'СПЕЦИАЛИТЕТА')
        if (rate_bach == max(rate_spec, rate_mag, rate_bach)):
            return "БАКАЛАВРА"
        if (rate_mag == max(rate_spec, rate_mag, rate_bach)):
            return "МАГИСТРА"
        if (rate_spec == max(rate_spec, rate_mag, rate_bach)):
            return "СПЕЦИАЛИТЕТА"
        return 'БАКАЛАВРА'
        """
    def findNumberSeries(self, number, series):
        num = number
        ser = series
        if len(num) == 5:
            if number[0] != '1':
                num = '1'+num
            else:
                num = num+'0'

        if len(ser) == 6:
                ser = '0' + ser
        return num, ser

    def checkfordigits(self, word):
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

    def filterForRegNum(self, word):
        word = self.remove_signs(word)
        return word

    def cmp_lv_date(self, word, months):
        best_rate = self.cmp_rates(word, months[0])
        best_month = months[0]
        best_three = [[None, 0] for i in range(0, 3)]

        for month in months:
            rate = self.cmp_rates(word, month)
            #if rate >best_rate:
            for j in range(len(best_three)):
                if rate > best_three[j][1]:
                    for z in range(len(best_three) - 1, j):
                        best_three[z] = best_three[z - 1].copy()
                    best_three[j] = [month, rate]
                    best_rate = rate
                    best_word = month
                #best_rate = rate
                #best_month = month
        best_three = [word[0].lower() for word in best_three if word[0] is not None]
        return best_three
        #return best_month.lower()

    def filter_words_month(self, word):
        start_w = ['А', 'Я', 'Ф', 'М', 'И', 'С', 'О', 'Н', 'Д']

        if len(word) > 8 or len(word) < 3:
            return False
        if word[0] not in start_w:
            return False
        if word[-1] not in start_w[:2]:
            return False
        #if word[:-2] not in ['РЯ', 'ЛЯ','НЯ', 'ТА', 'АЯ']:
        #    return False

        for w in range(0, len(word)-2):
            count_gl = 0
            count_sogl = 0
            for j in word[w:w+3]:
                if j in self.gl:
                    count_gl +=1
                if j in self.sogl:
                    count_sogl += 1
            if count_sogl == 3 or count_gl == 3:
                return False
        return True

    def translate_cmp_month(self, word, new_word, months):
        if len(word) == 0:
            if not self.filter_words_month(new_word):
                return
            for ind, month in enumerate(months):
                rate = self.cmp_rates(new_word, month)
                if rate > 0.5:
                    self.count_month[ind] += 1
                #if rate > self.best_month_rate:
                for j in range(len(self.best_month)):
                    if rate > self.best_month[j][1]:
                        for z in range(len(self.best_month)-1, j):
                            self.best_month[z] = self.best_month[z-1].copy()
                        self.best_month[j] = [month, rate]
                        break
                    #self.best_month_rate = rate
                    #self.best_month = month
            return
        if len(word) != 1:
            if word[0:2] in self.alphabet.keys():
                start_w = word[0:2]
                another_word = word[2:]
                for w in self.alphabet[start_w]:
                    self.translate_cmp_month(another_word, new_word+w, months)
        start_w = word[0]
        another_word = word[1:]
        for w in self.alphabet[start_w]:
            self.translate_cmp_month(another_word, new_word+w, months)
        return

    def findRegNumbDataGiven(self, reg_number, data):

        #ПОМЕНЯТЬ ФИЛЬТР ДЛЯ РЕГ НОМЕРА и ДАТЫ ВЫДАЧИ
        for i in range(len(reg_number)):
            reg_number[i] = self.filterForRegNum(reg_number[i])
            #reg_number[i] = self.filt_words(reg_number[i])
        reg_num = reg_number[0]
        if reg_number[0] == '':
            reg_num = reg_number[1]

        year = 'года'
        cmp_month = ['января', 'февраля', 'марта', 'апреля','мая',
                     'июня', 'июля', 'августа', 'сентября', 'октября',
                     'ноября', 'декабря']
        for d in range(len(cmp_month)):
            cmp_month.append(cmp_month[d].upper())
        cmp_month = cmp_month[len(cmp_month)//2:]

        i = 0
        data_length = len(data)
        while i < data_length:
            word_0 = self.filterForRegNum(data[i][0])
            word_1 = self.filterForRegNum(data[i][1])
            data[i][0] = word_0
            data[i][1] = word_1
            #data[i][0] = self.filt_words(data[i][0])
            #data[i][1] = self.filt_words(data[i][1])
            d1 = self.checkfordigits(data[i][0])
            d2 = self.checkfordigits(data[i][1])
            '''if i-1 > -1:
                if self.check_digit(data[i-1][0]):
                    if len(d1) < len(d2):
                        if self.check_digit(d2[0]):
                            del d2[0]
                    elif len(d1) > len(d2):
                        if self.check_digit(d1[0]):
                            del d1[0]'''

            if len(d1) < len(d2):

                d1 = d1+['']*(len(d2)-len(d1))
            elif len(d1) > len(d2):
                d2 = d2+['']*(len(d1) - len(d2))
            new_d = []
            for j in range(len(d1)):
                new_d.append([d1[j], d2[j]])
            data = data[:i]+new_d+data[i+1:]
            t = len(new_d)
            if t == 0:
                t = 1
            i+=t
            data_length = len(data)
        date = []
        data = data[:-1]

        for ind, word in enumerate(data):
            if self.check_digit(word[0]):
                if ind ==2:
                    if len(word[0]) > 4:
                        date.append(word[0][:4])
                    else:
                        date.append(word[0])
                elif ind == 0:
                    if word[0][0] == '3' and word[0] > '31':
                        date.append('31')
                    elif len(word[0]) > 2:
                        date.append(word[0][:2])
                    else:
                        date.append(word[0])
                else:
                    date.append(word[0])
            elif self.check_digit(word[1]):
                date.append(word[1])
            elif word[1] != '':
                cmp_month = [month.lower().capitalize() for month in cmp_month]
                months = self.__check_best_word(cmp_month, word[1].lower().capitalize())
                months = [month.lower() for month in months]
                date.append(months)
                #date.append(self.cmp_lv_date(word[1], cmp_month))
            elif word[1] == '' and word[0] != '':
                best_rate = 0
                best_months = cmp_month[0]
                months = self.translate_cmp_month(word[0], '', cmp_month)
                for i in range(len(self.best_month)-1, -1):
                    if self.best_month[i][0] in [month[0] for month in self.best_month[:i]]:
                        del self.best_month[i]
                months = [month[0].lower() for month in self.best_month if month[0] is not None]
                print(self.count_month)
                date.append(months)
                ''' words = self.find_all_words(word)
                for month in cmp_month:
                    #for
                    words = self.find_all_words(word)
                    rate, month = self.translate_cmp_month(word, month,
                first_string = self.find_string(word[0])'''
                #date.append(self.cmp_lv_date(first_string, cmp_month))
            else:
                date.append('')
        date.append('года')
        full_dates = []
        i = 0
        l = len(date)
        while i<l:
            if type(date[i]) != str:
                if type(date[i]) == list and len(date[i]) > 1:
                    i+=1
                else:
                    del date[i]
                    l = len(date)
            else:
                i+=1
        if type(date[1]) != list:
            date[1] = ['января', 'июня', 'ноября']
        for month in date[1]:
            full_dates.append(date[0]+' '+month+' '+date[2]+' '+date[3])
        #full_date = ' '.join(date)

        return reg_num, full_dates
        #for date in data:
        #    date[0]

    def find_name_part(self, word, bd):
        best_rate = self.cmp_rates(word, bd[0])
        best_word = bd[0]
        for word_true in bd:
            rate = self.cmp_rates(word, word_true)
            if rate > best_rate:
                best_rate = rate
                best_word = word_true
        return best_word

    def findFIO(self, fio):
        fio_bd = []
        for i in range(len(fio)):
            word = fio[i][1]
            if word == '':
                word = fio[i][2]
                word = word.upper()
                all_words = self.check_before_translate(word, size = len(word))
                #all_words = self.translate_word(word, words=[])
                #all_words = self.filt_words(all_words)
                first_string = lv.quickmedian(all_words)
                '''dict = enchant.Dict('ru_RU')
                all_enchant_words = []
                for word in all_words:
                    true_w = dict.suggest(word)
                    all_enchant_words += true_w
                    # all_enchant_words.append(true_w)
                first_string = lv.quickmedian(all_enchant_words)'''
                word = self.filterForRegNum(first_string)
                word = word.lower().capitalize()
            else:
                word = self.filterForRegNum(word)
                word = word.lower().capitalize()
            #next_symb_ind = self.rus_alphabet_upper.find(fio[i][0])+1
            #next_symb = self.rus_alphabet_upper[next_symb_ind]
            if None in self.begin_fio_bd[word[0]]:
                if i == 0:
                    word = self.__check_best_word(self.families, word)
                if i == 1:
                    word = self.__check_best_word(self.names, word)
                if i == 2:
                    word = self.__check_best_word(self.surnames, word)
            else:
                if i == 0:
                    word = self.__check_best_word(self.families[self.begin_fio_bd[word[0]][i][0]: self.begin_fio_bd[word[0]][i][1]], word)
                if i == 1:
                    word = self.__check_best_word(self.names[self.begin_fio_bd[word[0]][i][0]: self.begin_fio_bd[word[0]][i][1]], word)
                if i == 2:
                    word = self.__check_best_word(self.surnames[self.begin_fio_bd[word[0]][i][0]: self.begin_fio_bd[word[0]][i][1]], word)
            fio_bd.append(word)
        return fio_bd

    def findSpecNumber(self, number):
        num_1 = None
        num_1_spec = None
        num_2 = None
        num_2_spec = None
        if number[0] != '':
            number[0] = self.clean_number(number[0])
            if number[0] in self.bach_number:
                num_1 = number[0]
                num_1_spec = self.bach_name[np.where(num_1 == self.bach_number)]
            elif number[0] in self.mag_number:
                num_1 = number[0]
                num_1_spec = self.mag_name[np.where(num_1 == self.mag_number)]
            elif number[0] in self.spec_number:
                num_1 = number[0]
                num_1_spec = self.spec_name[np.where(num_1 == self.spec_number)]
        if number[1] != '':
            num = self.clean_number(number[1])
            number[1] = num
            if number[1] in self.bach_number:
                num_2 = number[1]
                num_2_spec = self.bach_name[np.where(num_2 == self.bach_number)]
            elif number[1] in self.mag_number:
                num_2 = number[1]
                num_2_spec = self.mag_name[np.where(num_2 == self.mag_number)]
            elif number[1] in self.spec_number:
                num_2 = number[1]
                num_2_spec = self.spec_name[np.where(num_2 == self.spec_number)]
        if num_1 != None:
            return num_1, num_1_spec
        if num_1 == None and num_2 != None:
            return num_2, num_2_spec
        else:
            if number[1] != '' and number[1] != None:
                return number[1], ''
        return '', ''

    def clean_number(self, number):
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

    def findSpecialName(self, name, level):
        word1 = ''
        for w in name:
            word1 += w[1]+' '
        word1 = word1.strip(' ')
        data = None
        if level != '':
            if level == 'бакалавр':
                data = self.bach_name.copy()
            if level == 'магистр':
                data = self.mag_name.copy()
            if level == 'специалитет':
                data = self.spec_name.copy()
        else:
            data = np.unique(np.append(np.append(self.bach_name, self.mag_name), self.spec_name)).copy()

        if word1 != '':
            all_names = self.__check_best_word(data, word1)
        else:
            words = []
            for word in name:
                all_words = self.check_before_translate(word[0], size=12)
                if len(all_words)>100000:
                    all_words = list(np.random.choice(all_words, size = 100000//len(all_words[0])))
                all_words = self.filt_words(all_words)

                poss_word = lv.quickmedian(all_words)
                words.append(poss_word)
            poss_word = ' '.join(words)
            all_names = self.__check_best_word(data, poss_word)
        return all_names

    def findSpeciality(self, number, name, level):
        confidence = True
        if '.' not in number[0] and '.' not in number[1]:
            confidence = False
        number, prev_name = self.findSpecNumber(number)
        if prev_name == '' or confidence == False:
            confidence = False
            prev_name = self.findSpecialName(name, level)
        return number, prev_name, confidence

    def findCmpLevel(self, level, cmp_level):
        #level = level.lower()
        word_cmp = level[1]
        if word_cmp == '':
            word_cmp = level[0].upper()
            all_words = self.check_before_translate(word_cmp, size=3)
            # all_words = self.translate_word(word, words=[])
            all_words = self.filt_words(all_words)
            word_cmp = lv.quickmedian(all_words)
            word_cmp = word_cmp.lower().capitalize()
        #print(level)
        specs = ['бакалавр', 'магистр', 'специалитет']
        level = self.__check_best_word(specs, word_cmp)
        #print(level)
        #level = self.cmp_lv_date(level, specs)
        return level[0].lower()

    def findQuality(self, word, special_number):
        word_cmp = word[1]
        if word_cmp == '':
            word_cmp = word[0].upper()
            all_words = self.check_before_translate(word_cmp, size=15)
            # all_words = self.translate_word(word, words=[])
            all_words = self.filt_words(all_words)
            word_cmp = lv.quickmedian(all_words)

        if special_number !='':
            poss_qualities = self.spec_quality[np.where(self.spec_number == special_number)].copy()
            poss_qualities = poss_qualities.split('.')
            best_word = self.__check_best_word(poss_qualities, word_cmp)
        else:
            best_rate = self.cmp_rates(word_cmp, '')
            best_word = ''
            for all_words in self.spec_quality:
                words = all_words.split('.')
                for word_valid in words:
                    rate = self.cmp_rates(word_cmp, word_valid)
                    if rate>best_rate:
                        best_rate = rate
                        best_word = word_valid

            #best_word = self.__check_best_word(self.spec_quality, word_cmp)

        if best_word == '' or best_word == [] or best_word == None:
            best_word = ''

        return best_word


    """def compare_strings(self, new_string):
        # Предобработка строки
        new_string = self.proccess_string(new_string)


        return new_string
        new_string = new_string.split(' ')
        #return new_string
        for (ind,word) in enumerate(new_string):
            max_score = 0
            best_word = self.special_words[0]
            for spec in self.special_words:
                score = lv.ratio(word, spec)
                if score>max_score:
                    max_score = score
                    best_word = spec

            if max_score < 0.4:
                bd = None
                bd_word = None
                max_score_bd = 0
                for my_bd in self.my_bd:
                    if max_score_bd == 1:
                        break
                    else:
                        bd, bd_word, max_score_bd = self.find_bd_word(word, my_bd, max_score_bd, bd, bd_word)

                if max_score_bd > 0.5:
                    best_word = bd_word

                else:
                    d = self.dict.suggest(word)
                    print('Dict suggest: ', d)
                    if len(d) != 0:
                        opcodes = []
                        all_opcodes = {}
                        for d_word in d:
                            op = len(lv.opcodes(d_word, word))
                            opcodes.append(op)
                            if op in all_opcodes.keys():
                                all_opcodes[op].append(d_word)
                            else:
                                all_opcodes[op] = [d_word]
                        op = min(opcodes)
                        best_word = lv.quickmedian(all_opcodes[op])
                    else:
                        best_word = word
            new_string[ind] = best_word
        new_string = ''.join(new_string)

        print("Find word: ", new_string)
        return new_string
    """

    """"@classmethod
    def longest_substring(cls, str1, str2, m, n):

        lcSuff = [[0 for k in range(n+1)] for l in range(m+1)]

        result = 0
        substr = ''
        for i in range(m+1):
            for j in range(n+1):
                if (i==0 or j==0):
                    lcSuff[i][j] = 0
                elif (str1[i-1] == str2[j-1]):
                    substr += str1[i-1]
                    lcSuff[i][j] = lcSuff[i-1][j-1]+1
                    result = max(result, lcSuff[i][j])
                else:
                    substr = ''
                    lcSuff[i][j] = 0
        return substr
    """
    """
    def parse_words(self, words):
        new_words = []
        for (ind, word1) in enumerate(words):

            w_length = len(word1)
            j = 0
            while j < w_length:
                if word1[j] not in self.rus_alphabet  and word1[j] not in self.rus_alphabet_upper and word1[j] not in self.spec_symbols and not word1[j].isdigit():
                    word1 = word1[0:j]+word1[j+1:]
                    w_length = len(word1)
                    continue
                elif word1[j] in self.rus_alphabet_upper:
                    word1 = word1[0:j]+' '+word1[j:]
                    w_length += 1
                    j+=1
                j+=1
            for i in self.spec_symbols:
                word1 = word1.replace(i, ' ')
            words = word1.split(' ')
            words = list(filter(None, words))

            #new_words1 = []
            new_words += words
        return new_words
    
    """
    """
    def find_special_words(self, words):
        i = 0
        w_length = len(words)
        while i < w_length:
            word_to_find = ''
            for spec_word in self.special_words:
                if spec_word in words[i]:
                    word_to_find = spec_word
                if spec_word.upper() in words[i]:
                    word_to_find = spec_word.upper()
                if spec_word.capitalize() in words[i]:
                    word_to_find = spec_word.capitalize()
                if word_to_find != '':
                    ind = words[i].find(word_to_find)
                    new_words = [words[i][0:ind], words[i][ind:ind+len(word_to_find)], words[i][ind+ind+len(word_to_find):]]
                    new_words = [word for word in new_words if word != '']
                    ind_find = 0
                    for j in range(len(new_words)):
                        if word_to_find == new_words[j]:
                            ind_find = j
                            break
                    words = words[:i]+new_words+words[i+1:]
                    i += ind_find
                    w_length = len(words)
                    break
            i+=1
        return words
    """
    """
    def clean_string(self, words):

        words = self.parse_words(words)
        print('new words: ', words)
        i = 0
        w_length = len(words)
        while i < w_length:
            j = i+1

            while j<w_length:
                if len(words[i])>len(words[j]):
                    if words[j] in words[i]:
                        del words[j]
                        w_length -= 1
                    else:
                        j+=1
                else:
                    if words[i] in words[j]:
                        del words[i]
                        w_length -= 1
                    else:
                        j+=1
            i+=1
        print('filter words: ', words)
        words = self.find_special_words(words)

        print('special words: ', words)
        return words

        for (ind, word1) in words:
            l = len(words)
            new_word = []
            for i in range(ind+1, l):
                substr = FindWord.longest_substring(word1, words[i], len(word1),len(words[i]))
                if len(substr) == '':
                    break
                else:
                    index1 = word1.find(substr)
                    index2 = words[i].find(substr)
                    new_1 = [word1[0:index1], word1[index1:]]
                    new_2 = [words[i][:index2], words[i][index2+len(substr):]]
                    new_word = new_1+new_2
                    ll = len(new_word)
                    for j in range(ll):
                        if len(new_word[j]) < 2:
                            del new_word[j]
                            j-=1
                            ll= len(new_word)
                    del words
    """