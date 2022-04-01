import pandas as pd
import numpy as np
import Levenshtein as lv

class Search:
    special_words = ('имя', 'фамилия', 'отчество', 'дата', 'рождения', 'место')

    def start(self):
        #self.name_filepath = 'templates/Names and surnames.csv'
        #self.fams_filepath = 'templates/families2.csv'
        self.df_name = pd.read_csv('templates/Names and surnames.csv', encoding='utf-8')
        self.names = np.array(self.df_name['name'])
        self.surnames = np.array(self.df_name['surname'])
        self.df_families = pd.read_csv('templates/families2.csv', encoding = 'utf-8')
        self.families = np.array(self.df_families['family'])

    def __check__(self, text, param):               # lv.ratio - compute similarity of two strings
        lev_dist = lv.distance(text, param[0])      # lv.median - делает медианное значение из нескольких вариаций распознанных слов (также median_improve, quickmedian, setmedian)
        corr = param[0]
        for i in param:
            lev = lv.distance(text, i)
            if lev < lev_dist:
                lev_dist = lev
                corr = i
        if lev_dist <= 2:
            return corr
        else:
            return text

    def look_name(self, name):
        return self.__check__(name, self.names)

    def look_surname(self, surname):
        return self.__check__(surname, self.surnames)

    def look_family(self, family):
        return self.__check__(family, self.families)

    def look_spec_words(self, text):
        return self.__check__(text, self.special_words)
        #ham_dist = lv.hamming(name, 'ham')


