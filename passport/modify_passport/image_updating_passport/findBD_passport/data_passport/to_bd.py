from __future__ import annotations

import pandas as pd

import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from passport.modify_passport.image_updating_passport.findBD_passport.data_passport.models import Namessurnames, Families, FMSUnit, Cities, Base

url_server = "postgresql+psycopg2://postgres:postgres@localhost/sqlalchemypassport"
directory = './'
# Создать Базу данных
def create_db():
    connection = psycopg2.connect(user="postgres", password="postgres")
    connection.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)

    cursor = connection.cursor()
    cursor.execute('create database sqlalchemypassport')
    cursor.close()
    connection.close()

# Создать все таблицы в БД
def create_table():
    engine = create_engine(url_server)
    Base.metadata.create_all(engine)

# Удалить все таблицы в БД
def drop_all_table():
    engine = create_engine(url_server)
    Base.metadata.drop_all(engine)

# Добавить данные в БД
def append_data():
    engine = create_engine(url_server)
    Session = sessionmaker(bind=engine)
    session = Session()
    #with Session(engine) as session:
    all_list = []


    df_name = pd.read_csv(f'{directory}Names_and_surnames2.csv', encoding='utf-8', sep=',')
    for i, row in df_name.iterrows():
        item = Namessurnames(
            name=row['name'],
            surname=row['surname'],
            priority=row['priority'],
        )
        all_list.append(item)

    df_families = pd.read_csv(f'{directory}families3.csv', encoding='utf-8')
    for i, row in df_families.iterrows():
        item = Families(
            family = row['family'],
        )
        all_list.append(item)

    df_fms = pd.read_csv(f'{directory}fms_unit.csv', sep=',', index_col=False, encoding='utf-8')
    for i, row in df_fms.iterrows():
        item = FMSUnit(
            code=row['code'],
            name = row['name'],
        )
        all_list.append(item)

    df_cities = pd.read_csv(f'{directory}RussianCities.tsv', sep='\t', index_col=False, encoding='utf-8')
    for i, row in df_cities.iterrows():
        item = Cities(
            name=row['name'],
            regionname = row['region_name'],
        )
        all_list.append(item)

    session.add_all(all_list)

    session.commit()
    session.close()

# Достать все данные из БД
def get_items():
    engine = create_engine(url_server)
    Session = sessionmaker(bind=engine)
    session = Session()
    models = [Namessurnames, Families, FMSUnit, Cities]
    for model in models:
        print(model)
        for item in session.query(model).all():
            for values in model.__dict__.keys():
                if values[0] != '_' and values != 'id':
                    print(item.__dict__[values], end=' ')
            print()
        print()

    session.close()

def get_item(model):
    engine = create_engine(url_server)
    Session = sessionmaker(bind=engine)
    session = Session()

    all_values = dict()
    for values in model.__dict__.keys():
        if values[0] != '_' and values != 'id':
            all_values[values] = []
            #print(item.__dict__[values], end=' ')
    for item in session.query(model).all():
        for value in all_values.keys():
            all_values[value].append(item.__dict__[value])
            #print(item.__dict__[value], end=' ')
        #print()
    #print()

    session.close()
    return all_values


'''if __name__ == '__main__':
    create_db()
    drop_all_table()
    create_table()
    append_data()
    get_items()'''
