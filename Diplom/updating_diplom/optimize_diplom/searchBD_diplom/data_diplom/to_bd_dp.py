from __future__ import annotations

import pandas as pd

import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
# Не исправлять: так правильно
from Diplom.updating_diplom.optimize_diplom.searchBD_diplom.data_diplom.models_dp import Bachelor, Magistr, Speciality, University, Namessurnames, Families, Base

url_server = 'postgresql+psycopg2://postgres:postgres@localhost/sqlalchemydiplom'
#directory = 'Diplom/Updating_dp/Optimize_dp/SearchBD_dp/data_diplom/'
directory = './'
#Base = declarative_base()
def create_db():
    connection = psycopg2.connect(user="postgres", password="postgres")
    connection.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)

    cursor = connection.cursor()
    cursor.execute('create database sqlalchemydiplom')
    cursor.close()
    connection.close()


def create_table():
    engine = create_engine(url_server)
    Base.metadata.create_all(engine)
    #Base.metadata.drop(Bachelor)


def connect_db():
    engine = create_engine(url_server)
    engine.connect()
    print(engine)

def drop_all_table():
    engine = create_engine(url_server)
    Base.metadata.drop_all(engine)



def append_data():
    engine = create_engine(url_server)
    print(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    #with Session(engine) as session:
    bach = pd.read_csv(f'{directory}Bachelor.csv', encoding='utf-8', sep=';')
    all_list = []
    for i, row in bach.iterrows():
        if row['number'][-2:] != '00':
            item = Bachelor(
                number=row['number'],
                name=row['name'],
            )
            all_list.append(item)

    mag = pd.read_csv(f'{directory}Magistr.csv', encoding='utf-8', sep=';')
    for i, row in mag.iterrows():
        if row['number'][-2:] != '00':
            item = Magistr(
                number=row['number'],
                name=row['name'],
            )
            all_list.append(item)

    spec_all = pd.read_csv(f'{directory}special_quality.csv', encoding='utf-8', sep=';')
    spec_all['quality'].fillna('', inplace=True)
    for i, row in spec_all.iterrows():
        if row['number'][-2:] != '00':
            item = Speciality(
                number=row['number'],
                name=row['name'],
                quality=row['quality'],
            )
            all_list.append(item)

    universities = pd.read_csv(f'{directory}Университеты России.csv', encoding='utf-8', sep=',')
    for i, row in universities.iterrows():
        item = University(
            fullname=row['Full Name'],
            adress=row['Adress'],
        )
        all_list.append(item)

    df_name = pd.read_csv(f'{directory}Names_and_surnames2.csv', encoding='utf-8', sep=',')
    for i, row in df_name.iterrows():
        if row['priority'] > 100:
            item = Namessurnames(
                name=row['name'],
                surname=row['surname'],
                priority=row['priority'],
            )
            all_list.append(item)

    df_families = pd.read_csv(f'{directory}families3.csv', encoding='utf-8')
    for i, row in df_families.iterrows():
        item = Families(
            family=row['family'],
        )
        all_list.append(item)

    session.add_all(all_list)

    session.commit()
    session.close()



def get_items():
    engine = create_engine(url_server)
    Session = sessionmaker(bind=engine)
    session = Session()
    models = [Bachelor, Magistr, Speciality, University, Namessurnames, Families]
    for model in models:
        print(model)
        all_values = []
        for values in model.__dict__.keys():
            if values[0] != '_' and values != 'id':
                all_values.append(values)
                #print(item.__dict__[values], end=' ')
        for item in session.query(model).all():
            for value in all_values:
                print(item.__dict__[value], end=' ')
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




if __name__ == '__main__':
    #create_db()
    drop_all_table()
    create_table()
    append_data()
    get_items()