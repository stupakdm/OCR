from __future__ import annotations

import numpy as np
import pandas as pd

import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from psycopg2 import Error

from enum import Enum
from typing import List, Optional

from sqlalchemy import create_engine, Table, Integer, String, Column
from sqlalchemy.orm import Session
from sqlalchemy import select
from sqlalchemy.ext.declarative import declarative_base
from pydantic import BaseModel, Field, constr

Base = declarative_base()


class Bachelor(Base):
    __tablename__ = 'bachelor'

    id = Column(Integer, primary_key=True)
    number = Column(String(20), nullable=False)
    name = Column(String(50), nullable=False)

    def __repr__(self) -> str:
        return f"Bachelor(id={self.id!r}, number={self.number!r}, name={self.name!r}"

class Magistr(Base):
    __tablename__ = 'magistr'

    id = Column(Integer, primary_key=True)
    number = Column(String(20), nullable=False)
    name = Column(String(50), nullable=False)

    def __repr__(self) -> str:
        return f"Magistr(id={self.id!r}, number={self.number!r}, name={self.name!r}"
    
class Speciality(Base):
    __tablename__ = 'speciality'

    id = Column(Integer, primary_key=True)
    number = Column(String(20), nullable=False)
    name = Column(String(50), nullable=False)
    quality = Column(String(100))

    def __repr__(self) -> str:
        return f"Speciality(id={self.id!r}, number={self.number!r}, name={self.name!r}, quality={self.quality!r}"
    
class University(Base):
    __tablename__ = 'university'

    id = Column(Integer, primary_key=True)
    fullname = Column(String(200), nullable=False)
    adress = Column(String(200), nullable=False)

    def __repr__(self) -> str:
        return f"University(id={self.id!r}, full name={self.fullname!r}, address={self.adress!r}"

class Namessurnames(Base):
    __tablename__ = 'namessurnames'

    id = Column(Integer, primary_key=True)
    name = Column(String(50), nullable=False)
    surname = Column(String(50), nullable=False)
    priority = Column(Integer, nullable=False)

    def __repr__(self) -> str:
        return f"Names_Surnames(id={self.id!r}, number={self.name!r}, name={self.surname!r}, priority={self.priority!r}"

class Families(Base):
    __tablename__ = 'families'

    id = Column(Integer, primary_key=True)
    family = Column(String(60), nullable=False)

    def __repr__(self) -> str:
        return f"Families(id={self.id!r}, family={self.family!r}"
    
"""
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

def get_values():
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
"""

def create_table():
    engine = create_engine("postgresql+psycopg2://postgres:1111@localhost/sqlalchemy_diplom")
    Base.metadata.create_all(engine)

def append_data():
    engine = create_engine("postgresql+psycopg2://postgres:1111@localhost/sqlalchemy_diplom")

    with Session(engine) as session:
        bach = pd.read_csv(f'Bachelor.csv', encoding='utf-8', sep=';')
        all_list = []
        for i, row in bach.iterrows():
            item = Bachelor(
                number=row['number'],
                name=row['name'],
            )
            all_list.append(item.copy())

        mag = pd.read_csv(f'Magistr.csv', encoding='utf-8', sep=';')
        for i, row in mag.iterrows():
            item = Magistr(
                number=row['number'],
                name=row['name'],
            )
            all_list.append(item.copy())

        spec_all = pd.read_csv(f'special_quality.csv', encoding='utf-8', sep=';')
        spec_all['quality'].fillna('', inplace=True)
        for i, row in spec_all.iterrows():
            item = Speciality(
                number=row['number'],
                name=row['name'],
                quality=row['quality'],
            )
            all_list.append(item.copy())

        universities = pd.read_csv(f'Университеты России.csv', encoding='utf-8', sep=',')
        for i, row in universities.iterrows():
            item = University(
                number=row['number'],
                fullname=row['Full Name'],
                adress=row['Adress'],
            )
            all_list.append(item.copy())

        df_name = pd.read_csv(f'Names_and_surnames2.csv', encoding='utf-8', sep=',')
        for i, row in df_name.iterrows():
            item = Namessurnames(
                name=row['name'],
                surname=row['surname'],
                priority=row['priority'],
            )
            all_list.append(item.copy())

        df_families = pd.read_csv(f'families3.csv', encoding='utf-8')
        for i, row in df_families.iterrows():
            item = Namessurnames(
                family = row['family'],
            )
            all_list.append(item.copy())

        session.add_all(all_list)

        session.commit()

def get_data(model = None):
    engine = create_engine("postgresql+psycopg2://postgres:1111@localhost/sqlalchemy_diplom")

    if model != None:
        session = Session(engine)
        result = session.execute(select(model))
        session.close()
        return result
    else:
        return None

if __name__ == '__main__':
    pass
