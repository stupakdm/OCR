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


def create_table(engine_name = "postgresql+psycopg2://postgres:1111@localhost/sqlalchemy_diplom",):
    engine = create_engine(engine_name)
    Base.metadata.create_all(engine)

def append_data(engine_name = "postgresql+psycopg2://postgres:1111@localhost/sqlalchemy_diplom",):
    engine = create_engine(engine_name)

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

def get_data(engine_name = "postgresql+psycopg2://postgres:1111@localhost/sqlalchemy_diplom", model = None):
    engine = create_engine(engine_name)

    if model != None:
        session = Session(engine)
        result = session.execute(select(model))
        session.close()
        return result
    else:
        return None
