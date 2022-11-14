from __future__ import annotations

import pandas as pd

from sqlalchemy import create_engine, Integer, String, Column
from sqlalchemy.orm import Session
from sqlalchemy import select
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()


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

class FMSUnit(Base):
    __tablename__ = 'fmsunit'

    id = Column(Integer, primary_key=True)
    code = Column(String(10), nullable=False)
    name = Column(String(200), nullable=False)

    def __repr__(self) -> str:
        return f"FMSUnit(id={self.id!r}, code={self.code!r}, name={self.name!r}"

class Cities(Base):
    __tablename__ = 'cities'

    id = Column(Integer, primary_key=True)
    name = Column(String(50), nullable=False)
    regionname = Column(String(75), nullable=False)

    def __repr__(self) -> str:
        return f"Cities(id={self.id!r}, name={self.name!r}, region_name={self.regionname!r}"



def create_table():
    engine = create_engine("postgresql+psycopg2://postgres:1111@localhost/sqlalchemy_passport")
    Base.metadata.create_all(engine)

def append_data():
    engine = create_engine("postgresql+psycopg2://postgres:1111@localhost/sqlalchemy_passport")

    with Session(engine) as session:
        all_list = []


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

        df_fms = pd.read_csv(f'fms_unit.csv', sep=',', index_col=False, encoding='utf-8')
        for i, row in df_fms.iterrows():
            item = FMSUnit(
                code=row['code'],
                name = row['name'],
            )
            all_list.append(item.copy())

        df_cities = pd.read_csv(f'RussianCities.tsv', sep='\t', index_col=False, encoding='utf-8')
        for i, row in df_cities.iterrows():
            item = Cities(
                name=row['name'],
                regionname = row['region_name'],
            )
            all_list.append(item.copy())

        session.add_all(all_list)

        session.commit()

def get_data(model = None):
    engine = create_engine("postgresql+psycopg2://postgres:1111@localhost/sqlalchemy_passport")

    if model != None:
        session = Session(engine)
        result = session.execute(select(model))
        session.close()
        return result
    else:
        return None


if __name__ == '__main__':
    pass
