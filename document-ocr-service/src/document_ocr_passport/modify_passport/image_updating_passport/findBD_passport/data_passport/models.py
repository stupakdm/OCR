from sqlalchemy import Integer, String, Column
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

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
    name = Column(String(150), nullable=False)
    regionname = Column(String(75), nullable=False)

    def __repr__(self) -> str:
        return f"Cities(id={self.id!r}, name={self.name!r}, region_name={self.regionname!r}"

def get_item(model):
    url_server = "postgresql+psycopg2://postgres:postgres@localhost/sqlalchemypassport"
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
