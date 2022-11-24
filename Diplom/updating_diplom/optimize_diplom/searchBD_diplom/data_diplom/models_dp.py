from sqlalchemy import Integer, String, Column

from sqlalchemy.ext.declarative import declarative_base
Base = declarative_base()


class Bachelor(Base):
    __tablename__ = 'bachelor'

    id = Column(Integer, primary_key=True)
    number = Column(String(10), nullable=False)
    name = Column(String(150), nullable=False)

    def __repr__(self) -> str:
        return f"Bachelor(id={self.id!r}, number={self.number!r}, name={self.name!r}"


class Magistr(Base):
    __tablename__ = 'magistr'

    id = Column(Integer, primary_key=True)
    number = Column(String(10), nullable=False)
    name = Column(String(150), nullable=False)

    def __repr__(self) -> str:
        return f"Magistr(id={self.id!r}, number={self.number!r}, name={self.name!r}"

class Speciality(Base):
    __tablename__ = 'speciality'

    id = Column(Integer, primary_key=True)
    number = Column(String(10), nullable=False)
    name = Column(String(150), nullable=False)
    quality = Column(String)

    def __repr__(self) -> str:
        return f"Speciality(id={self.id!r}, number={self.number!r}, name={self.name!r}, quality={self.quality!r}"


class University(Base):
    __tablename__ = 'university'

    id = Column(Integer, primary_key=True)
    fullname = Column(String(350), nullable=False)
    adress = Column(String(300), nullable=False)

    def __repr__(self) -> str:
        return f"University(id={self.id!r}, full name={self.fullname!r}, address={self.adress!r}"


class Namessurnames(Base):
    __tablename__ = 'namessurnames'

    id = Column(Integer, primary_key=True)
    name = Column(String(50), nullable=False)
    surname = Column(String(50), nullable=False)
    priority = Column(Integer, nullable=False)

    def __repr__(self) -> str:
        return f"Names_Surnames(id={self.id!r}, name={self.name!r}, surname={self.surname!r}, priority={self.priority!r}"


class Families(Base):
    __tablename__ = 'families'

    id = Column(Integer, primary_key=True)
    family = Column(String(60), nullable=False)

    def __repr__(self) -> str:
        return f"Families(id={self.id!r}, family={self.family!r}"