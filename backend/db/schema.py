from sqlalchemy import create_engine,ForeignKey 
from sqlalchemy.orm import sessionmaker, DeclarativeBase,Mapped, mapped_column
from sqlalchemy import Column,Integer,String



DATABASE_URL = "sqlite:///main.db"

class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"

    id :  Mapped[int] = mapped_column(primary_key=True)
    location: Mapped[str] = mapped_column()
    age: Mapped[int] = mapped_column(nullable=True)


class Book(Base):
    __tablename__ = "books"

    isbn :  Mapped[str] = mapped_column(primary_key=True)
    title: Mapped[str] = mapped_column()
    author: Mapped[str] = mapped_column()
    year: Mapped[int] = mapped_column(nullable=True)

class Rating(Base):
    __tablename__ = "ratings"

    isbn: Mapped[str] = mapped_column(ForeignKey("books.isbn"), primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), primary_key=True)
    rating :Mapped[int] = mapped_column()
   


engine  = create_engine(DATABASE_URL) # talks to the database, creates connections
#reprenest a database session, used to talk with database, autocommit prevents accidental writes, autflush
#prevents syncing of objects with db
session_local = sessionmaker(autocommit=False, autoflush=False, bind=engine)  

Base.metadata.create_all(bind=engine)#creates all tables


def get_db():
    db = session_local()
    try:
        yield db #used in dependency injection and pauses execution until endpoint is done using it
    finally:
        db.close()

