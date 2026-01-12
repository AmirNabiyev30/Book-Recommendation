from pydantic import BaseModel

#Data Transfer Objects

#USERS
class UserCreate(BaseModel):
    Location:str
    Age:int

class UserRead(BaseModel):
    id:int
    Location:str
    Age:int
#BOOKS
class BookCreate(BaseModel):
    isbn:int
    title:str
    author:str
    year:int
class BookRead(BaseModel):
    isbn:int
    title:str
    author:str
    year:int
#RATINGS
class RatingCreate(BaseModel):
    isbn:int
    user_id:int
    rating:int
class RatingRead(BaseModel):
    isbn:int
    user_id:int
    rating:int
