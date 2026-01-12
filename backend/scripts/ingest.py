from db.schema import session_local,engine,Base,User,Book,Rating
import csv
from sqlalchemy.orm import Session


def ingest_users(db:Session,filepath:str):
    with open(filepath,'r',encoding = 'latin-1') as file:
        count = 0
        reader = csv.DictReader(file,delimiter=';')
        for row in reader:
            db.add(
                User(
                    id = int(row["User-ID"]),
                    location = row["Location"],
                    age = int(row["Age"]) if row["Age"].isdigit() else None
                )
            )
            count += 1
            if count %1000 == 0:
                db.commit()
        db.commit()
        print(f"Ingested {count} users")



def ingest_books(db:Session,filepath:str):
    with open(filepath,'r',encoding = 'latin-1') as file:
        count = 0
        reader = csv.DictReader(file,delimiter=';')
        for row in reader:
            db.add(
                Book(
                    isbn = row["ISBN"],
                    title = row["Book-Title"],
                    author = row["Book-Author"],
                    year = int(row["Year-Of-Publication"]) if row["Year-Of-Publication"].isdigit() else None
                )
            )
            count += 1
            if count %1000 == 0:
                db.commit()
        db.commit()
        print(f"Ingested {count} books")

def ingest_ratings(db:Session,filepath:str):
    with open(filepath,'r',encoding = 'latin-1') as file:
        count = 0
        reader = csv.DictReader(file,delimiter=';')
        for row in reader:
            db.add(
                Rating(
                    isbn = row["ISBN"],
                    user_id = int(row["User-ID"]),
                    rating = int(row["Book-Rating"])
                )
            )
            count += 1
            if count %1000 == 0:
                db.commit()
        db.commit()
        print(f"Ingested {count} ratings")

def main():
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    db = session_local()
    try:
        ingest_users(db,'./books_data/users.csv')
        ingest_books(db,'./books_data/books.csv')
        ingest_ratings(db,'./books_data/ratings.csv')
    except Exception as e:
        print(f"Error during ingestion: {e}")
    finally:
        db.close()


if __name__ == "__main__":
    main()
        
