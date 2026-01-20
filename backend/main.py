from fastapi import FastAPI
import json
import mlflow.pytorch
import torch


app = FastAPI()

#load our isbn to book id mapping
with open("./isbn_to_bookid_mapping.json","r") as f:
    ISBN_to_BOOKID = json.load(f)

#load the trained model
mlflow.set_tracking_uri('http://127.0.0.1:5000')
model_uri = f"models:/factorization_model_ with_60_factors/1"
#load model onto appropriate device at startup
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = mlflow.pytorch.load_model(model_uri,map_location = device)
model.eval()


@app.post("/")
def predict(isbn: str, user_id: int):
    book_id = ISBN_to_BOOKID.get(isbn)
    with torch.no_grad():
        user_tensor = torch.tensor([user_id - 1], device=device)
        item_tensor = torch.tensor([book_id], device=device)
        prediction = model(user_tensor, item_tensor)
        predicted_rating = prediction.item()
    return {"predicted_rating": predicted_rating}



    

