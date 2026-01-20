import torch 
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import mlflow 
import sqlite3
import pandas as pd


#enable auto logging and set tracking uri and experiment name
mlflow.pytorch.autolog()
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("my-third-experiment")

#Create our facotrization model

class FactorizationModel(nn.Module):
    def __init__(self,num_users,num_items,num_factors):
        super(FactorizationModel,self).__init__()
        #creates our smaller matrices, intialized randomly and doesnt follow traditioanal tensor initialization
        self.U = nn.Embedding(num_users,num_factors) # 3x2
        self.V = nn.Embedding(num_items,num_factors) # 2x5


    def forward(self,user_indices,item_indices):
        #this is how our predictions are computed
        user_factors = self.U(user_indices)
        item_factors = self.V(item_indices)
        #dot product
        return (user_factors * item_factors).sum(1)


#now we need to extract our data and train the model
#we can query the database to get our user item interactions
#When we extract the data, we need to convert to user indices and item indices to feed into our model
#We can convert it into a pandas df and then use .ascategory.cat.codes to get indices 

#we connect a cursor to the database that will allow us to execute queries

conn  = sqlite3.connect("../main.db")

query = "SELECT user_id,isbn,rating FROM ratings"

df = pd.read_sql_query(query,conn)

print(df.head())
#convert to indices

df["user_idx"] = df["user_id"]-1
df["item_idx"] = df["isbn"].astype("category").cat.codes

num_users = df["user_idx"].max() + 1
num_items = df["item_idx"].max() + 1

print(f"Number of users: {num_users}, Number of items: {num_items}")

#switching to integrated gpu
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("MPS is available! Using Apple GPU.")
else:
    device = torch.device("cpu")
    print("MPS not available, using CPU.")

#model parameters
params = {
    "num_factors":60,
    "lr":.01,
    "weight_decay":0.0001,
    "epochs":250,
    "batch_size":4096
}
mlflow.log_params(params)
#now that we have our data ready, we can train our model
model = FactorizationModel(num_users,num_items,params["num_factors"])
model.to(device)
#loss function and optimizer
loss_fn  = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(),lr=params["lr"],weight_decay=params["weight_decay"])
#convert data to torch tensors
user_indices = torch.tensor(df["user_idx"].values,dtype=torch.long)
item_indices = torch.tensor(df["item_idx"].values,dtype=torch.long)
ratings = torch.tensor(df["rating"].values,dtype=torch.float)

#we need to create a dataset and a dataloader for batching since we are dealing with a large dataset
dataset = TensorDataset(user_indices,item_indices,ratings)
dataloader = DataLoader(dataset,batch_size = params["batch_size"], shuffle = True)


#training loop that trains per batch
for iter in range(params["epochs"]):
    train_loss = 0.0
    for _, (batch_user_indices,batch_item_indices,batch_ratings) in enumerate(dataloader):
        batch_user_indices = batch_user_indices.to(device)
        batch_item_indices = batch_item_indices.to(device)
        batch_ratings = batch_ratings.to(device)
        model.train()
        #clear out gradients
        optimizer.zero_grad()

        #forward pass
        preds = model(batch_user_indices,batch_item_indices)
        loss = loss_fn(preds,batch_ratings)
        #backward pass
        loss.backward()
        optimizer.step()

        train_loss += loss.item() 
    
    avg_loss = train_loss / len(dataloader)
    mlflow.log_metric("train_loss",avg_loss,step=iter)
    if (iter+1) % 10 == 0:
        print(f"Iteration {iter+1}/{params['epochs']}, Loss: {avg_loss:.4f}")

mlflow.pytorch.log_model(model,registered_model_name = f"factorization_model_ with_{params['num_factors']}_factors")
print("Model training complete and logged to MLflow.")
