import mlflow
from mlflow.tracking import MlflowClient
from dotenv import load_dotenv

load_dotenv()

client = MlflowClient()


model_name = 'lr-v1'
model_version = 1


# # Load version 1
# model = mlflow.sklearn.load_model("models:/lr-v1/1")

# print(model)

# load model from alias
model = mlflow.sklearn.load_model("models:/lr-v1@champion")

print(model)






