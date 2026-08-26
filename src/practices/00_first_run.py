import mlflow 
from src.config import Config
from dotenv import load_dotenv

load_dotenv()

print(f"Tracking URI: {mlflow.get_tracking_uri()}")

mlflow.set_experiment('lesson-01-first-run')

with mlflow.start_run(run_name='my-run'):
    mlflow.log_param('alpha', 0.1)
    mlflow.log_metric('rmse' , 0.42)
    # mlflow.log_artifact('plot.png')


    # mlflow.sklearn.log_model(model , 'model')``