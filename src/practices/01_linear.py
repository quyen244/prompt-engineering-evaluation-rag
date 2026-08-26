import mlflow
from sklearn.linear_model import LinearRegression
from src.practices.utils import get_linear_dataset , LrEvaluationResult , evaluate_lr
from dotenv import load_dotenv


load_dotenv()

print(mlflow.get_tracking_uri())

# Enable auto-logging for sklearn
mlflow.sklearn.autolog()

# Data
X_train, X_test, y_train, y_test = get_linear_dataset(1000, 100)

# Experiment
mlflow.set_experiment('lesson-01-linear-regression')

with mlflow.start_run(run_name='lesson-01-linear-regression'):
    lr = LinearRegression()
    lr.fit(X_train, y_train)  # Auto-logged!
    
    y_predict = lr.predict(X_test)
    
    # Custom evaluation (not auto-logged)
    results = evaluate_lr(y_predict, y_test)
    
    # Log custom metrics manually
    if hasattr(results, 'rmse'):
        mlflow.log_metric("custom_rmse", results.rmse)

    if hasattr(results, 'mse'):
        mlflow.log_metric("custom_mse", results.mse)

    if hasattr(results, 'mae'):
        mlflow.log_metric("custom_mae", results.mae)
    
    print(f"✅ Run ID: {mlflow.active_run().info.run_id}")


    


# Khi đăng ký model lần đầu, bạn có thể set stage luôn
# model_version = mlflow.register_model(
#     model_uri=f"runs:/{run_id}/model",
#     name="lr-v1"
# )

