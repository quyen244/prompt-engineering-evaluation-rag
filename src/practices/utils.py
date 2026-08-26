from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from sklearn.metrics import mean_squared_error , mean_absolute_error
import math 

def get_linear_dataset(n_samples : int , n_features : int, test_size : float = 0.2):
    X , y = make_regression(n_samples=n_samples,
                           n_features=n_features)

    X_train , X_test , y_train , y_test = train_test_split(X , y , test_size=test_size , shuffle= True)

    return  X_train , X_test , y_train , y_test

@dataclass
class LrEvaluationResult:
    mse : float 
    rmse : float 
    mae : float

def evaluate_lr(y_predict , y_test) -> LrEvaluationResult:
    mse = mean_squared_error(y_test , y_predict)
    mae = mean_absolute_error(y_test , y_predict)
    rmse = math.sqrt(mse)
    return {
        mse , 
        rmse, 
        mae
    }


# if __name__ == '__main__':
#     try :
#          X_train , X_test , y_train , y_test = get_linear_dataset(1000 , 100)
#          print(X_train.shape)
#          print(X_train[0])
#     except Exception as e:
#         print(e)