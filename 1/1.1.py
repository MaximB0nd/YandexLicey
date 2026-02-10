import numpy as np

target = np.array(list(map(float, input().split(" "))))
predict = np.array(list(map(float, input().split(" "))))


def MSE(target, prediction):
    mse = (target - prediction)**2
    mse = np.mean(mse)
    return mse


def MAE(target, prediction):
    mae = np.abs((target - prediction))
    mae = np.mean(mae)
    return mae


def RMSE(target, prediction):
    rmse = np.sqrt(np.mean((target - prediction)**2))
    return rmse


print(f"MSE: {MSE(target, predict):.2f}")
print(f"MAE: {MAE(target, predict):.2f}")
print(f"RMSE: {RMSE(target, predict):.2f}")