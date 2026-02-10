import numpy as np

target = np.array(list(map(float, input().split(" "))))
predict = np.array(list(map(float, input().split(" "))))


def determ(target, predict):
    y = target.mean()
    r = 1 - (((target - predict) ** 2).sum() / ((y - predict) ** 2).sum())
    return r


print(f"R2: {determ(target, predict):.2f}")