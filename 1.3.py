import pandas as pd
from sklearn.model_selection import train_test_split
import math
import numpy as np

df = pd.read_csv("penguins.csv")

df = df.dropna()

labels = df["species"]
features = df["bill_length_mm"]
features = pd.concat([features, df["bill_depth_mm"]], axis=1)

species_dict = {
    "Adelia": 1,
    "Gentoo": 2,
    "Chinstrap": 3,
}

island_dict = {
    "Biscoe": 1,
    "Dream": 2,
    "Torgersen": 3
}

sex_dict = {
    "MALE": 0,
    "FEMALE": 1,
}

df["species"] = df["species"].map(species_dict).astype(np.float64)
df["island"] = df["island"].map(island_dict).astype(np.float64)
df["sex"] = df["sex"].map(sex_dict).astype(np.float64)

train_features, test_features, train_labels, test_labels = train_test_split(
    features, labels, test_size=0.2, random_state=123)

train_features = train_features.values
test_features = test_features.values
train_labels = train_labels.values
test_labels = test_labels.values


def euclidean_distance(point1, point2):
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)


def knn_predict_uniform(train_features, train_labels, test_point, k):
    distances = []

    for i in range(len(train_features)):
        dist = euclidean_distance(train_features[i], test_point)
        distances.append((dist, train_labels[i]))

    distances.sort(key=lambda x: x[0])
    nearest_neighbors = distances[:k]

    votes = {}
    for _, label in nearest_neighbors:
        votes[label] = votes.get(label, 0) + 1

    return max(votes.items(), key=lambda x: x[1])[0]


def knn_predict_distance(train_features, train_labels, test_point, k):
    distances = []

    for i in range(len(train_features)):
        dist = euclidean_distance(train_features[i], test_point)
        distances.append((dist, train_labels[i]))

    distances.sort(key=lambda x: x[0])
    nearest_neighbors = distances[:k]

    votes = {}
    for dist, label in nearest_neighbors:
        weight = 1 / (dist + 1e-10)
        votes[label] = votes.get(label, 0) + weight

    return max(votes.items(), key=lambda x: x[1])[0]


def calculate_accuracy(predictions, true_labels):
    correct = sum(1 for pred, true in zip(predictions, true_labels) if pred == true)
    return correct / len(true_labels)


best_accuracy = 0
worst_accuracy = 1

for n_neighbors in range(1, 11):
    for weights_type in ['uniform', 'distance']:
        predictions = []

        for test_point in test_features:
            if weights_type == 'uniform':
                pred = knn_predict_uniform(train_features, train_labels, test_point, n_neighbors)
            else:  # 'distance'
                pred = knn_predict_distance(train_features, train_labels, test_point, n_neighbors)
            predictions.append(pred)

        accuracy = calculate_accuracy(predictions, test_labels)

        if accuracy > best_accuracy:
            best_accuracy = accuracy
        if accuracy < worst_accuracy:
            worst_accuracy = accuracy

print('Best accuracy: {:.6f}'.format(best_accuracy))
print('Worst accuracy: {:.6f}'.format(worst_accuracy))