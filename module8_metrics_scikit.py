# module8_metrics_scikit.py

import numpy as np
from sklearn.metrics import precision_score, recall_score


def main():
    N = int(input("Enter N: "))

    true_labels = np.zeros(N, dtype=int)
    predicted_labels = np.zeros(N, dtype=int)

    for i in range(N):
        x = int(input("Enter x value: "))
        y = int(input("Enter y value: "))

        true_labels[i] = x
        predicted_labels[i] = y

    precision = precision_score(true_labels, predicted_labels, zero_division=0)
    recall = recall_score(true_labels, predicted_labels, zero_division=0)

    print("Precision:", precision)
    print("Recall:", recall)


if __name__ == "__main__":
    main()
