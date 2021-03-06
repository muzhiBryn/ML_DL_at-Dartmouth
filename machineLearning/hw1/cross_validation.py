import math
import pandas as pd


def my_cross_validation(model, X, y, k):
    performance = []
    # split the dataset
    size = len(y) // k

    # train and test the data
    for i in range(k):
        test_idx = range(i * size, min(len(y), (i + 1) * size))
        testX = X.loc[X.index.isin(test_idx)]
        trainX = X.loc[~X.index.isin(test_idx)]
        testy = y.iloc[y.index.isin(test_idx)].tolist()
        trainy = y.iloc[~y.index.isin(test_idx)]

        clf = model()
        clf.fit(trainX, trainy)
        predicted_y = clf.predict(testX)

        # calculate the F1 score
        # TP = actual class 1 in predict class1
        # FP = actual class 2 in predict class1
        # FN = actual class 1 in predict class2
        # F1 = 2TP/(2TP + FP + FN)
        TP = 0
        FP = 0
        FN = 0

        for i in range(len(testy)):
            if testy[i] == 1: #actual class 1
                if predicted_y[i] == 1: #predict class1
                    TP += 1
                else: #predict class2
                    FN += 1
            else: #actual class 2
                if predicted_y[i] == 1: #predict class1
                    FP += 1
        f1 = 2 * TP / (2 * TP + FP + FN)
        performance.append(f1)

    return performance


