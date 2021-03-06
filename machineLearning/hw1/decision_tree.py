# %%
import pandas as pd
import random



# %%
def cal_gini(Y):
    res = 1;
    label_counts = Y.value_counts()
    for k in label_counts.index:
        res -= (label_counts[k] / len(Y)) ** 2
    return res


def find_mostly_occurred_label(trainY):
    max_count = 0;
    mostly_occurred_label = None
    for label, count in trainY.value_counts().items():
        if count > max_count:
            max_count = count
            mostly_occurred_label = label
    return mostly_occurred_label


class MyDecisionTree(object):
    def predict(self, X_one_row):
        pass

    @staticmethod
    def gini(Y):
        return cal_gini(Y)

    @staticmethod
    def fit(trainX, trainY, min_size, max_splits, nfeats):
        if len(trainX) < min_size or max_splits == 0:
            return MyDecisionLeaf.fit(trainX, trainY, min_size, max_splits, nfeats)
        else:
            return MyDecisionInner.fit(trainX, trainY, min_size, max_splits, nfeats)


class MyDecisionLeaf(MyDecisionTree):
    def __init__(self, label):
        self.label = label

    def predict(self, X_one_row):
        return self.label

    @staticmethod
    def fit(trainX, trainY, min_size, max_splits, nfeats):
        return MyDecisionLeaf(find_mostly_occurred_label(trainY))


class MyDecisionInner(MyDecisionTree):
    def __init__(self, feat, val, lt, ge):
        self.feat = feat
        self.val = val
        self.lt = lt
        self.ge = ge

    def predict(self, X_one_row):
        if X_one_row[self.feat] < self.val:
            return self.lt.predict(X_one_row)
        else:
            return self.ge.predict(X_one_row)

    @staticmethod
    def find_split(trainX, trainY, nfeats):
        total_feature_num = trainX.shape[1]
        if nfeats <= total_feature_num:
            features_to_test = sorted(random.sample(range(total_feature_num), nfeats))
        else:
            features_to_test = range(total_feature_num)

        res = None
        for feature in features_to_test:
            values_for_split = trainX.iloc[:, feature]
            splitting_points = sorted(values_for_split.unique())
            gap = int(len(splitting_points) / 20) + 1
            splitting_points = splitting_points[::gap]

            #  take the midpoints between adjacent values
            # for i in range(len(splitting_points) - 1):
            #     splitting_points[i] = (splitting_points[i] + splitting_points[i + 1]) / 2
            # splitting_points.pop()

            # calculate the impurity of each of the two subsets of profiles split accordingly.
            for sp_value in splitting_points:
                subset1 = trainY[trainX.iloc[:, feature] < sp_value]
                gini1 = cal_gini(subset1)
                subset2 = trainY[trainX.iloc[:, feature] >= sp_value]
                gini2 = cal_gini(subset2)
                # Sum these, weighted by the fraction of profiles in each subset.
                gini = len(subset1) / len(trainY) * gini1 + len(subset2) / len(trainY) * gini2
                if res is None:
                    res = (feature, sp_value, gini)
                elif gini < res[2]:
                    res = (feature, sp_value, gini)

        return res

    @staticmethod
    def fit(trainX, trainY, min_size, max_splits, nfeats):
        gini_if_no_split = MyDecisionInner.gini(trainY)
        if gini_if_no_split == 0:
            return MyDecisionLeaf(find_mostly_occurred_label(trainY))

        feature, sp_value, gini = MyDecisionInner.find_split(trainX, trainY, nfeats)
        if gini < gini_if_no_split:  # split
            subset_lt_X = trainX[trainX.iloc[:, feature] < sp_value]
            subset_lt_Y = trainY[trainX.iloc[:, feature] < sp_value]
            subset_ge_X = trainX[trainX.iloc[:, feature] >= sp_value]
            subset_ge_Y = trainY[trainX.iloc[:, feature] >= sp_value]
            max_splits -= 1
            if len(subset_lt_Y) < min_size or max_splits == 0:
                lt = MyDecisionLeaf.fit(subset_lt_X, subset_lt_Y, min_size, max_splits, nfeats)
            else:
                lt = MyDecisionInner.fit(subset_lt_X, subset_lt_Y, min_size, max_splits, nfeats)

            if len(subset_ge_Y) < min_size or max_splits == 0:
                ge = MyDecisionLeaf.fit(subset_ge_X, subset_ge_Y, min_size, max_splits, nfeats)
            else:
                ge = MyDecisionInner.fit(subset_ge_X, subset_ge_Y, min_size, max_splits, nfeats)

            return MyDecisionInner(feature, sp_value, lt, ge)
        else:
            return MyDecisionLeaf(find_mostly_occurred_label(trainY))


train_data = pd.read_csv("hw1_trainingset.csv")
test_data = pd.read_csv("hw1_testset.csv")
my_tree = MyDecisionTree()
my_tree = my_tree.fit(
    train_data.iloc[:, : -1],
    train_data.iloc[:, -1],
    min_size=10,
    max_splits=10,
    nfeats=6
)

predicted_y = []
for i in range(len(test_data)):
    predicted_y.append(my_tree.predict(test_data.iloc[i, :]))

test_data['Label'] = predicted_y
test_data.to_csv("my_decision_tree_predict_result.csv")