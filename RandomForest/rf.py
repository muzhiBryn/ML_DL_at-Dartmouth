import collections, math, numpy as np, random, sys
from expression import ExpressionProfile


def cal_gini(eps):
    res = 1
    label_counts = {}
    for ep in eps:
        if ep.label not in label_counts:
            label_counts[ep.label] = 0
        label_counts[ep.label] += 1
    for k in label_counts:
        res -= (label_counts[k] / len(eps)) ** 2
    return res


def find_mostly_occurred_label(eps):
    """
    :param eps: [ExpressionProfile]
    :return: the mostly occurred label in eps
    """
    label_counts = {}
    for ep in eps:
        if ep.label not in label_counts:
            label_counts[ep.label] = 0
        label_counts[ep.label] += 1
    max_count = max(label_counts.values())
    for k in label_counts:
        if label_counts[k] == max_count:
            return k


class ExpressionDecisionTree (object):
    """Base class for decision trees of expression profiles"""
    
    def classify(self, ep):
        """Predict the label for the expression profile
        Args:
            ep: ExpressionProfile
        Returns:
            label: String
        """
        # just here for defining the interface; work is done in subclasses
        pass
    
    def add_mdis(self, mdis):
        """Add the MDI (mean decrease in impurity) for the tree and its children (if any)
        Args:
            mdis: { String: float }, maping label to total MDI so far
        Returns:
            nothing, but mdis is updated
        """
        # just here for defining the interface; work is done in subclasses
        pass

    @staticmethod
    def impurity(eps):
        """Calculate the Gini impurity for the profiles
        Args:
            eps: [ ExpressionProfile ]
        Returns:
            the impurity
        """
        # TODO your code here
        return cal_gini(eps)

    
    @staticmethod
    def train(eps, total_num_eps, min_size, max_splits, nfeats_test):
        """Create a ExpressionDecisionTree (could be leaf or inner) based on the expression profiles
        Args:
            eps: [ ExpressionProfile ], training data
            total_num_eps: the number of expression profiles at the root (we might now have fewer)
            min_size: don't split further when len(eps) is less than this
            max_splits: don't split more than this many times (so if 0, don't split)
            nfeats_test: how many features to test for splitting (randomly sample if < the actual number)
        Returns:
            ExpressionDecisionTree
        """
        if len(eps) < min_size or max_splits == 0:
            return ExpressionDecisionLeaf.train(eps, total_num_eps, min_size, max_splits, nfeats_test)
        else:
            return ExpressionDecisionInner.train(eps, total_num_eps, min_size, max_splits, nfeats_test)


class ExpressionDecisionLeaf (ExpressionDecisionTree):
    """A leaf in a decision tree of expression profiles.
    Instance variables:
        label: the label to be predicted if reach this leaf
    """
    
    def __init__(self, label):
        self.label = label
        
    def __repr__(self):
        return str(self)

    def __str__(self):
        return '*'+str(self.label)

    def classify(self, ep):
        # TODO your code here (definitely don't overthink this one :)
        return self.label

    @staticmethod
    def train(eps, total_num_eps, min_size, max_splits, nfeats_test):
        # TODO your code here
        return ExpressionDecisionLeaf(find_mostly_occurred_label(eps))


class ExpressionDecisionInner (ExpressionDecisionTree):
    """An inner node in a decision tree of expression profiles.
    Instance variables:
        feat: the feature to test (index into expression profile values array)
        val: the value to compare the feature's value agains
        lt: the child for profiles with the feature's value < val
        ge: the child for profiles with the feature's value >= val
        mdi: the mean decrease in impurity for this node
    """
    
    def __init__(self, feat, val, lt, ge, mdi):
        self.feat = feat
        self.val = val
        self.lt = lt
        self.ge = ge
        self.mdi = mdi
        
    def __repr__(self):
        return str(self)

    def __str__(self):
        return str(self.feat)+'@%.2f'%(self.val,)+'('+str(self.lt)+', '+str(self.ge)+')'

    def classify(self, ep):
        # TODO your code here
        if ep[self.feat] < self.val:
            return self.lt.classify(ep)
        else:
            return self.ge.classify(ep)

    def add_mdis(self, mdis):
        # TODO your code here
        # So here, a node adds its saved mdi value (computed during train) for their feature, and recurses to its children. (3 points)
        if self.feat not in mdis:
            mdis[self.feat] = self.mdi
        self.lt.add_mdis(mdis)
        self.ge.add_mdis(mdis)

    @staticmethod
    def find_split(eps, nfeats_test):
        """Determine a feature and value that will split the expression profiles well
        Args:
            eps: [ ExpressionProfile ]
            nfeats_test: how many features to test (randomly sampled if less than the total number)
        Returns:
            (feat, val, sub_imp): the feature to test, the value to test it with, and the total weighted impurity of the split sets
        """
        # TODO your code here
        # Consider a random subset of features of the specified size
        total_feature_num = len(eps[0].values)
        if nfeats_test <= total_feature_num:
            features_to_test = sorted(random.sample(range(total_feature_num), nfeats_test))
        else:
            features_to_test = range(total_feature_num)

        # For each feature under consideration, and each splitting value:
        #   various values that show up in the profiles for the feature at hand, and take the midpoints between adjacent values.
        # calculate the impurity of each of the two subsets of profiles split accordingly.
        res = None
        for feature in features_to_test:
            values_for_split = [ep[feature] for ep in eps]
            splitting_points = sorted(list(set(values_for_split)))
            #  take the midpoints between adjacent values
            for i in range(len(splitting_points) - 1):
                splitting_points[i] = (splitting_points[i] + splitting_points[i + 1]) / 2
            splitting_points.pop()

            # calculate the impurity of each of the two subsets of profiles split accordingly.
            for sp_value in splitting_points:
                subset1 = [ep for ep in eps if ep[feature] < sp_value]
                gini1 = cal_gini(subset1)
                subset2 = [ep for ep in eps if ep[feature] >= sp_value]
                gini2 = cal_gini(subset2)
                # Sum these, weighted by the fraction of profiles in each subset.
                gini = len(subset1) / len(eps) * gini1 + len(subset2) / len(eps) * gini2
                if res is None:
                    res = (feature, sp_value, gini)
                elif gini < res[2]:
                    res = (feature, sp_value, gini)

        return res


    
    @staticmethod
    def train(eps, total_num_eps, min_size, max_splits, nfeats_test):
        # TODO your code here
        gini_if_no_split = ExpressionDecisionInner.impurity(eps)
        if gini_if_no_split == 0:
            return ExpressionDecisionLeaf(find_mostly_occurred_label(eps))

        feature, sp_value, gini = ExpressionDecisionInner.find_split(eps, nfeats_test)
        if gini < gini_if_no_split: # split
            subset_lt = [ep for ep in eps if ep[feature] < sp_value]
            subset_ge = [ep for ep in eps if ep[feature] >= sp_value]
            max_splits -= 1
            if len(subset_lt) < min_size or max_splits == 0:
                lt = ExpressionDecisionLeaf.train(subset_lt, total_num_eps, min_size, max_splits, nfeats_test)
            else:
                lt = ExpressionDecisionInner.train(subset_lt, total_num_eps, min_size, max_splits, nfeats_test)

            if len(subset_ge) < min_size or max_splits == 0:
                ge = ExpressionDecisionLeaf.train(subset_ge, total_num_eps, min_size, max_splits, nfeats_test)
            else:
                ge = ExpressionDecisionInner.train(subset_ge, total_num_eps, min_size, max_splits, nfeats_test)
            # also keep track of the mean decrease in impurity (mdi), for use in calculating feature importances.
            mdi = (gini_if_no_split - gini) * len(eps) / total_num_eps
            return ExpressionDecisionInner(feature, sp_value, lt, ge, mdi)
        else:
            return ExpressionDecisionLeaf(find_mostly_occurred_label(eps))



class ExpressionRandomForest (object):
    """A random forest for expression profiles, comprised of expression decision trees"""
    def __init__(self, trees):
        self.trees = trees

    def __str__(self):
        return '\n'.join(str(t) for t in self.trees)
        
    def classify(self, ep):
        # TODO your code here
        candidate_labels = []
        for tree in self.trees:
            candidate_labels.append(tree.classify(ep))
        label_counts = {}
        for label in candidate_labels:
            if label not in label_counts:
                label_counts[label] = 0
            label_counts[label] += 1
        max_count = max(label_counts.values())
        for k in label_counts:
            if label_counts[k] == max_count:
                return k

    def importances(self):
        """Calculate feature importances by averaging mdi over the trees in the forest"""
        # TODO your code here
        # compute feature importances by asking each tree to
        # add_mdis to a “running total” dictionary, and then dividing by the number
        # of trees to get the overall mean
        mdis = {}
        for tree in self.trees:
            tree.add_mdis(mdis)
        for k in mdis:
            mdis[k] = mdis[k] / len(self.trees)
        return mdis

    
    @staticmethod
    def train(eps, ntrees, min_size, max_splits, nfeats_test, resample=True):
        """Create a ExpressionRandomForest based on the expression profiles
        Args:
            eps: [ ExpressionProfile ], training data
            ntrees: how many trees to include in the forest
            min_size: don't split further when len(eps) is less than this
            max_splits: don't split more than this many times (so if 0, don't split)
            nfeats_test: how many features to test for splitting (randomly sample if < the actual number)
            resample: sample (with replacement) from eps for each tree? (normally do, but for testing can be helpful to just use original eps)
        Returns:
            ExpressionRandomForest
        """
        # TODO your code here
        trees = []
        for _ in range(ntrees):
            # repeatedly add values from the list of expression profiles without removal to a set
            # (so there could be duplicate expression profiles in the set we are creating) until the size of the set
            # is equal to the size of the original list of profiles
            if resample:
                resampled_eps = []
                for _ in range(len(eps)):
                    idx = random.randint(0, len(eps) - 1)
                    resampled_eps.append(eps[idx])
                trees.append(
                    ExpressionDecisionTree.train(resampled_eps, len(resampled_eps), min_size, max_splits, nfeats_test))
            else:
                trees.append(
                    ExpressionDecisionTree.train(eps, len(eps), min_size, max_splits, nfeats_test))
        return ExpressionRandomForest(trees)


def xval(eps, trainer, nfold=5, nrep=10):
    """Cross-validate the random forest
    Repeat:
    * split profiles into folds randomly
    * for each fold, use the other folds as training, assign test labels to the left-out fold, and see which are right
    Average the number of correct labels for each label, over the repetitions.
    Args:
      eps: [ ExpressionProfile ], training data
      trainer: [ ExpressionProfile ] -> ExpressionRandomForest, trained on the provided data
      nfold: the number of roughly-equally-sized sets to split the profiles into
      nrep: the number of times to repeat the split, with random partitioning each time
    Returns:
      { label : correct }, average # correct for that label over the folds and repetitions
    """
    # TODO: your code here

    res = {}
    count = {}
    for _ in range(nrep):
        # shuffle the eps: randomly partition into new folds
        random.shuffle(eps)
        for i in range(nfold):
            validate_eps = eps[i::nfold]
            training_eps = [ep for ep in eps if ep not in validate_eps]
            clf = trainer(training_eps)
            for ep in validate_eps:
                predicted_label = clf.classify(ep)
                # for each ground truth label, how many (count) of the test instances are assigned correct labels
                # "to make life easier for grading, since they're equivalent correctness-wise anyway, please stick with the average count." - Piazza
                if predicted_label == ep.label:
                    if ep.label not in res:
                        res[ep.label] = 0
                    res[ep.label] += 1
                if ep.label not in count:
                    count[ep.label] = 0
                count[ep.label] += 1
    for k in res:
        res[k] = res[k] / (nrep * nfold) # for grading
        #res[k] = res[k] / count[k] # for print accuracy
    return res




def concoct_dataset(n_per_label, feat_specs, sigma=0, shuffle=True):
    """Generate expression profiles according to the specs
    Class labels are 0, 1, ...
    Feature values are also 0, 1, ..., potentially with Gaussian noise added
    Args:
        n_per_label: how many instances to make with each label; e.g., [10,10] means 10 labeled 0 and 10 labeled 1
        feat_specs: a list of specs for each feature (i.e., # feats = len(feat_specs))
                    each spec is of the form [[n00, n01, ...], [n10, n11, ...], ...] -- of the class 0 instances, n00 have feature value 0, n01 have 1; 
                      while of the class 1 instances, n10 have value 0, n11 have value 1, ...
        sigma: for Gaussian noise, so that the feature value aren't exactly 0, 1, 2, but rather centered on those values
        shuffle: if true, randomly assign the nij feature values to the instances; else assign in order (e.g., first n00 of the label-0 instances get value 0, next n01 get value 1, ...)
    Returns:
        [ ExpressionProfile ] with values distributed this way
    Examples:
        concoct_dataset([10,10], [ [[8,2],[3,7]] ] , 0, False)
        => # [10,10] -- 10 samples with label 0, 10 with label 1
           {0: [[0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [1.0], [1.0]], # first 8 have 0 and remaining 2 have 1
            1: [[0.0], [0.0], [0.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0]]} # first 3 have 0 and remaining 7 have 1
        concoct_dataset([5,5,5], [ [[4,0,1],[1,3,1],[0,0,5]], [[3,2,0],[0,5,0],[1,1,3]] ], 0, False)
        => # [5,5,5] -- 5 samples with label 0, 5 with label 1, and 5 with label 2
           {0: [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 1.0], [2.0, 1.0]], # for the first feature [4.0,1] -- 4 have 0 and 1 has 2
                                                                             # for the second features [3,2,0] -- 3 have 0 and 2 have 1
            1: [[0.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [2.0, 1.0]], # first [1,3,1] -- 1 has 0, 3 have 1, 1 has 2
                                                                             # second [0,5,0] -- all have 1
            2: [[2.0, 0.0], [2.0, 1.0], [2.0, 2.0], [2.0, 2.0], [2.0, 2.0]]} # first [0,0,5] -- all have 2
                                                                             # second [1,1,3] -- 1 has 0, 1 has 1, 3 have 2
    """
    samples = dict((label, [list() for i in range(n)]) for (label,n) in enumerate(n_per_label))
    for feat_spec in feat_specs:
        for (label, breakdown) in enumerate(feat_spec):
            if shuffle: random.shuffle(samples[label])
            s = 0
            for (value, n_per_value) in enumerate(breakdown):
                if s+n_per_value > len(samples[label]): raise Exception(str(feat_spec)+' has too many samples')
                for i in range(n_per_value):
                    samples[label][s+i].append(np.random.normal(value, sigma))
                s += n_per_value
            if s < len(samples[label]): raise Exception(str(feat_spec)+' has too few samples')
    print(samples)
    return [ExpressionProfile(str(i),str(label),values) for (label, value_sets) in samples.items() for (i,values) in enumerate(value_sets)]
    
# ------------------------------------------------------------------------
# simplistic command-line driver

# arg1 = command
# the "sim" command says to use simulated data, with args giving rf parameters and a bunch of specs of how to simulate
#   <# trees> <min size> <max splits> <# feats to test> <resample>
#   <n_per_label> <sigma> <shuffle> and the rest are <feat0_spec> <feat1_spec> ...
# sim 1 2 2 1 0 "10,10" 0 0 "8,2;3,7"
# sim 2 2 2 3 1 "5,5,5" 0 0 "4,0,1;1,3,1;0,0,5" "3,2,0;0,5,0;1,1,3"

# the "data" command says to load data, with remaining args giving rf parameters
#   <# trees> <min size> <max splits> <# feats to test> <resample>
#   if <# feats to test> is 0, it is calculated as the sqrt of the actual # feats
# data data/recur.csv 100 2 3 0 1

if __name__ == '__main__':
    random.seed(12345) # comment this out if you want different results every time

    if sys.argv[1] == 'sim':
        ntrees = int(sys.argv[2]); min_size = int(sys.argv[3]); max_splits = int(sys.argv[4])
        nfeats_test = int(sys.argv[5]); resample = sys.argv[6] == '1'
        n_per_label = [int(n) for n in sys.argv[7].split(',')]
        sigma = float(sys.argv[8])
        shuffle = sys.argv[9] == '1'
        feat_specs = [[[int(n) for n in ln.split(',')] for ln in fs.split(';')] for fs in sys.argv[10:len(sys.argv)]]
        eps = concoct_dataset(n_per_label, feat_specs, sigma, shuffle)
        rf = ExpressionRandomForest.train(eps, ntrees, min_size, max_splits, nfeats_test, resample=resample)
        print(rf)
        print('\n'.join(str(feat)+' %2.2e'%(imp,) for (feat,imp) in sorted(rf.importances().items(), key=lambda fi:fi[1], reverse=True)))
        
    elif sys.argv[1] == 'data':        
        eps = ExpressionProfile.load(sys.argv[2])
        ntrees = int(sys.argv[3]); min_size = int(sys.argv[4]); max_splits = int(sys.argv[5])
        nfeats_test = int(sys.argv[6]); resample = sys.argv[7] == '1'
        if nfeats_test==0: nfeats_test = math.floor(math.sqrt(len(eps[0].values)))
        rf = ExpressionRandomForest.train(eps, ntrees, min_size, max_splits, nfeats_test, resample=resample)
        print(rf)
        print('\n'.join(str(feat)+' %2.2e'%(imp,) for (feat,imp) in sorted(rf.importances().items(), key=lambda fi:fi[1], reverse=True)))
        # test on training...
        print([(str(ep), rf.classify(ep)) for ep in eps])
        # xval
        print(xval(eps, lambda eps: ExpressionRandomForest.train(eps, ntrees, min_size, max_splits, nfeats_test, resample=resample)))

    else:
        print('no such command')