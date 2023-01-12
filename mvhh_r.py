#
# MVHH(R)
# 在根结点随机确定一个视图，后续结点沿用
#
# .....Importing all the packages.........................
#
#
import numpy as np
import pandas as pd
import random
from copy import deepcopy
from scipy.linalg import norm
from sklearn.decomposition import PCA
from sklearn.base import BaseEstimator, ClassifierMixin
#
#
#
from sklearn.preprocessing import MinMaxScaler

view_choice = -1

class MVHHNode:

    def __init__(self, depth, labels, featureIDs, **kwargs):  # Defining the node structure
        self.depth = depth  # Depth of the node
        self.labels = labels  # Label associated with the node
        self.featureIDs = featureIDs  # 结点用到的features
        self.is_leaf = kwargs.get('is_leaf', False)  # 'is_leaf' is set to 'False' for internal nodes
        self._split_rules = kwargs.get('split_rules', None)  # 'split_rules' is set to 'None'
        self._weights = kwargs.get('weights', None)  # weights associated with the node
        self._left_child = kwargs.get('left_child', None)  # left_child index
        self._right_child = kwargs.get('right_child', None)  # right_child index

        if not self.is_leaf:
            assert self._split_rules
            assert self._left_child
            assert self._right_child

    def get_child(self, datum):
        if self.is_leaf:
            raise Warning("Leaf node does not have children.")
        X = deepcopy(datum)
        if X[self.featureIDs].dot(np.array(self._weights[:-1]).T) - self._weights[-1] < 0:
            return self.left_child
        else:
            return self.right_child

    @property
    def label(self):
        if not hasattr(self, '_label'):
            classes, counts = np.unique(self.labels, return_counts=True)
            self._label = classes[np.argmax(counts)]
        return self._label

    @property
    def split_rules(self):
        if self.is_leaf:
            raise Warning("Leaf node does not have split rule.")
        return self._split_rules

    @property
    def left_child(self):
        if self.is_leaf:
            raise Warning("Leaf node does not have split rule.")
        return self._left_child

    @property
    def right_child(self):
        if self.is_leaf:
            raise Warning("Leaf node does not have split rule.")
        return self._right_child


class MultiViewHouseHolder(BaseEstimator):

    def __init__(self, impurity, segmentor, view,  max_depth, min_samples_split=2, method='eig', tau=0.05,  **kwargs):

        self.impurity = impurity
        self.segmentor = segmentor
        self.method = method
        self.tau = tau
        self.view = view
        self._max_depth = max_depth
        self._min_samples = min_samples_split
        self._alpha = kwargs.get('alpha', None)  # only for linreg method
        self._root = None
        self._nodes = []

    def _variance_multi_view_subspcace(self, X, start, end, max_features='constant'):
        feature_num = end - start + 1
        subset_feature_num = feature_num

        if max_features == 'all':
            subset_feature_num = feature_num
        if max_features == 'constant':
            subset_feature_num = int((feature_num/3)*2)

        feature_ids = range(start, end+1)
        subset_feature_ids = random.sample(feature_ids, subset_feature_num)
        subset_feature_ids.sort()

        var_threshold=1
        temp = X[:, subset_feature_ids]
        temp = MinMaxScaler().fit_transform(temp)
        b = np.var(temp, axis=0)  #列
        var_threshold=np.mean(b)
        true_ids = []
        for i in range(len(subset_feature_ids)):
            if b[i] >= var_threshold:
                true_ids.append(subset_feature_ids[i])
        return true_ids

    def _extract_class(self,subFeatures_X,y,class_value):
        subFeatures_X = pd.DataFrame(subFeatures_X)
        y = pd.DataFrame(y)
        y.columns=['Y']

        x_with_labels = pd.concat([subFeatures_X,y], axis=1)
        extract = x_with_labels.loc[x_with_labels['Y'] == class_value]
        extract_y = extract['Y']
        extract_X = extract.drop('Y', axis=1)

        extract_y = np.array(extract_y)
        extract_X = np.array(extract_X)
        return extract_X,extract_y

    def _terminate(self, X, y, cur_depth):  # termination conditions

        if self._max_depth != None and cur_depth == self._max_depth:  # maximum depth is reached
            return True
        elif y.size < self._min_samples:  # minimum number of samples has been reached
            return True
        elif np.unique(y).size == 1:  # terminate if the node is homogeneous
            return True
        else:
            return False

    def _generate_leaf_node(self, cur_depth, y, subset_feature_ids):

        node = MVHHNode(cur_depth, y, featureIDs=subset_feature_ids, is_leaf=True)
        self._nodes.append(node)
        return node

    def _generate_node(self, X, y, cur_depth):
        count_y = list(y)
        if self._terminate(X, y, cur_depth):
            feature_num = X.shape[1]
            feature_ids = range(feature_num)
            return self._generate_leaf_node(cur_depth, y, feature_ids)
        else:
            impurity_flag = 100000
            subFeatures_ids_flag = None
            sr_flag = None
            householder_flag = None

            for i in range(0, 1):
                global view_choice
                i = view_choice

                subset_feature_ids = self._variance_multi_view_subspcace(X, self.view[i][0], self.view[i][1])
                subFeatures_X = X[:, subset_feature_ids]
                n_objects, n_features = subFeatures_X.shape
                impurity_best, sr, left_indices, right_indices = self.segmentor(subFeatures_X, y, self.impurity)

                for j in range(0, 2):
                    extract_X,extract_y = self._extract_class(subFeatures_X,y,j)
                    if self.method == 'eig':
                        extractor = PCA(n_components=1)
                        extractor.fit(extract_X)
                        mu = extractor.components_[0]

                    I = np.diag(np.ones(n_features))
                    check_ = np.sqrt(((I - mu) ** 2).sum(axis=1))
                    if (check_ > self.tau).sum() > 0:
                        i = np.argmax(check_)
                        e = np.zeros(n_features)
                        e[i] = 1.0
                        w = (e - mu) / norm(e - mu)
                        householder_matrix = I - 2 * w[:, np.newaxis].dot(w[:, np.newaxis].T)

                        X_house = subFeatures_X.dot(householder_matrix)
                        impurity_house, sr_house, left_indices_house, right_indices_house = self.segmentor(X_house, y,
                                                                                                           self.impurity)
                        if (impurity_best > impurity_house):
                            impurity_best = impurity_house
                            left_indices = left_indices_house
                            right_indices = right_indices_house
                            sr = sr_house
                    else:
                        householder_matrix = I

                    if sr_flag == None or impurity_flag > impurity_best:
                        impurity_flag = impurity_best
                        sr_flag = sr
                        householder_flag = householder_matrix
                        subFeatures_ids_flag = subset_feature_ids


            subset_feature_ids = subFeatures_ids_flag
            sr = sr_flag
            householder_matrix = householder_flag

            if not sr_flag:
                return self._generate_leaf_node(cur_depth, y, subFeatures_ids_flag)

            subFeatures_X = X[:, subset_feature_ids]
            n_objects, n_features = subFeatures_X.shape

            i, treshold = sr
            weights = np.zeros(n_features + 1)
            weights[:-1] = householder_matrix[:, i]
            weights[-1] = treshold
            left_indices = subFeatures_X.dot(np.array(weights[:-1]).T) - weights[-1] < 0
            right_indices = np.logical_not(left_indices)
            X_left, y_left = X[left_indices], y[left_indices]
            X_right, y_right = X[right_indices], y[right_indices]

            # 判断生成的左右两个节点，叶子or继续分
            if (len(y_right) <= self._min_samples):
                return self._generate_leaf_node(cur_depth, y, subset_feature_ids)
            elif (len(y_left) <= self._min_samples):
                return self._generate_leaf_node(cur_depth, y, subset_feature_ids)
            else:
                #
                node = MVHHNode(cur_depth, y, featureIDs=subset_feature_ids, split_rules=sr, weights=weights,
                                  left_child=self._generate_node(X_left, y_left, cur_depth + 1),
                                  right_child=self._generate_node(X_right, y_right, cur_depth + 1), is_leaf=False)
                self._nodes.append(node)
                return node

    def fit(self, X, y):
        # 在根结点随机确定一个视图
        i = random.randint(0, len(self.view) - 1)
        global view_choice
        view_choice= i

        self._root = self._generate_node(X, y, 0)

    def get_params(self, deep=True):

        return {'max_depth': self._max_depth, 'min_samples_split': self._min_samples,
                'impurity': self.impurity, 'segmentor': self.segmentor,'view':self.view}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def predict(self, X):
        def predict_single(datum):
            cur_node = self._root
            while not cur_node.is_leaf:
                cur_node = cur_node.get_child(datum)
            return cur_node.label

        if not self._root:
            raise Warning("Decision tree has not been trained.")
        size = X.shape[0]
        predictions = np.empty((size,), dtype=int)
        for i in range(size):
            predictions[i] = predict_single(X[i, :])
        return predictions

    def score(self, data, labels):
        if not self._root:
            raise Warning("Decision tree has not been trained.")
        predictions = self.predict(data)
        correct_count = np.count_nonzero(predictions == labels)
        return correct_count / labels.shape[0]


#
# Definition of classes provided
#
class MVHH_R_Classifier(ClassifierMixin, MultiViewHouseHolder):#VAR
    def __init__(self, impurity, segmentor, view, max_depth=50, min_samples_split=2, **kwargs):
        super().__init__(impurity=impurity, segmentor=segmentor, view=view, max_depth=max_depth,
                         min_samples_split=min_samples_split, **kwargs)