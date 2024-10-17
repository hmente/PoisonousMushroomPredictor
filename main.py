import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import warnings
import numpy as np
import itertools
from sklearn.model_selection import KFold
from joblib import Parallel, delayed

warnings.simplefilter(action='ignore', category=FutureWarning)

class TreeNode:
    def __init__(self, is_leaf=False, decision_criterion=None, prediction=None):
        self.is_leaf = is_leaf
        self.decision_criterion = decision_criterion
        self.left_child = None
        self.right_child = None
        self.prediction = prediction
        self.depth = 0
        self.leaf_count = 0

    def add_children(self, left, right):
        self.left_child = left
        self.right_child = right
class DecisionTree:
    def __init__(self, max_depth=None, min_samples_split=2, split_function="gini", entropy_threshold=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.split_function = split_function
        self.entropy_threshold = entropy_threshold
        self.root = None

    def fit(self, X, y):
        self.root = self._grow_tree(X, y, depth=0)

    def _grow_tree(self, X, y, depth):
        num_samples, num_features = X.shape
        num_labels = len(np.unique(y))

        if depth >= self.max_depth or num_samples < self.min_samples_split or num_labels == 1:
            leaf_value = self._most_common_label(y)
            return TreeNode(is_leaf=True, prediction=leaf_value)

        best_criterion, best_sets = self._best_split(X, y)

        if best_criterion is None:
            leaf_value = self._most_common_label(y)
            return TreeNode(is_leaf=True, prediction=leaf_value)

        left_child = self._grow_tree(best_sets['left_X'], best_sets['left_y'], depth + 1)
        right_child = self._grow_tree(best_sets['right_X'], best_sets['right_y'], depth + 1)

        node = TreeNode(is_leaf=False, decision_criterion=best_criterion)
        node.depth = depth
        node.add_children(left_child, right_child)
        node.leaf_count = 1 if left_child.is_leaf and right_child.is_leaf else 0
        return node

    def _best_split(self, X, y):
        best_criterion = None
        best_sets = None
        best_score = float('inf') if self.split_function in ["gini", "squared_impurity"] else -float('inf')

        num_samples, num_features = X.shape

        for feature_index in range(num_features):
            feature_values = X[:, feature_index]
            unique_values = np.unique(feature_values)

            for threshold in unique_values:
                left_indices = feature_values <= threshold
                right_indices = ~left_indices

                left_y, right_y = y[left_indices], y[right_indices]

                if len(left_y) == 0 or len(right_y) == 0:
                    continue

                if self.split_function == "gini":
                    score = self._gini_impurity(left_y, right_y)
                elif self.split_function == "entropy":
                    score = self._scaled_entropy(left_y, right_y)
                elif self.split_function == "squared_impurity":
                    score = self._squared_impurity(left_y, right_y)

                if (self.split_function in ["gini", "squared_impurity"] and score < best_score) or (self.split_function == "entropy" and score > best_score):
                    best_score = score
                    best_criterion = (feature_index, threshold)
                    best_sets = {
                        'left_X': X[left_indices], 'right_X': X[right_indices],
                        'left_y': left_y, 'right_y': right_y
                    }

        return best_criterion, best_sets

    def _gini_impurity(self, left_y, right_y):
        n = len(left_y) + len(right_y)
        left_prob = len(left_y) / n
        right_prob = len(right_y) / n

        def gini(y):
            classes, counts = np.unique(y, return_counts=True)
            prob = counts / len(y)
            return 1 - np.sum(prob ** 2)

        gini_left = gini(left_y)
        gini_right = gini(right_y)

        return left_prob * gini_left + right_prob * gini_right

    def _scaled_entropy(self, left_y, right_y):
        n = len(left_y) + len(right_y)
        left_prob = len(left_y) / n
        right_prob = len(right_y) / n

        def entropy(y):
            classes, counts = np.unique(y, return_counts=True)
            prob = counts / len(y)
            return -np.sum(prob * np.log2(prob + 1e-9))

        entropy_left = entropy(left_y)
        entropy_right = entropy(right_y)

        return left_prob * entropy_left + right_prob * entropy_right

    def _squared_impurity(self, left_y, right_y):
        n = len(left_y) + len(right_y)
        left_prob = len(left_y) / n
        right_prob = len(right_y) / n

        def squared_impurity(y):
            classes, counts = np.unique(y, return_counts=True)
            prob = counts / len(y)
            return 1 - np.sum(prob ** 4)

        sq_impurity_left = squared_impurity(left_y)
        sq_impurity_right = squared_impurity(right_y)

        return left_prob * sq_impurity_left + right_prob * sq_impurity_right

    def _most_common_label(self, y):
        return np.bincount(y).argmax()

    def predict(self, X):
        predictions = [self._traverse_tree(x, self.root) for x in X]
        return np.array(predictions)

    def _traverse_tree(self, x, node):
        if node.is_leaf:
            return node.prediction

        feature_index, threshold = node.decision_criterion
        if x[feature_index] <= threshold:
            return self._traverse_tree(x, node.left_child)
        else:
            return self._traverse_tree(x, node.right_child)
def zero_one_loss(y_true, y_pred):
    return np.mean(y_pred != y_true)
def grid_search(X_train, y_train, param_grid, scoring_func):

    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    def evaluate_params(params):
        current_scores = []
        depths = []
        leafs = []

        for train_idxs, val_idxs in cv.split(X_train, y_train):
            X_train_cv, y_train_cv = X_train.iloc[train_idxs], y_train.iloc[train_idxs]
            X_val_cv, y_val_cv = X_train.iloc[val_idxs], y_train.iloc[val_idxs]

            model = DecisionTree(
                max_depth=params.get('max_depth'),
                entropy_threshold=0.0001,
                split_function=params['split_function'],
                min_samples_split=2
            )

            model.fit(X_train_cv.values, y_train_cv.values)
            y_val_pred = model.predict(X_val_cv.values)

            score = scoring_func(y_val_cv, y_val_pred)
            current_scores.append(score)

            depths.append(model.root.depth)
            leafs.append(model.root.leaf_count)

        mean_score = np.mean(current_scores)
        mean_depth = np.mean(depths)
        mean_leafs = np.mean(leafs)

        print(f"Zero-one loss: {mean_score:.5f} with params: {params} \t Mean depth: {mean_depth:.1f}, Mean leafs: {mean_leafs:.1f}")

        return params, mean_score

    param_combinations = [
        dict(zip(param_dict.keys(), values)) for param_dict in param_grid for values in
        itertools.product(*param_dict.values())
    ]

    print(f"\nTotal number of combinations: {len(param_combinations)}  x  5 cv = {5 * len(param_combinations)} iterations\n")

    results = Parallel(n_jobs=-1)(delayed(evaluate_params)(params) for params in param_combinations)

    sorted_results = sorted(results, key=lambda x: x[1])

    print("\nTop 5 Best Results:")
    for rank, (params, mean_score) in enumerate(sorted_results[:5], 1):
        print(f"Rank {rank}: Zero-one loss: {mean_score:.6f} with params: {params}")

    print("\nTop 5 Worst Results:")
    for rank, (params, mean_score) in enumerate(sorted_results[-5:], 1):
        print(f"Rank {rank}: Zero-one loss: {mean_score:.6f} with params: {params}")

    return results, sorted_results[0][0], sorted_results[0][1]
def evaluate_test_result(test_loss, is_grid_search=False):
    if test_loss < 0.05:
        print("Great result! The model performs very well with almost perfect classification.")
    elif test_loss < 0.1:
        print("Good result! The model's performance is strong, but there may be room for improvement.")
    else:
        if is_grid_search:
            print("The result is not satisfactory. The model may be underfitting or overfitting. Review the parameters and try again.")
        else:
            print("The result can be improved. The model may be underfitting or overfitting. Consider adjusting parameters like max_depth or experimenting with different split functions.")

if __name__ == '__main__':
    data = pd.read_csv('data/secondary_data.csv', delimiter=';')

    missing_data_percentage = (data.isnull().sum() / data.shape[0]) * 100
    missing_data_percentage = missing_data_percentage.sort_values(ascending=False)
    # print(missing_data_percentage)

    columns_to_drop = ['veil-type', 'spore-print-color', 'veil-color', 'stem-root', 'stem-surface']
    data_cleaned = data.drop(columns=columns_to_drop)
    # print(data_cleaned.head())

    columns_to_fill = ['gill-spacing', 'cap-surface', 'gill-attachment', 'ring-type']
    for column in columns_to_fill:
        most_frequent_value = data_cleaned[column].mode()[0]  # Finding the most frequent value
        data_cleaned[column].fillna(most_frequent_value, inplace=True)

    # print(data_cleaned.isnull().sum())

    non_numeric_columns = data_cleaned.select_dtypes(exclude=['number']).columns
    print(non_numeric_columns)

    print(data_cleaned.info())
    print(data_cleaned.nunique())

    one_hot_columns = ['cap-shape', 'gill-attachment', 'cap-surface', 'cap-color', 'gill-color', 'stem-color',
                       'ring-type', 'habitat']
    data_encoded = pd.get_dummies(data_cleaned, columns=one_hot_columns)
    data_encoded = data_encoded.apply(lambda col: col.map(lambda x: 1 if x == True else (0 if x == False else x)))
    label_columns = ['does-bruise-or-bleed', 'gill-spacing', 'class', 'has-ring', 'season']
    le = LabelEncoder()
    for column in label_columns:
        data_encoded[column] = le.fit_transform(data_encoded[column])
    # print(data_encoded.head())

    #print(data_encoded.select_dtypes(exclude=['number']).columns)

    X = data_encoded.drop(['class'], axis=1)
    y = data_encoded['class']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    param_grid = [
        {
            'max_depth': [10, 15, 20],
            'split_function': ['scaled_entropy', 'gini', 'squared_impurity']
        }
    ]

    try:
        user_choice = input("\nWould you like to use the best parameters found by grid search? (yes/no): ").strip().lower()

        if user_choice == 'yes':
            print("\nRunning grid search for best parameters...")
            results, best_params, best_score = grid_search(X_train, y_train, param_grid, scoring_func=zero_one_loss)

            print(f"\nBest parameters: {best_params}")
            print(f"Best score (zero-one loss): {best_score}")

            best_tree = DecisionTree(
                max_depth=best_params['max_depth'],
                split_function=best_params['split_function'],
                min_samples_split=2,
                entropy_threshold=0.0001
            )
            best_tree.fit(X_train.values, y_train.values)

            y_test_pred = best_tree.predict(X_test.values)
            test_loss = zero_one_loss(y_test, y_test_pred)
            print(f"Zero-one loss on test set with best params: {test_loss:.6f}")
            evaluate_test_result(test_loss)
        elif user_choice == 'no':
            print("\nPlease enter your custom parameters.")
            max_depth = int(input("Enter the max depth for the tree (e.g., 10, 15, 20): "))
            split_function = input("Enter the split function (scaled_entropy/gini/squared_impurity): ").strip()

            custom_tree = DecisionTree(
                max_depth=max_depth,
                split_function=split_function,
                min_samples_split=2,
                entropy_threshold=0.0001
            )

            custom_tree.fit(X_train.values, y_train.values)

            y_test_pred = custom_tree.predict(X_test.values)
            test_loss = zero_one_loss(y_test, y_test_pred)
            print(f"Zero-one loss on test set with custom params: {test_loss:.6f}")
            evaluate_test_result(test_loss)
        else:
            print("Invalid input. Please run the program again and choose 'yes' or 'no'.")

    except Exception as exc:
        print(f"There was an exception running the program. Exception: {exc}")