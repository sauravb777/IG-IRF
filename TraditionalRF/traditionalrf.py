import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from collections import Counter, defaultdict
from statistics import mode
from feature_ranking.rank_feature import LocalGlobalWt
from feature_weight_update.promote_demote_features import compute
# Load Titanic dataset
url = 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'
df = pd.read_csv(url)
df.columns = df.columns.str.strip()

# Preprocess
df = df.drop(['Name', 'Ticket', 'Cabin', 'PassengerId', 'Fare'], axis=1)
df['Age'] = df['Age'].fillna(df['Age'].mean())
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

# Features and target
X = df.drop('Survived', axis=1).reset_index(drop=True)
y = df['Survived'].reset_index(drop=True)

feature_ranking = LocalGlobalWt(number_of_features=len(X.columns)-1, main_features=X.columns)
# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
oob_predictions = []
local_weight = []
# Entropy
def entropy(y):
    counts = np.bincount(y)
    probabilities = counts / len(y)
    return -np.sum([p * np.log2(p) for p in probabilities if p > 0])


# Information Gain
def information_gain(y, left_y, right_y):
    H = entropy(y)
    HL = entropy(left_y)
    HR = entropy(right_y)
    p_left = len(left_y) / len(y)
    p_right = len(right_y) / len(y)
    return H - (p_left * HL + p_right * HR)


# Best Split
def best_split(X, y):
    best_gain = 0
    best_feature = None
    best_threshold = None

    for feature in X.columns:
        thresholds = np.unique(X[feature])
        for threshold in thresholds:
            left_mask = X[feature] <= threshold
            right_mask = ~left_mask
            if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                continue
            left_y = y[left_mask]
            right_y = y[right_mask]
            gain = information_gain(y, left_y, right_y)
            if gain > best_gain:
                best_gain = gain
                best_feature = feature
                best_threshold = threshold

    return best_feature, best_threshold, best_gain


# Tree Node
class TreeNode:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None, info_gain=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
        self.info_gain = info_gain


# Build Tree
def build_tree(X, y, depth=0, max_depth=3):
    if len(y) == 0:
        return TreeNode(value=None)

    if len(set(y)) == 1 or depth >= max_depth:
        return TreeNode(value=Counter(y).most_common(1)[0][0])

    feature, threshold, best_gain = best_split(X, y)
    if feature is None:
        return TreeNode(value=Counter(y).most_common(1)[0][0])

    left_mask = X[feature] <= threshold
    right_mask = ~left_mask
    left_X, left_y = X[left_mask], y[left_mask]
    right_X, right_y = X[right_mask], y[right_mask]

    if len(left_y) == 0 or len(right_y) == 0:
        return TreeNode(value=Counter(y).most_common(1)[0][0])

    node = TreeNode(feature=feature, threshold=threshold, info_gain=best_gain)
    node.left = build_tree(left_X, left_y, depth + 1, max_depth)
    node.right = build_tree(right_X, right_y, depth + 1, max_depth)
    return node


# Predict single sample
def predict_tree(x, tree):
    if tree.value is not None:
        return tree.value
    if x[tree.feature] <= tree.threshold:
        return predict_tree(x, tree.left)
    else:
        return predict_tree(x, tree.right)


# Predict forest
def predict_forest(X, trees):
    predictions = []
    for _, row in X.iterrows():
        tree_preds = [predict_tree(row, tree) for tree in trees]
        try:
            final_pred = mode(tree_preds)
        except:
            final_pred = tree_preds[0]
        predictions.append(final_pred)
    return np.array(predictions)


# def oob_score(tree, X_test, y_test):
#     mis_label = 0
#     for i in range(len(X_test)):
#         pred = predict_tree(tree, X_test[i])
#         if pred != y_test[i]:
#             mis_label += 1
#     return mis_label / len(X_test)


# Random Forest with OOB
def random_forest_with_oob(X, y, n_trees=3, sample_ratio=0.8, max_depth=3):
    n_samples = len(X)
    trees = []
    global oob_predictions, local_weight

    for i in range(n_trees):
        oob_mis_label = 0
        bootstrap_indices = np.random.choice(n_samples, int(sample_ratio * n_samples), replace=True)
        oob_indices = list(set(range(n_samples)) - set(bootstrap_indices))

        X_bootstrap = X.iloc[bootstrap_indices]
        y_bootstrap = y.iloc[bootstrap_indices]
        tree = build_tree(X_bootstrap, y_bootstrap, max_depth=max_depth)
        trees.append(tree)
        local_weight.append(feature_ranking.compute_feature_importance(tree))
        for idx in oob_indices:
            x_row = X.iloc[idx]
            pred = predict_tree(x_row, tree)
            if pred != y.iloc[idx]:
                oob_mis_label += 1

        oob_predictions.append(round(oob_mis_label / n_samples, 4))

    return trees, oob_predictions

def tree_to_dict(node):
    if node.value is not None:
        return {"value": node.value}
    return {
        "feature": node.feature,
        "threshold": node.threshold,
        "info_gain": node.info_gain,
        "left": tree_to_dict(node.left),
        "right": tree_to_dict(node.right),
    }


# Train and evaluate
trees, oob_score = random_forest_with_oob(X_train, y_train, n_trees=10, max_depth=3)
tree_normalized_weight = feature_ranking.normalized_weight_of_tree(oob_predictions)
global_weight = feature_ranking.global_wt(local_weight, tree_normalized_weight)
print("global_wt", global_weight)
updated_features, u, v_new, del_u, del_v = compute(global_weight)
y_pred = predict_forest(X_test, trees)
acc = accuracy_score(y_test, y_pred)
# print(f"Random Forest Accuracy (3 trees, depth=3): {acc:.4f}")

print(f"Test Accuracy: {acc:.4f}")
