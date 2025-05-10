import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from collections import Counter
from statistics import mode
import graphviz

# Load Titanic Dataset
url = 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'
df = pd.read_csv(url)
df.columns = df.columns.str.strip()

# Drop unnecessary columns
df = df.drop(['Name', 'Ticket', 'Cabin', 'PassengerId', 'Fare'], axis=1)

# Handle missing values
df['Age'] = df['Age'].fillna(df['Age'].mean())
df['Sex'] = df['Sex'].fillna(df['Sex'].mode()[0])
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

# Encode categorical variables
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
df['Survived'] = df['Survived'].map({0: 0, 1: 1})



# Features and target
X = df.drop('Survived', axis=1)
y = df['Survived']
print(df.head())
# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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

# Best Split with IG return
def best_split(X, y):
    best_gain = 0
    best_feature = None
    best_threshold = None

    for feature in X.columns:
        if X[feature].dtype == 'object':
            unique_values = X[feature].unique()
            for value in unique_values:
                left_mask = X[feature] == value
                right_mask = ~left_mask
                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue
                left_y = y[left_mask]
                right_y = y[right_mask]
                gain = information_gain(y, left_y, right_y)
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = value
        else:
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

    if X[feature].dtype == 'object':
        left_mask = X[feature] == threshold
        right_mask = ~left_mask
    else:
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

# Predict a single sample
def predict_tree(x, tree):
    if tree.value is not None:
        return tree.value
    feature_val = x[tree.feature]
    if isinstance(tree.threshold, str):  # Categorical
        if feature_val == tree.threshold:
            return predict_tree(x, tree.left)
        else:
            return predict_tree(x, tree.right)
    elif feature_val <= tree.threshold:  # Numerical
        return predict_tree(x, tree.left)
    else:
        return predict_tree(x, tree.right)

# Predict using forest
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

# Visualize Tree
def visualize_tree(tree, dot=None, node_id=0):
    if dot is None:
        dot = graphviz.Digraph()
    if tree.value is not None:
        dot.node(str(node_id), f"Leaf\nValue: {tree.value}", shape="box")
        return dot, node_id
    dot.node(str(node_id), f"{tree.feature} = {tree.threshold}\nIG: {tree.info_gain:.4f}")
    left_id = node_id + 1
    dot, left_end = visualize_tree(tree.left, dot, left_id)
    dot.edge(str(node_id), str(left_id), label="Yes")
    right_id = left_end + 1
    dot, right_end = visualize_tree(tree.right, dot, right_id)
    dot.edge(str(node_id), str(right_id), label="No")
    return dot, right_end

# Random Forest
def random_forest(X, y, n_trees=3, sample_ratio=0.8, max_depth=3):
    trees = []
    for i in range(n_trees):
        sample_X = X.sample(frac=sample_ratio, replace=True, random_state=i)
        sample_y = y.sample(frac=sample_ratio, replace=True, random_state=i)
        tree = build_tree(sample_X, sample_y, max_depth=max_depth)
        trees.append(tree)
    return trees

# Train 3 trees
trees = random_forest(X_train, y_train, n_trees=3, max_depth=3)

# Predict and evaluate
y_pred = predict_forest(X_test, trees)
acc = accuracy_score(y_test, y_pred)
print(f"Random Forest Accuracy (3 trees, depth=3): {acc:.4f}")

# Visualize trees
for i, tree in enumerate(trees):
    dot, _ = visualize_tree(tree)
    dot.render(f"tree_{i+1}", format="png", cleanup=True)
    print(f"Tree {i+1} saved as tree_{i+1}.png")