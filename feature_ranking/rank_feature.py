# Tree data (without comments for proper parsing)

# Organize collected data


class LocalGlobalWt:
    # Function to traverse the tree and collect (feature_idx, quality_of_split
    def __init__(self, number_of_features, main_features):
        self.feature_quality_list = []
        self.N = 0
        self.number_of_features = number_of_features
        self.counter = 0
        self.features = main_features


    def compute_feature_importance(self, tree):
        importance = {feature: 0.0 for feature in self.features}

        def traverse(node):
            if node is None or node.value is not None:
                return
            if node.feature in importance:
                importance[node.feature] += node.info_gain
            traverse(node.left)
            traverse(node.right)

        traverse(tree)
        return importance

    def normalized_weight_of_tree(self, oob_list):
        inverse_list = [1 / oob if oob != 0 else 0 for oob in oob_list]
        highest = max(inverse_list)
        normalized_weight = [round(oob / highest, 4) for oob in inverse_list]
        return normalized_weight

    def global_wt(self, feature_wt, normalized_tree_wt):
        # Compute weighted sum for each index
        result = {}
        for i in range(len(feature_wt)):
            for k, v in feature_wt[i].items():
                result[k] = result.get(k, 0) + v * normalized_tree_wt[i]
        print()
        print(result)
        print()
        largest = max(result.values())
        global_wt = {key: round(value / largest, 4) for key, value in result.items()}
        return global_wt
