import math
import numpy as np

del_u = 0
del_v = 0
important_features = {}
unimportant_features = {}
common_imp_unimp = {}
updated_global_list = {}
first_time = True


def compute(global_list):
    global important_features, unimportant_features, del_u, del_v, common_imp_unimp, updated_global_list, first_time
    if first_time:
        sorted_feature_list = dict(sorted(global_list.items(), key=lambda item: item[1], reverse=True))
        n_important = math.isqrt(len(sorted_feature_list))
        items = list(sorted_feature_list.items())
        important_features = dict(items[:n_important])
        unimportant_features = dict(items[n_important:])
        first_time = False
    else:
        for feat in global_list.keys():
            if feat in important_features:
                important_features[feat] = global_list[feat]
            elif feat in unimportant_features:
                unimportant_features[feat] = global_list[feat]

    unimp_feature_size_before = len(unimportant_features)
    imp_feature_size_before = len(important_features)

    remove_low_unimportant()
    promote_features()
    common_imp_unimp = {**important_features, **unimportant_features}

    del_u = len(important_features) - imp_feature_size_before
    del_v = len(unimportant_features) - unimp_feature_size_before

    print("del-u", del_u)
    print("del-v", del_v)
    print()
    return common_imp_unimp, len(important_features), len(unimportant_features), del_u, del_v


def remove_low_unimportant():
    global unimportant_features
    if not unimportant_features:
        return
    values = np.array(list(unimportant_features.values()))
    mean = np.mean(values)
    std = np.std(values)
    threshold = mean - (2 * std)
    # print("length of Unimp feature Before Pruning--------------", len(unimportant_features))
    # print("unnnnnnnnnnnnnnnnnnnn", unimportant_features)
    # print()
    print("This was the threshold value", threshold)
    print()
    unimportant_features_len = len(unimportant_features)
    unimportant_features = {k: v for k, v in unimportant_features.items() if v > threshold}
    if unimportant_features_len == len(unimportant_features):
        min_key = min(unimportant_features, key=unimportant_features.get)
        del unimportant_features[min_key]
        print("This RAN Instead")
    # print("length of Unimp feature After Pruning---------------", len(unimportant_features))
    # print("unnnnnnnnnnnnnnnnnnnn", unimportant_features)
    # print()


def promote_features():
    global important_features, unimportant_features
    if not unimportant_features:
        return
    # print("length of Unimp feature Before Promotion--------------", len(unimportant_features))
    # print("unnnnnnnnnnnnnnnnnnnn", unimportant_features)
    # print()
    min_important_value = min(important_features.values()) if important_features else -float('inf')
    promoted = {k: v for k, v in unimportant_features.items() if v >= min_important_value}
    for k in promoted:
        important_features[k] = unimportant_features.pop(k)

    # print("length of Unimp feature After Promotion--------------", len(unimportant_features))
    print()
    print("Unimporant Feature List after pruning and promotion", unimportant_features)
    print()