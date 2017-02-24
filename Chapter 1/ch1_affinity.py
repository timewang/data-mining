import numpy as np
dataset_filename = "affinity_dataset.txt"
X = np.loadtxt(dataset_filename)
n_samples, n_features = X.shape
print("This dataset has {0} samples and {1} features".format(n_samples, n_features))

#是面包、牛奶、奶酪、苹果、香蕉
print(X[:5])

# The names of the features, for your reference.
features = ["bread", "milk", "cheese", "apples", "bananas"]

# First, how many rows contain our premise: that a person is buying apples
num_apple_purchases = 0
for sample in X:
    if sample[3] == 1:  # This person bought Apples
        num_apple_purchases += 1
print("{0} people bought Apples".format(num_apple_purchases))

# How many of the cases that a person bought Apples involved the people purchasing Bananas too?
# Record both cases where the rule is valid and is invalid.
rule_valid = 0
rule_invalid = 0
for sample in X:
    if sample[3] == 1:  # This person bought Apples
        if sample[4] == 1:
            # This person bought both Apples and Bananas
            rule_valid += 1
        else:
            # This person bought Apples, but not Bananas
            rule_invalid += 1
print("{0} cases of the rule being valid were discovered".format(rule_valid))
print("{0} cases of the rule being invalid were discovered".format(rule_invalid))

# Now we have all the information needed to compute Support and Confidence
support = rule_valid  # The Support is the number of times the rule is discovered.
confidence = rule_valid / num_apple_purchases # 置信度，买了苹果又买了香蕉的人，所有买了苹果的人的百分比
print("The support is {0} and the confidence is {1:.3f}.".format(support, confidence))
# Confidence can be thought of as a percentage using the following:
print("As a percentage, that is {0:.1f}%.".format(100 * confidence))

from collections import defaultdict
# Now compute for all possible rules
valid_rules = defaultdict(int) # 规则应验
invalid_rules = defaultdict(int) # 规则无效
num_occurences = defaultdict(int) # 每种商品有多少人买

for sample in X:
    for premise in range(n_features):
        if sample[premise] == 0: continue # 什么都没买
        # Record that the premise was bought in another transaction 买了 a
        num_occurences[premise] += 1
        for conclusion in range(n_features):
            if premise == conclusion:  # It makes little sense to measure if X -> X. 如果是同一种货物
                continue
            if sample[conclusion] == 1:
                # This person also bought the conclusion item 既买了 a 又买了 b
                valid_rules[(premise, conclusion)] += 1
            else:
                # This person bought the premise, but not the conclusion 买了 a 但是没买 b
                invalid_rules[(premise, conclusion)] += 1
support = valid_rules
confidence = defaultdict(float)
# 每种规则的可信度，
for premise, conclusion in valid_rules.keys():
    confidence[(premise, conclusion)] = valid_rules[(premise, conclusion)] / num_occurences[premise]


for premise, conclusion in confidence:
    premise_name = features[premise]
    conclusion_name = features[conclusion]
    print("Rule: If a person buys {0} they will also buy {1}".format(premise_name, conclusion_name))
    print(" - Confidence: {0:.3f}".format(confidence[(premise, conclusion)]))
    print(" - Support: {0}".format(support[(premise, conclusion)]))
    print("")

def print_rule(premise, conclusion, support, confidence, features):
    premise_name = features[premise]
    conclusion_name = features[conclusion]
    print("Rule: If a person buys {0} they will also buy {1}".format(premise_name, conclusion_name))
    print(" - Confidence: {0:.3f}".format(confidence[(premise, conclusion)]))
    print(" - Support: {0}".format(support[(premise, conclusion)]))
    print("")

# 既买牛奶又买苹果的概率
premise = 1
conclusion = 3
print_rule(premise, conclusion, support, confidence, features)

# Sort by support
from pprint import pprint
pprint(list(support.items()))

from operator import itemgetter
sorted_support = sorted(support.items(), key=itemgetter(1), reverse=True)

for index in range(5):
    print("Rule #{0}".format(index + 1))
    (premise, conclusion) = sorted_support[index][0]
    print_rule(premise, conclusion, support, confidence, features)

sorted_confidence = sorted(confidence.items(), key=itemgetter(1), reverse=True)

for index in range(5):
    print("Rule #{0}".format(index + 1))
    (premise, conclusion) = sorted_confidence[index][0]
    print_rule(premise, conclusion, support, confidence, features)