# %%

data = [
    ['A', 'B', 'D'],
    ['A', 'B', 'F'],
    ['B', 'C', 'F'],
    ['C', 'E'],
    ['A', 'C', 'F'],
    ['A', 'B', 'E'],
    ['B', 'E', 'F']
]

'''
Implement the aprior algorithm with lift. 
The function should take three parameters: 
support, confidence, and lift and 
return rules that satisfy these conditions.
'''


def sublist(a, b):
    for i in b:
        if i not in a:
            return False
    return True


def apriori(support, confidence, lift):
    support = support * len(data)

    unique_item = set()
    for items in data:
        for item in items:
            unique_item.add((item,))  # add as tuple with single value

    candidates = list(unique_item)  # start with all unique single values
    L_support = []
    single_dict = None

    while candidates:
        s_dict = {c: 0 for c in candidates}
        # check the support for each candidate
        for items in data:
            for key in s_dict:
                if sublist(items, key):
                    s_dict[key] += 1
        if single_dict is None:
            single_dict = s_dict.copy() # for lift
        # check the support
        for key in list(s_dict.keys()):
            if s_dict[key] < support:
                del s_dict[key]

        # all they keys in the dict is now valid, add in to L
        curr_level_rule = sorted(list(s_dict.keys()))
        if len(curr_level_rule) > 0:
            for rule in curr_level_rule:
                to_add = tuple(sorted(rule))
                if to_add not in L_support:
                    L_support.append(tuple(sorted(rule)))

                # prepare for next level
        next_level_candidates = set()
        for i in range(len(curr_level_rule)):
            for j in range(i + 1, len(curr_level_rule)):
                combined_set = set()
                for item in curr_level_rule[i]:
                    combined_set.add(item)
                for item in curr_level_rule[j]:
                    combined_set.add(item)
                next_level_candidates.add(tuple(sorted(combined_set)))

        candidates = list(next_level_candidates)

    #print("L after support is " + str(L_support))

    # singletons can’t generate a rule
    L_support_above_level_2 = []
    for rule in L_support:
        if len(rule) > 1:
            L_support_above_level_2.append(rule)

    candidates = []   # [provide(other items), get(one item)]
    for items in L_support_above_level_2:
        for i in range(len(items)):
            if (items[:i] + items[i + 1 :], [items[i]]) not in candidates:
                candidates.append((items[:i] + items[i + 1 :], (items[i],)))

    L_conf = []
    # check confidence
    for (from_items, to_item) in candidates:
        count_k_under_v_in_data = 0
        count_v_in_data = 0
        for items in data:
            if sublist(items, from_items):
                count_v_in_data += 1
            if sublist(items, from_items + to_item):
                count_k_under_v_in_data += 1
        if count_v_in_data > 0 and count_k_under_v_in_data / count_v_in_data > confidence:
            if (count_k_under_v_in_data / count_v_in_data) / (single_dict[to_item] / len(data)): # lift
                L_conf.append((from_items, to_item))

    print("L after support is " + str(L_conf))
    return L_conf


# %%
# '''
# testing
# 1) Support=0.1, Confidence =0.30, minimum lift = 1
# 2) Support=0.2, Confidence =0.65, minimum lift = 0
# 3) Support=0.2, Confidence =0.65, minimum lift = 1.1
# 4) Support=0.2, Confidence =0.65, minimum lift = 2
# 5) Support=0.2, Confidence=0.50, minimum lift = 1.1
# 6) Support=0.3, Confidence=0.70, minimum lift = 1.1
# '''
results = []
results.append(apriori(0.1, 0.30, 1))
results.append(apriori(0.2, 0.65, 0))
results.append(apriori(0.2, 0.65, 1.1))
results.append(apriori(0.2, 0.65, 2))
results.append(apriori(0.2, 0.5, 1.1))
results.append(apriori(0.3, 0.7, 1.1))

output = open("apriori.csv", "w")
while True:
    has_left = False
    for rules in results:
        if len(rules) > 0:
            rule = rules.pop(0)
            output.write('&'.join(rule[0]) + '=>' + ','.join(rule[1]) + ',')
            if len(rules) > 0:
                has_left = True
        else:
            output.write(',');
    output.write("\n")
    if not has_left:
        break
# %%



