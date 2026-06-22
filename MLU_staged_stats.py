import shutil
import os
import sys
import statistics
from scipy.stats import ttest_ind, ttest_rel, mannwhitneyu
import matplotlib.pyplot as plt
import logging
import math

final_Brown_val = -4.00
# dr = "/Users/milamarcheva/Desktop/unsupervised-modeling/outputs/21032024/"
dr = "/Users/milamarcheva/Desktop/unsupervised-modeling/outputs/out2025/"

logs = []
logs_MLU = []
logs_random = []
for root, dirs, files in os.walk(dr):
    path = root.split(os.sep)
    # print(path)
    # if path[-1].startswith('MLUStages'):
    if "MLUcompliant" in path:
        for file in files:
            if file == "log":
                # print(dr + os.path.basename(root) + '/'+ file)
                logs_MLU.append(dr + "MLUcompliant/" +os.path.basename(root) + '/'+ file)
    else:
        for file in files:
            if file == "log":
                logs_random.append(dr + "noncompliant/" + os.path.basename(root) + '/'+ file)

print("len logsMLU:",len(logs_MLU))
print("len logs random:",len(logs_random))

def log2tuple(pathname):
    # print(pathname)
    basepath = '/Users/milamarcheva/Desktop/unsupervised-modeling/outputs/20032024/'
    basepath = "/Users/milamarcheva/Desktop/unsupervised-modeling/outputs/out2025/"
    id = 0
    # print(pathname)
    if pathname[len(basepath):][0] == 'M':
        # id = int(pathname[len('/Users/milamarcheva/Desktop/unsupervised-modeling/outputs/20032024/MLUStages_order'):len(pathname)-8])
        offset1 = len("/Users/milamarcheva/Desktop/unsupervised-modeling/outputs/out202520250503_MorphBrownStagesOrderedNoUnlocking_dilute2_order5555_grouping5_")
        id = pathname[offset1:len(pathname)-8]
    else:
        offset1 = len("/Users/milamarcheva/Desktop/unsupervised-modeling/outputs/out2025/noncompliant/20250505_MorphBrownStagesOrderedNoUnlocking_dilute2_order5555_grouping5_")
        # id = int(pathname[len('/Users/milamarcheva/Desktop/unsupervised-modeling/outputs/20032024/randomStages_order'):len(pathname)-8])
        id = pathname[offset1:len(pathname)-8]

    file1 = open(pathname, 'r')
    lines = file1.readlines()
    file1.close()

    lines = [l.strip() for l in lines]

    order = ''
    for l in lines:
        if l.startswith('orderStr'):
            order = l[len('orderStr List(')-1: len(l)-3]
            break

    lines_tokenised = [l.split() for l in lines]
    final_logZ = 0

    for l in lines_tokenised[::-1]:
        if l[0] == 'train:':
            final_logZ = float(l[3][:-1])
            break
    return (id, order, final_logZ)


logs = logs_MLU
ids = [0]*len(logs)
orders = ['']*len(logs)
final_logZs = [0]*len(logs)

# print(logs[0])
for i in range(len(logs_MLU)):
    ids[i] , orders[i], final_logZs[i] =  log2tuple(logs[i])

final_MLU_vals= final_logZs

ids_r = [0]*len(logs_random)
orders_r= ['']*len(logs_random)
final_random_vals = [0]*len(logs_random)

for i in range(len(logs_random)):
    ids_r[i], orders_r[i], final_random_vals[i] = log2tuple(logs_random[i])

for i in range(len(final_MLU_vals)):
    # if final_MLU_vals[i]>final_Brown_val:
    #     print(logs[i].split('/')[-2].split('_')[-1][5:-4])
    #
    if final_MLU_vals[i]<-4.6:
        print(final_logZs[i], ids[i], orders[i])
    # if final_MLU_vals[i]<-4.6:
    #     print(final_logZs[i], ids[i], orders[i])
    # if orders[i].endswith('IR3') and final_MLU_vals[i]>-4.6:
    #     print(orders[i], final_MLU_vals[i])

# for i in range(len(final_random_vals)):
#     if final_random_vals[i]<-5:
#         print(final_random_vals[i], ids_r[i], orders_r[i])

# if final_MLU_vals[i]<-4.6:
#     print(final_logZs[i], ids[i], orders[i])
# if final_MLU_vals[i]<-4.6:
#     print(final_logZs[i], ids[i], orders[i])
# if orders[i].endswith('IR3') and final_MLU_vals[i]>-4.6:
#     print(orders[i], final_MLU_vals[i])

print('MLU: ', statistics.mean(final_MLU_vals), statistics.variance(final_MLU_vals), statistics.stdev(final_MLU_vals))
print('non-MLU: ', statistics.mean(final_random_vals), statistics.variance(final_random_vals),statistics.stdev(final_random_vals) )
print(ttest_ind(final_MLU_vals, final_random_vals, equal_var=False))
print(mannwhitneyu(final_MLU_vals, final_random_vals,))

def cohend(d1, d2):
    # calculate the size of samples
    n1, n2 = len(d1), len(d2)
    # calculate the variance of the samples
    s1, s2 = statistics.variance(d1), statistics.variance(d2)
    # calculate the pooled standard deviation
    s = math.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
    # calculate the means of the samples
    u1, u2 = statistics.mean(d1), statistics.mean(d2)
    # calculate the effect size
    return (u1 - u2) / s

# pooled_stdev = sqrt(((len(final_MLU_vals) - 1) . statistics.stdev(final_MLU_vals))^2 + (len(final_random_vals) - 1) . statistics.stdev(final_random_vals))^2) / (len(final_MLU_vals) + len(final_random_vals) - 2))
# effect_size = (statistics.mean(final_MLU_vals) - statistics.mean(final_random_vals))/pooled_stdev #cohen's d
print('effect size: ',cohend(final_MLU_vals, final_random_vals))


# counts, edges, bars = plt.hist(final_MLU_vals, bins=20)
plt.hist(final_MLU_vals, bins=50, edgecolor='blue', alpha = 0.5, label ='Brown-compliant')
plt.hist(final_random_vals, bins=50, edgecolor='red', alpha = 0.3, label = 'non-Brown-compliant')
# plt.hist(final_rand_vals, bins=8)
plt.axvline(final_Brown_val, color='k', linestyle='dashed', linewidth=1)
min_ylim, max_ylim = plt.ylim()
plt.text(final_Brown_val*1.149, max_ylim*0.96, "Mean order of acquisition {:.3f}".format(final_Brown_val))
# plt.bar_label(bars)
plt.xlabel("Final LL")
# plt.ylabel("Division of final LL in bins")
plt.legend()

plt.show()





