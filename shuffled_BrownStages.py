import shutil
import os
import sys
import statistics
from scipy.stats import ttest_ind, ttest_rel
import matplotlib.pyplot as plt
import logging

dr = "/Users/milamarcheva/Desktop/unsupervised-modeling/outputs/31072023/"
logs = []
for root, dirs, files in os.walk(dr):
    path = root.split(os.sep)
    for file in files:
        if file == "log":
            logs.append(dr + os.path.basename(root) + '/'+ file)

print("len logs:",len(logs))

def log2dict_new(pathname):
    file1 = open(pathname, 'r')
    lines = file1.readlines()
    file1.close()
    lines = [l.strip() for l in lines]
    lines_tokenised = [l.split() for l in lines]
    logZs = []
    examples = []
    for l in lines_tokenised:
        if l[0] == 'Example':
            logZ = (l[5][:-1])
            if logZ!= "NaN":
                logZs.append([float(logZ)]) #remove the comma
                exampleno = int(l[1].split('/')[0])
                examples.append(exampleno)
    return examples, logZs


final_logZs = [log2dict_new(l)[1][-1][0] for l in logs ]


final_Brown_val = -4.030
final_rand_vals= final_logZs
counts_bigger = [final_Brown_val>v for v in final_rand_vals]
print('max value: ', max(final_rand_vals))

print('brown is bigger than ', sum(counts_bigger))
counts, edges, bars = plt.hist(final_rand_vals, bins=10)

# plt.hist(final_rand_vals, bins=8)
plt.axvline(final_Brown_val, color='k', linestyle='dashed', linewidth=1)
min_ylim, max_ylim = plt.ylim()
# plt.text(final_Brown_val*1.11, max_ylim*0.7, 'Brown order: {:.2f}'.format(final_Brown_val))
# plt.text(0.5, 0.9 , 'Mila')
plt.bar_label(bars)
plt.xlabel("Final logZ")
plt.ylabel("Count of logZ in each bin")

plt.show()






