import shutil
import os
import sys
import statistics
from scipy.stats import ttest_ind, ttest_rel
import matplotlib.pyplot as plt
import logging

dr = "/Users/milamarcheva/Desktop/unsupervised-modeling/outputs/28072023/"
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


final_Brown_val = -4.025
final_rand_vals= final_logZs
counts_bigger = [final_Brown_val>v for v in final_rand_vals]

print('max value: ', max(final_rand_vals))

print('brown is bigger than ', sum(counts_bigger))

print(ttest_ind([final_Brown_val], final_rand_vals))

print(statistics.mean(final_rand_vals), statistics.stdev(final_rand_vals) )
print('effect size: ', (statistics.mean(final_rand_vals) - final_Brown_val)/statistics.stdev(final_rand_vals)  )

for i in range(len(logs)):
    if final_rand_vals[i]>final_Brown_val:
        print(logs[i].split('/')[-2].split('_')[-1][5:-4])

counts, edges, bars = plt.hist(final_rand_vals, bins=20)

# plt.hist(final_rand_vals, bins=8)
plt.axvline(final_Brown_val, color='k', linestyle='dashed', linewidth=1)
min_ylim, max_ylim = plt.ylim()
plt.text(final_Brown_val*1.14, max_ylim*0.7, "Brown's order: {:.3f}".format(final_Brown_val))
plt.bar_label(bars)
plt.xlabel("Final MLL")
# plt.ylabel("Division of MLL values in bins")
# plt.legend()

plt.show()






