import shutil
import os
import sys
import statistics
from scipy.stats import ttest_ind, ttest_rel
import matplotlib.pyplot as plt
import numpy as np

# dr = sys.argv[1] #run with argument
dr = "/Users/milamarcheva/Desktop/unsupervised-modeling/outputs/02082023/anchor1/"
dr = "/Users/milamarcheva/Desktop/unsupervised-modeling/outputs/out2025/line4/"

logs = []
for root, dirs, files in os.walk(dr):
    # print(root)
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

def catch(func, handle=lambda e : e, *args, **kwargs):
    try:
        return func(*args, **kwargs)
    except Exception as e:
        return [-100]

# logZ = [catch(log2dict_new(l)[1]) for l in logs]
final_logZ = [log2dict_new(l)[1][-1][0] for l in logs]
highest = max(final_logZ)
ind = final_logZ.index(max(final_logZ))

print("Lenght: ", len(final_logZ))
print('highest: ', highest, 'index ', ind, 'dir: ', logs[ind])
print('lowest: ', min(final_logZ), 'index ', final_logZ.index(min(final_logZ)), 'dir: ', logs[final_logZ.index(min(final_logZ))])

print('mean: ', np.average(final_logZ), ' std: ', np.std(final_logZ))

final_rand_vals= final_logZ
final_Brown_val = -4.025
counts, edges, bars = plt.hist(final_rand_vals, bins=10)

# plt.hist(final_rand_vals, bins=8)
plt.axvline(final_Brown_val, color='k', linestyle='dashed', linewidth=1)
min_ylim, max_ylim = plt.ylim()
# plt.text(-4.105, max_ylim*0.7, "Brown's order: {:.3f}".format(final_Brown_val))
plt.bar_label(bars)
plt.xlabel("Final MLL")
# plt.ylabel("Count of logZ in each bin")

plt.show()