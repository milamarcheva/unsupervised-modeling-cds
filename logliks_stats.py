import shutil
import os
import sys
import statistics
from scipy.stats import ttest_ind, ttest_rel
import matplotlib.pyplot as plt
import logging

logging.basicConfig(filename='norm_vs_optim.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s')
logging.warning('This message will get logged on to a file')

# dr = sys.argv[1] #run with argument
dr = "/Users/milamarcheva/Desktop/unsupervised-modeling/outputs/20072023/"

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
print('lowest: ', highest, 'index ', ind, 'dir: ', logs[ind])
