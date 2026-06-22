import numpy as np

from scipy.stats import ttest_ind, mannwhitneyu
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

random_filenames = ['tag3001_r'+ str(i) +'.outlog' for i in range(1, 101)]

logs_Brown = log2dict_new('log_BrownOrderStaged')
list_log_random = [log2dict_new('random/'+rf) for rf in random_filenames ]

vals_Brown = [item for sublist in logs_Brown[1] for item in sublist]
sig_dif = [0]*100

for i in range(1, 101):
    rand = list_log_random[i-1]
    vals_rand = [item for sublist in rand[1] for item in sublist]


    res = ttest_ind(vals_Brown, vals_rand)

    # print('Random order ' + str(i) + ' sigtest results: ' + str(res))
    sig = False
    if res.pvalue<0.05:
        sig = True
        sig_dif[i-1] = 1

    if sig:
        print('Random order ' + str(i) + ': significant difference')
    else:
        print('Random order ' + str(i) + ': NOT significant difference')

    print(res, '\n')


sum(sig_dif)