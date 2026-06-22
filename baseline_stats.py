import shutil
import os
import sys
import statistics
from scipy.stats import ttest_ind, ttest_rel
import matplotlib.pyplot as plt
import logging


# logging.basicConfig(filename='norm_vs_optim.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s')
# logging.warning('This message will get logged on to a file')


dr = "/Users/milamarcheva/Desktop/unsupervised-modeling/outputs/25072023/baseline/"
logs = []
for root, dirs, files in os.walk(dr):
    path = root.split(os.sep)
    for file in files:
        if file == "log":
            logs.append(dr + os.path.basename(root) + '/'+ file)

# print("len logs:",len(logs), logs)

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

# logZs = [log2dict_new(l)[1] for l in logs]

vals_BrownOrderNoStages = []
vals_BrownOrderMLUStages_dilute2 = []
vals_BrownOrderStages_dilute2 = []
vals_BrownShuffledNoStages = []
vals_BrownOrderShuffledStages_dilute2 = []
vals_shuffled = []
vals_sortedLen = []
vals_reverseSortedLen = []
vals_sortedLogprobs = []
vals_reverseSortedLogprobs = []
vals_sortedLogprobsNorm = []
vals_reverseSortedLogprobsNorm = []
vals_randomstages = []
for l in logs:
    if "BrownOrderNoStages" in l:
        vals_BrownOrderNoStages = log2dict_new(l) #[item for sublist in log2dict_new(l)[1] for item in sublist]
    elif "BrownOrderMLUStages_dilute2" in l:
        vals_BrownOrderMLUStages_dilute2 = log2dict_new(l) #[item for sublist in log2dict_new(l)[1] for item in sublist]
    elif "BrownOrderStages_dilute2" in l:
        vals_BrownOrderStages_dilute2 = log2dict_new(l) #[item for sublist in log2dict_new(l)[1] for item in sublist]
    elif "BrownOrderShuffledStages_dilute2" in l:
        vals_BrownOrderShuffledStages_dilute2 = log2dict_new(l) #[item for sublist in log2dict_new(l)[1] for item in sublist]
    elif "BrownShuffledNoStages" in l:
        vals_BrownShuffledNoStages = log2dict_new(l) #[item for sublist in log2dict_new(l)[1] for item in sublist]
    elif "shuffled" in l:
        vals_shuffled = log2dict_new(l) #[item for sublist in log2dict_new(l)[1] for item in sublist]
    elif "sortedLen" in l:
        vals_sortedLen = log2dict_new(l) #[item for sublist in log2dict_new(l)[1] for item in sublist]
    elif "reverseSortedLen" in l:
        vals_reverseSortedLen = log2dict_new(l) #[item for sublist in log2dict_new(l)[1] for item in sublist]
    elif "sortedLogprobsNorm" in l:
        vals_sortedLogprobsNorm = log2dict_new(l) #[item for sublist in log2dict_new(l)[1] for item in sublist]
    elif "reverseSortedLogprobsNorm" in l:
        vals_reverseSortedLogprobsNorm = log2dict_new(l) #[item for sublist in log2dict_new(l)[1] for item in sublist]
    elif "sortedLogprobs" in l:
        vals_sortedLogprobs = log2dict_new(l) #[item for sublist in log2dict_new(l)[1] for item in sublist]
    elif "reverseSortedLogprobs" in l:
        vals_reverseSortedLogprobs = log2dict_new(l) #[item for sublist in log2dict_new(l)[1] for item in sublist]

# vals_randomstages = log2dict_new("/Users/milamarcheva/Desktop/unsupervised-modeling/outputs/28072023/tag28072023_MorphBrownStages_dilute2_order361.out/log")

# print("shuffled", "sortedLogprobs", ttest_ind(vals_shuffled, vals_sortedLogprobs))
# print("shuffled", "sortedLen", ttest_ind(vals_shuffled, vals_sortedLen))
# print("shuffled", "reverseSortedLogprobs", ttest_ind(vals_shuffled, vals_reverseSortedLogprobs))
# print("shuffled", "reverseSortedLogprobs", ttest_ind(vals_shuffled, vals_reverseSortedLogprobs))
# print("shuffled", "reverseSortedLen", ttest_ind(vals_shuffled, vals_reverseSortedLen))
# print('\n')
# print("sortedLogprobs", "reverseSortedLogprobs", ttest_ind(vals_sortedLogprobs, vals_reverseSortedLogprobs))
# print("sortedLen", "reverseSortedLen", ttest_ind(vals_sortedLen, vals_reverseSortedLen))
# print('\n')
#
#
# print("BrownShuffledNoStages", "BrownOrderStages_dilute2", ttest_ind(vals_BrownShuffledNoStages, vals_BrownOrderStages_dilute2))
# print("BrownShuffledNoStages", "BrownOrderNoStages", ttest_ind(vals_BrownShuffledNoStages, vals_BrownOrderNoStages))
# print("BrownShuffledNoStages", "BrownOrderMLUStages_dilute2", ttest_ind(vals_BrownShuffledNoStages, vals_BrownOrderMLUStages_dilute2))
# print("BrownOrderStages_dilute2", "BrownOrderMLUStages_dilute2", ttest_ind(vals_BrownOrderStages_dilute2, vals_BrownOrderMLUStages_dilute2))
# print("BrownOrderStages_dilute2", "BrownOrderShuffledStages_dilute2", ttest_ind(vals_BrownOrderStages_dilute2, vals_BrownOrderShuffledStages_dilute2))




#Are the normalized random orders stat sig diff from the normalized brown order
# print("Normalized")
# for i in range(1, 101):
#     rand = normalized_logZ[i-1]
#     vals_rand = [item for sublist in rand for item in sublist]
#     res = ttest_ind(vals_Brown_norm, vals_rand)
#
#     # print('Random order ' + str(i) + ' sigtest results: ' + str(res))
#     sig = False
#     if res.pvalue<0.05:
#         sig = True
#         sig_dif[i-1] = 1
#
#     if sig:
#         print('Random order ' + str(i) + ': significant difference')
#     else:
#         print('Random order ' + str(i) + ': NOT significant difference')
#
#     print(res, '\n')
#
#
# print("Normalized: ",sum(sig_dif))
#
#
#
def plot_lastlogZ_iter_debug(tuples, n = []):

    # print(tuples[0])
    fig,ax = plt.subplots(figsize=(8, 8))

    #new added 04/08
    # tuples = [log2dict_new(t) for t in tuples]


    for i in range(len(tuples)):
        # print(i)
        # print(tuples[i])
        olditers = tuples[i][0]
        newiters = [olditers[0]]
        # print(newiters)
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        # print(colors)
        styles = ['-', '--', '-.', ':','-.', '--',  ':','-.', '--',  ':','-.', '--']
        colormap = {}
        stylemap = {}
        index = 0
        # for label in ['shuffled', 'sorted', 'reversed', 'ordered', 'stagedMorph', 'stagedMLU', 'randomStaged', 'reverseMLUStaged', 'reverseMorphStaged', 'functionalFirst', 'nounyVerby', 'verbyNouny']:

        #uncomment whichever type of training you want to print
        # for label in ['reversed', 'ordered', 'stagedMorph', 'stagedMLU', 'randomMorphStaged', 'reverseMLUStaged', 'reverseMorphStaged', 'functionalFirst', 'nounyVerby', 'verbyNouny']:
        # for label in ['stagedMorph', 'stagedMorphRandom', 'stagedMorphReverse']:
        # for label in ['stagedMLU', 'reverseMLUStaged', ]:
        #     colormap[label] = colors[index]
        #     stylemap[label] = styles[index]
        #     index += 1
        # print(colormap)
        for j in range(1,len(olditers)):
            newiters.append(max(1,olditers[j]-olditers[j-1])+newiters[-1])
        ax.plot(newiters,  tuples[i][1], linewidth=1, label=n[i], ) #color = colormap[n[i]]#ls = stylemap[n[i]]) # ewm computes moving average
        # ax.plot(range(0, len(ds[i].values())),  last_vals[i], linewidth=1, label=n[i]) # ewm computes moving average

        # for var in tuples[i][1]:
        # plt.annotate('%0.3f' % tuples[i][1][-1][0], xy=(0.93, tuples[i][1][-1][0]), xytext=(6, 0),
        #              xycoords=('axes fraction', 'data'), textcoords='offset points', fontsize=16)

    # ax.set_title('loglik over iterations', fontsize=14)
    ax.set_xlabel('Utterances', fontsize=16)
    ax.set_ylabel('logZ', fontsize=16)
    # ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), shadow=True, ncol=2)
    #(bbox_to_anchor=(1,.5)
    plt.legend( loc='best', frameon=True, fontsize=16)

    # for var in (y1, y2):
    #   plt.annotate('%0.2f' % var.max(), xy=(1, var.max()), xytext=(8, 0),
    #                xycoords=('axes fraction', 'data'), textcoords='offset points')

    plt.show()

# print(normalized_logs[0], optimized_logs[0])

# vals_BrownOrderNoStages = []
# vals_BrownOrderMLUStages_dilute2 = []
# vals_BrownOrderStages_dilute2 = []
# vals_BrownShuffledNoStages = []
# vals_BrownOrderShuffledStages_dilute2 = []
#
# vals_shuffled = []
# vals_sortedLen = []
# vals_reverseSortedLen = []
# vals_sortedLogprobs = []
# vals_reverseSortedLogprobs = []


# plot_lastlogZ_iter_debug([vals_BrownOrderStages_dilute2, vals_BrownOrderShuffledStages_dilute2, vals_BrownShuffledNoStages,], # vals_randomstages,vals_BrownOrderMLUStages_dilute2
#                          n=['BrownOrder+BrownStages','Shuffled+BrownStages', 'Shuffled+NoStages', ]) #'BrownOrder+RandomStages', 'BrownOrder+MLUStages'

vals = [vals_BrownShuffledNoStages, vals_sortedLen, vals_reverseSortedLen, vals_sortedLogprobsNorm, vals_reverseSortedLogprobsNorm]

# print([v[-1] for v in vals])
names = ['shuffled', 'sortedLen', 'reverseSortedLen','sortedLogprobsNorm', 'reverseSortedLogprobsNorm']
plot_lastlogZ_iter_debug(vals, n = names)