import shutil
import os
import sys
import statistics
from scipy.stats import ttest_ind, ttest_rel
import matplotlib.pyplot as plt
import logging


# dr = sys.argv[1] #run with argument
dr = "/Users/milamarcheva/Desktop/unsupervised-modeling/outputs/19072023/"
logs = []
for root, dirs, files in os.walk(dr):
    path = root.split(os.sep)
    for file in files:
        if file == "log":
            logs.append(dr + os.path.basename(root) + '/'+ file)

print("len logs:",len(logs))



# subset_logs = [l for l in logs if l.startswith(dr+'tag19072023_MorphBrownStages_dilute2')]
subset_logs = logs
print(len(subset_logs))
normalized_logs = [l for l in subset_logs if l.endswith('_normalizeTrue.out/log')]
optimized_logs = [l for l in subset_logs if l.endswith('_normalizeFalse.out/log')]
normalized_logs.sort()
optimized_logs.sort()

print(normalized_logs[10], optimized_logs[10], len(normalized_logs), len(optimized_logs))

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

normalized_logZ = [log2dict_new(nl)[1] for nl in normalized_logs ]
optimized_logZ = [log2dict_new(nl)[1] for nl in optimized_logs]

#Paired tests of the final values
print("Paired")
last_vals_opt = [l[-1][0] for l in optimized_logZ]
last_vals_norm =  [l[-1][0] for l in normalized_logZ]

print("len ", len(last_vals_norm))

res = ttest_rel(last_vals_norm, last_vals_opt)
print(res)

print("Mean and SD for opt and for norm: ", statistics.mean(last_vals_opt), statistics.stdev(last_vals_opt),
      statistics.mean(last_vals_norm), statistics.stdev(last_vals_norm),)

o_lessthan_n = [o<n for o,n in zip(last_vals_opt, last_vals_norm)]
print("Final norm is less than than final opt: ", sum(o_lessthan_n))

diff_o_n =  [o-n for o,n in zip(last_vals_opt, last_vals_norm)]

print("The avg differnce between optimized and normalized, SD",
      statistics.mean(diff_o_n), statistics.stdev(diff_o_n))



vals_Brown_norm = [item for sublist in normalized_logZ[0] for item in sublist]
vals_Brown_opt = [item for sublist in optimized_logZ[0] for item in sublist]

sig_dif = [0]*100

# #Are the normalized random orders stat sig diff from the normalized brown order
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
# #Are the optimized random orders stat sig diff from the optimized brown order
# sig_dif = [0]*100
#
# print("Optimized")
# for i in range(1, 101):
#     rand = optimized_logZ[i-1]
#     vals_rand = [item for sublist in rand for item in sublist]
#     res = ttest_ind(vals_Brown_opt, vals_rand)
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
# print("Optimized: ",sum(sig_dif))



def plot_lastlogZ_iter_debug(tuples, n = []):

    # print(tuples[0])
    fig,ax = plt.subplots(figsize=(8, 8))

    for i in range(len(tuples)):
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
        plt.annotate('%0.2f' % tuples[i][1][-1][0], xy=(0.945, tuples[i][1][-1][0]), xytext=(6, 0),
                     xycoords=('axes fraction', 'data'), textcoords='offset points')

    # ax.set_title('loglik over iterations', fontsize=14)
    ax.set_xlabel('Examples', fontsize=14)
    ax.set_ylabel('logZ', fontsize=14)
    # ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), shadow=True, ncol=2)
    #(bbox_to_anchor=(1,.5)
    plt.legend( loc='best', frameon=True, fontsize=12)

    # for var in (y1, y2):
    #   plt.annotate('%0.2f' % var.max(), xy=(1, var.max()), xytext=(8, 0),
    #                xycoords=('axes fraction', 'data'), textcoords='offset points')

    plt.show()

# print(normalized_logs[0], optimized_logs[0])

# plot_lastlogZ_iter_debug([log2dict_new(normalized_logs[0]), log2dict_new(optimized_logs[0])],
#                           n=['normalizedBrown','optimizedBrown'])

# final_Brown_val = vals_Brown_opt[-1]
# final_rand_vals= [v[-1][0] for v in optimized_logZ]
# counts_bigger = [final_Brown_val>v for v in final_rand_vals]
# print('brown is bigger than ', sum(counts_bigger))
# counts, edges, bars = plt.hist(final_rand_vals, bins=10)
#
# # plt.hist(final_rand_vals, bins=8)
# plt.axvline(final_Brown_val, color='k', linestyle='dashed', linewidth=1)
# min_ylim, max_ylim = plt.ylim()
# # plt.text(final_Brown_val*1.11, max_ylim*0.7, 'Brown order: {:.2f}'.format(final_Brown_val))
# # plt.text(0.5, 0.9 , 'Mila')
# plt.bar_label(bars)
# plt.xlabel("Final logZ")
# plt.ylabel("Count of logZ in each bin")

plt.show()






