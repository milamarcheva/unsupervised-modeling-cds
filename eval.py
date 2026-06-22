import shutil
import os
import sys
import statistics
from scipy.stats import ttest_ind, ttest_rel
import matplotlib.pyplot as plt
import logging
import numpy as np
import pandas as pd
import seaborn as sns



def kl_divergence(p, q):
    return np.sum(np.where(p != 0, p * np.log(p / q), 0))

def symmetric_kl(p,q):
    return kl_divergence(p,q) + kl_divergence(q,p)

def js_divergence(p, q):
    m = 0.5 * (p + q)
    return 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)

def get_tokenised_lines(pathname):
    fileb = open(pathname, 'r')
    linesb = fileb.readlines()
    fileb.close()

    linesb_stripped = [l.strip() for l in linesb]
    lines_tokenised = [l.split() for l in linesb_stripped]

    return lines_tokenised

ms = ["underwear", "measles", "airplane", "guitar", "racket", "movie", "sword", "sofa", "ant", "motorcycle", "hotel",
      "shirt", "apartment", "museum", "necklace", "spoon", "tea", "pear", "pepper", "peach",
      "pen", "cat", "saw", "flower", "newspaper", "fork", "dog", "bird", "grape", "necktie", "butterfly", "radish",
      "tent", "coffee", "chair", "net", "boot", "cup", "skate", "paper", "plate", "truck", "cow", "bean", "drum",
      "magazine", "onion", "tree", "bus", "horse", "weight", "knife", "novel", "potato", "lamp", "drill", "sock",
      "milk", "piano", "ring", "gun", "scarf", "pencil", "apple", "bed", "fish", "ball", "bomb", "mouse", "shoe",
      "hammer", "table", "grain", "bread", "watch", "drink", "eat", "bite", "suck", "spit", "vomit", "blow",
      "breathe", "laugh", "see", "hear", "know", "think", "smell", "fear", "sleep", "live", "die", "kill", "fight",
      "hunt", "hit", "cut", "split", "scratch", "dig", "swim", "fly", "walk", "come", "lie", "sit", "stand", "turn",
      "fall", "give", "hold", "squeeze", "rub", "wash", "wipe", "pull", "push", "throw", "tie", "sew", "count",
      "say", "sing", "play", "float", "freeze", "swell", "am", "are", "is", "was", "were", "in",
      "on", "ing", "ed", "saw", "ate", "drank", "did", "knew", "bit", "came", "fell", "froze", "gave", "sang", "swam",
      "threw", "a", "the", "has", "does", "goes", "s", "'re", "'s", "'s", "'m" ]

ms_func =  ["in", "on", "ing", "ed", "a", "the", "has", "does", "goes", "s", "'re", "'s", "'m", "am", "are", "is", "was", "were" ]

def eval(dr, ms, data_path, png_path, no_stages = False, stagesMLU = False, onlyNV = False):
    emissions_path = dr + "stageCAUX.emissions"
    pred_path  = dr+ "stageCAUX.train.pred.0"

    if no_stages:
        emissions_path = dr + "stage2.emissions"
        pred_path  = dr+ "stage2.train.pred.0"

    if stagesMLU:
        emissions_path = dr + "stageV.emissions"
        pred_path  = dr+ "stageV.train.pred.0"

    if onlyNV:
        emissions_path = dr + "stageAllMorphemes.emissions"
        pred_path  = dr+ "stageAllMorphemes.train.pred.0"

    c_lines = get_tokenised_lines(pred_path)
    counts_c = [0]*10
    for l in c_lines:
        for t in l:
            counts_c[int(t)]+=1
    total_c = sum([len(l) for l in c_lines])
    p_c = dict(zip(list(range(10)), [x/total_c for x in counts_c]))
    print(total_c, counts_c)
    print(p_c)
    print("")


    brown_tokenised_lines = get_tokenised_lines(data_path)
    ms_counts = {}
    for l in brown_tokenised_lines:
        for t in l:
            if t in ms:
                if t in ms_counts:
                    ms_counts[t] = ms_counts[t]+1
                else:
                    ms_counts[t] = 1

    total_counts= sum([len(l) for l in brown_tokenised_lines])
    p_m = dict(zip(ms, [ms_counts[m]/total_counts for m in ms]))

    emission_tokenised_lines = get_tokenised_lines(emissions_path)
    p_m_c = {}
    for l in emission_tokenised_lines:
        if l:
            c = l[1]
            m = l[2]
            p = float(l[3])
            if m in p_m_c:
                p_m_c[m] = p_m_c[m]+[p]
            else:
                p_m_c[m] = [p]

    # print(p_m_c)

    p_c_m = {}
    for c in range(10):
        pcm_forc= {}
        for m in ms:
            p_c_m_val = p_m_c[m][c] * p_c[c]/p_m[m]
            pcm_forc[m] = p_c_m_val

            # wsun
            p_m_c[m][c] = p_m_c[m][c] * p_c[c] # CAUTION: p_m_c is changed from a
            #          conditional probability to a joint probability

        p_c_m[c] = pcm_forc

    # wsun
    marginal_m = {}
    for m in ms:
        marginal_m[m] = 0
        for c in range(10):
            marginal_m[m] = marginal_m[m] + p_m_c[m][c];

    kl1 = {}
    for m1 in ms_func:
        kl1[m1] = {}
        for m2 in ms_func:
            # print(m1,m2)
            kl_m1m2 = 0
            kl_m2m1 = 0
            for c in range(10):
                kl_m1m2+=kl_divergence(p_m_c[m1][c]/marginal_m[m1], p_m_c[m2][c]/marginal_m[m2])
                kl_m2m1+=kl_divergence(p_m_c[m2][c]/marginal_m[m2], p_m_c[m1][c]/marginal_m[m1])

            # kl1[m1][m2] = kl_m1m2
            kl1[m1][m2] = (kl_m1m2 + kl_m2m1)/2
            # print(f'{m1} || {m2}', kl_m1m2)
            # print(f'{m2} || {m1}', kl_m2m1)
            # print('symmetric KL', kl_m1m2+kl_m2m1)
            # print(js_divergence(p_c_m[c][m1], p_c_m[c][m2]))
            # print("")


    df = pd.DataFrame.from_dict(kl1)

    return df

def plot_dual(df1, df2):
    # Create a mask
    mask1 = np.triu(np.ones_like(df1, dtype=bool))
    # np.fill_diagonal(mask1, False)

    mask2 = np.tril(np.ones_like(df2, dtype=bool))

    vmin = min(df1.values.min(), df2.values.min())
    vmax = max(df1.values.max(), df2.values.max())

    plt.figure(figsize=(12.6,6))

    sns.heatmap(df1,  vmin=vmin, vmax=vmax, mask=mask1, center=0, annot=True, cbar_kws = {"pad": 0.001},  )#fmt='.2f'
    sns.heatmap(df2, vmin=vmin, vmax=vmax, cbar=False, mask=mask2, center=0, annot=True,  fmt='.2f')#fmt='.2f'



    # fig, axs = plt.subplots()
    #
    # sns.heatmap(df1, annot=True, mask=mask1, fmt ='.3f', cbar=False, ax=axs[0], vmin=vmin)
    # sns.heatmap(df2, annot=True, mask=mask1, fmt ='.3f', yticklabels=False, cbar=False, ax=axs[1], vmax=vmax)
    #
    # fig.colorbar(axs[1].collections[0], cax=axs[2])

    # plt.savefig(png_path)

    plt.tight_layout()
    plt.show()

def plot_single(df1):
    # Create a mask
    mask1 = np.triu(np.ones_like(df1, dtype=bool))
    # np.fill_diagonal(mask1, False)


    vmin = df1.values.min()
    vmax = df1.values.max()

    plt.figure(figsize=(12.6,6))

    sns.heatmap(df1,  vmin=vmin, vmax=vmax, mask=mask1, center=0, annot=True, cbar_kws = {"pad": 0.001},  )#fmt='.2f'
    # sns.heatmap(df2, vmin=vmin, vmax=vmax, cbar=False, mask=mask2, center=0, annot=True,  fmt='.2f')#fmt='.2f'



    # fig, axs = plt.subplots()
    #
    # sns.heatmap(df1, annot=True, mask=mask1, fmt ='.3f', cbar=False, ax=axs[0], vmin=vmin)
    # sns.heatmap(df2, annot=True, mask=mask1, fmt ='.3f', yticklabels=False, cbar=False, ax=axs[1], vmax=vmax)
    #
    # fig.colorbar(axs[1].collections[0], cax=axs[2])

    # plt.savefig(png_path)

    plt.tight_layout()
    plt.show()

dr = "/Users/milamarcheva/Desktop/unsupervised-modeling/outputs/25072023/baseline/tag25072023_BrownOrderStages_dilute2.out/"
brown_all_sents_path = "/Users/milamarcheva/Desktop/unsupervised-modeling/sample-data/sents_Brown_order.raw"
dr_nostages = "/Users/milamarcheva/Desktop/unsupervised-modeling/outputs/25072023/baseline/tag25072023_BrownOrderNoStages.out/"
dr_MLUstages = "/Users/milamarcheva/Desktop/unsupervised-modeling/outputs/25072023/baseline/tag25072023_BrownOrderMLUStages_dilute2.out/"
dr_anchor1_n10_v10_3_1_0_2_0_12_1 = "/Users/milamarcheva/Desktop/unsupervised-modeling/outputs/02082023/anchor1/tag02082023_MorphBrownStages_dilute2_order5555_grouping5_PREPAnchorIndex0_n10_v10_3_1_0_2_0_12_1.out/"
dr_reduced_nv = "/Users/milamarcheva/Desktop/unsupervised-modeling/outputs/01082023/reducedNounsVerbs/tag01082023_BrownOrderStages_dilute2_nouns5_verbs5.out/"
dr_exNum01 = "/Users/milamarcheva/Desktop/unsupervised-modeling/outputs/27072023/numExamples/tag27072023_BrownOrderStages_dilute2_fracEx0.1.out/"
dr_BrownStagedShuffled = "/Users/milamarcheva/Desktop/unsupervised-modeling/outputs/25072023/baseline/tag25072023_BrownOrderShuffledStages_dilute2.out/"
dr_onlynv = "/Users/milamarcheva/Desktop/unsupervised-modeling/outputs/01082023/tag01082023_BrownOrderStages_onlyNounsVerbs.out/"

# df1 = eval(dr, ms, brown_all_sents_path, '/Users/milamarcheva/Desktop/unsupervised-modeling/figs/heatmap_func_BrownStaged_half.png')
# df2 = eval(dr_nostages, ms, brown_all_sents_path, '/Users/milamarcheva/Desktop/unsupervised-modeling/figs/heatmap_func_NoStages_half.png', no_stages=True)
# plot_dual(df1, df2)

df1 = eval(dr_anchor1_n10_v10_3_1_0_2_0_12_1, ms, brown_all_sents_path, '/Users/milamarcheva/Desktop/unsupervised-modeling/figs/heatmap_func_BrownStaged_half.png')
plot_single(df1 )

# df3 = eval(dr_anchor1_n10_v10_3_1_0_2_0_12_1, ms, brown_all_sents_path, '/Users/milamarcheva/Desktop/unsupervised-modeling/figs/heatmap_func_anchor1_half.png')
# df4 = eval(dr_exNum01, ms, brown_all_sents_path, '/Users/milamarcheva/Desktop/unsupervised-modeling/figs/heatmap_func_exNum01_half.png')
# plot_dual(df3, df4)


# eval(dr_onlynv, ms, brown_all_sents_path, '/Users/milamarcheva/Desktop/unsupervised-modeling/figs/heatmap_func_onlyNV_half.png', onlyNV=True)
#
# eval(dr_nostages, ms, brown_all_sents_path, '/Users/milamarcheva/Desktop/unsupervised-modeling/figs/heatmap_func_NoStages_half.png', no_stages=True)
#
# eval(dr_MLUstages, ms, brown_all_sents_path, '/Users/milamarcheva/Desktop/unsupervised-modeling/figs/heatmap_func_MLUstages_half.png', stagesMLU=True)
#
# eval(dr_anchor1_n10_v10_3_1_0_2_0_12_1, ms, brown_all_sents_path, '/Users/milamarcheva/Desktop/unsupervised-modeling/figs/heatmap_func_anchor1_half.png')
#
# eval(dr_reduced_nv, ms, brown_all_sents_path, '/Users/milamarcheva/Desktop/unsupervised-modeling/figs/heatmap_func_reduced_nv5_half.png')
# eval(dr_exNum01, ms, brown_all_sents_path, '/Users/milamarcheva/Desktop/unsupervised-modeling/figs/heatmap_func_exNum01_half.png')
# eval(dr_BrownStagedShuffled, ms, brown_all_sents_path, '/Users/milamarcheva/Desktop/unsupervised-modeling/figs/heatmap_BrownStagedShuffled_half.png')
#
