import os
import random
from itertools import permutations as permut

random.seed(10)

dilute_val = 0
order = 0
norm = False

BGPLUAnchorIndex = 0
BGDEFAnchorIndex = 0
BGPRESAnchorIndex = 0
BGINFAnchorIndex = 0
BGPASTAnchorIndex = 0
BGPREPAnchorIndex = 0



# Baseline results

baseline_script = """scala -cp induction.jar induction.Induction -create -modelType hmm -Options.K 8 -Options.stage2.numIters 1 -Options.stage2.online True -Options.onlinePerm False -Options.stage2.miniBatches True -Options.stage2.miniBatchSize 1 -inputFormat raw -log.maxIndLevel 100 -log.msPerLine 10 -Options.useBG True """

#OUTPUTS FOR TEST DATA ACCURACY
# os.system((baseline_script + """ -Options.inductionType morph -inputPaths sample-data/bg/bg_hmm_tokenised_morphLemma_duplicates_shuffled_seed42_plusTest.txt -inputFormat raw -execDir outputs/bg/20260622/ShuffledNoStagesWithGradualUnlocking.out -overwrite True -Options.priming False -trainEnd 239244 -testStart 239244 -testEnd 239594 -outputCurrentState True"""))
# os.system((baseline_script + """ -Options.inductionType normal -inputPaths sample-data/bg/bg_hmm_tokenised_morphLemma_duplicates_shuffled_seed42_plusTest.txt -inputFormat raw -execDir outputs/bg/20260622/ShuffledNoStagesNoGradualUnlocking.out -overwrite True -trainEnd 239244 -testStart 239244 -testEnd 239594 -outputCurrentState True """)) #

# os.system((baseline_script + """ -Options.inductionType morph -inputPaths sample-data/bg/bg_hmm_tokenised_morphLemma_duplicates_stage_order_plusTest.txt -inputFormat raw -execDir outputs/bg/20260622/OrderedNoStagesWithGradualUnlocking.out -overwrite True -Options.priming False -trainEnd 239244 -testStart 239244 -testEnd 239594 -outputCurrentState True"""))
# os.system((baseline_script + """ -Options.inductionType normal -inputPaths sample-data/bg/bg_hmm_tokenised_morphLemma_duplicates_stage_order_plusTest.txt -inputFormat raw -execDir outputs/bg/20260622/OrderedNoStagesNoGradualUnlocking.out -overwrite True -trainEnd 239244 -testStart 239244 -testEnd 239594 -outputCurrentState True"""))


# # SINGLE ANCHOR 1 experiment
anchor1 = True
dilute_val = 2
nounsNum = 10
verbsNum = 10

stages = ["BGPLU", "BGDEF", "BGPRES", "BGINF", "BGPAST", "BGFUT", "BGPREP"]
order5555 =",".join(stages)
grouping = 359

all_lists = []
for BGPLUAnchorIndex in range(0,3):
    for BGDEFAnchorIndex in range(0,7):
        for BGPRESAnchorIndex in range(0,17):
            for BGINFAnchorIndex in range(0,4):
                for BGPASTAnchorIndex in range(0,5):
                    for BGPREPAnchorIndex in range(0,2):
                        l = [BGPLUAnchorIndex, BGDEFAnchorIndex, BGPRESAnchorIndex, BGINFAnchorIndex, BGPASTAnchorIndex, BGPREPAnchorIndex, ]      
                        all_lists.append(l)


selected_lists = random.sample(all_lists, 200)

#SINGLE MORPHEME PRIMING BASELINES
base_anchor1_script = f"""scala -cp induction.jar induction.Induction -create -modelType hmm -Options.K 8 -Options.stage2.numIters 1 -Options.stage2.online True -Options.onlinePerm False  -Options.stage2.miniBatches True -Options.stage2.miniBatchSize 1 -inputFormat raw -Options.useBG True -Options.outputExampleFreq 5000 -log.msPerLine 5000 -log.maxIndLevel 30 -overwrite True -trainEnd 239244 -testStart 239244 -testEnd 239594 -outputCurrentState True -Options.inductionType morph -Options.diluteValue {dilute_val} -Options.orderStr {order5555} -Options.grouping {grouping} -Options.anchor1 {anchor1} -Options.nounsNum {nounsNum} -Options.verbsNum {verbsNum} """
base_anchor1_script2 = """ -inputPaths sample-data/bg/bg_hmm_tokenised_morphLemma_duplicates_shuffled_seed42_plusTest.txt  -Options.gradualUnlocking False  """

execdir_script4 = f""" -execDir outputs/bg/20260622/line4/ShuffledStagesNoUnlocking"""
execdir_script5 = f""" -execDir outputs/bg/20260622/line5/ShuffledStagesWithUnlocking"""
execdir_script7 = f""" -execDir outputs/bg/20260622/line7/OrderedStagesNoUnlocking"""
execdir_script8 = f""" -execDir outputs/bg/20260622/line8/OrderedStagesWithUnlocking"""

base_anchor1_script24 = """ -inputPaths sample-data/bg/bg_hmm_tokenised_morphLemma_duplicates_shuffled_seed42_plusTest.txt  -Options.gradualUnlocking False  """
base_anchor1_script25 = """ -inputPaths sample-data/bg/bg_hmm_tokenised_morphLemma_duplicates_shuffled_seed42_plusTest.txt  -Options.gradualUnlocking True  """
base_anchor1_script27 = """ -inputPaths sample-data/bg/bg_hmm_tokenised_morphLemma_duplicates_stage_order_plusTest.txt  -Options.gradualUnlocking False  """
base_anchor1_script28 = """ -inputPaths sample-data/bg/bg_hmm_tokenised_morphLemma_duplicates_stage_order_plusTest.txt  -Options.gradualUnlocking True  """


# execdir_scripts = [execdir_script4, execdir_script5, execdir_script7, execdir_script8]
# base_scripts = [base_anchor1_script24, base_anchor1_script25, base_anchor1_script27, base_anchor1_script28]

lines = [(execdir_script4, base_anchor1_script24), (execdir_script5,base_anchor1_script25), (execdir_script7, base_anchor1_script27),  (execdir_script8, base_anchor1_script28)]
sl_count = 0

# for l in lines:
#     exec = l[0]
#     base2 = l[1]
#     for sl in selected_lists:
#         # print(f"{sl_count}/500")
#         BGPLUAnchorIndex = sl[0]
#         BGDEFAnchorIndex = sl[1]
#         BGPRESAnchorIndex = sl[2]
#         BGINFAnchorIndex = sl[3]
#         BGPASTAnchorIndex = sl[4]
#         BGPREPAnchorIndex = sl[5]

#         os.system((base_anchor1_script + base2 + f""" -Options.BGPLUAnchorIndex {BGPLUAnchorIndex} -Options.BGDEFAnchorIndex {BGDEFAnchorIndex} -Options.BGPRESAnchorIndex {BGPRESAnchorIndex} -Options.BGINFAnchorIndex {BGINFAnchorIndex} -Options.BGPASTAnchorIndex {BGPASTAnchorIndex} -Options.BGPREPAnchorIndex {BGPREPAnchorIndex} """ + exec + f"""_{BGPLUAnchorIndex}_{BGDEFAnchorIndex}_{BGPRESAnchorIndex}_{BGINFAnchorIndex}_{BGPASTAnchorIndex}_{BGPREPAnchorIndex}""" + """.out"""))



### Compliant vs non-compliant experiments

selected_lists5 = random.sample(selected_lists, 5)

#Within stage randomisation MLU staging
stages_MLU = {2: list(permut(["BGPLU", "BGDEF", "BGPRES",])),3: list(permut(["BGINF", "BGPAST", "BGFUT", "BGPREP"]))}

orders_MLU = []
for o2 in stages_MLU[2]:
    for o3 in stages_MLU[3]:
                orders_MLU.append(",".join(o2+o3))

random_subset_orders_MLU = random.sample(orders_MLU, 100)

shuffled_orders = []
for i in range(5000):
    l = stages.copy()
    random.shuffle(l)
    if l not in orders_MLU:
        shuffled_orders.append(",".join(l))

shuffled_orders200 = shuffled_orders[:100]

base_anchor1_script_ncvsc = f"""scala -cp induction.jar induction.Induction -create -modelType hmm -Options.K 8 -Options.stage2.numIters 1 -Options.stage2.online True -Options.onlinePerm False  -Options.stage2.miniBatches True -Options.stage2.miniBatchSize 1 -inputFormat raw -Options.useBG True -Options.outputExampleFreq 5000 -log.msPerLine 5000 -log.maxIndLevel 30 -overwrite True -trainEnd 239244 -testStart 239244 -testEnd 239594 -outputCurrentState True -Options.inductionType morph -Options.diluteValue {dilute_val} -Options.grouping {grouping} -Options.anchor1 {anchor1} -Options.nounsNum {nounsNum} -Options.verbsNum {verbsNum} """

execdir_script_noncompliant = f""" -execDir outputs/bg/20260624/noncompliant/noncompliant_OrderedStagesNoUnlocking"""
execdir_script_compliant= f""" -execDir outputs/bg/20260624/compliant/compliant_OrderedStagesNoUnlocking"""


#UNCOMMENT TO DO RANDOM 2000 or MLU 2000
for sl in selected_lists5:
    BGPLUAnchorIndex = sl[0]
    BGDEFAnchorIndex = sl[1]
    BGPRESAnchorIndex = sl[2]
    BGINFAnchorIndex = sl[3]
    BGPASTAnchorIndex = sl[4]
    BGPREPAnchorIndex = sl[5]
    morph_indices = f""" -Options.BGPLUAnchorIndex {BGPLUAnchorIndex} -Options.BGDEFAnchorIndex {BGDEFAnchorIndex} -Options.BGPRESAnchorIndex {BGPRESAnchorIndex} -Options.BGINFAnchorIndex {BGINFAnchorIndex} -Options.BGPASTAnchorIndex {BGPASTAnchorIndex} -Options.BGPREPAnchorIndex {BGPREPAnchorIndex} """ 
    

    for i in range(len(shuffled_orders200)):
        os.system((base_anchor1_script_ncvsc + base_anchor1_script27 +  morph_indices + f"""-Options.orderStr "{shuffled_orders200[i]}" """ + execdir_script_noncompliant + f"""_{BGPLUAnchorIndex}_{BGDEFAnchorIndex}_{BGPRESAnchorIndex}_{BGINFAnchorIndex}_{BGPASTAnchorIndex}_{BGPREPAnchorIndex}_noncompliantStages_order{i}""" + """.out"""))
    
    for i in range(len(random_subset_orders_MLU)):
        os.system((base_anchor1_script_ncvsc + base_anchor1_script27 +  morph_indices + f"""-Options.orderStr "{random_subset_orders_MLU[i]}" """ + execdir_script_compliant + f"""_{BGPLUAnchorIndex}_{BGDEFAnchorIndex}_{BGPRESAnchorIndex}_{BGINFAnchorIndex}_{BGPASTAnchorIndex}_{BGPREPAnchorIndex}_compliantStages_order{i}""" + """.out"""))
        




# # base = f"""scala -cp induction.jar induction.Induction -create -modelType hmm -Options.K 10 -Options.stage2.numIters 1 -Options.stage2.online True -Options.onlinePerm False -Options.stage2.miniBatches True -Options.stage2.miniBatchSize 1 -Options.outputExampleFreq 1000 -log.msPerLine 100 -log.maxIndLevel 100  -inputPaths sample-data/sents_Brown_order.raw -inputFormat raw -Options.inductionType morph -Options.diluteValue 2 """
# # #
# # print(random_subset_orders_MLU[0])
# # print(shuffled_orders[0])

# #
# # for i in range(len(random_subset_orders_MLU)):
# #     iter = f"""-Options.orderStr "{random_subset_orders_MLU[i]}" -execDir outputs/21032024/MLUStages_order{i}.out -overwrite True -log.stdout false"""
# #     os.system(base+iter)
# #
# # for i in range(1000):
# #     iter = f"""-Options.orderStr "{shuffled_orders[i]}" -execDir outputs/21032024/randomStages_order{i}.out -overwrite True -log.stdout false"""
# #     os.system(base+iter)
