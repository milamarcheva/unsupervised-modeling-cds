import os
import random
from itertools import permutations as permut

random.seed(10)

# print("Hello, world")

# os.system("scala -cp induction.jar induction.Induction -create -modelType hmm -Options.K 30 -Options.stage2.numIters 1 -Options.stage2.online True -Options.onlinePerm False -Options.stage2.miniBatches True -Options.stage2.miniBatchSize 1 -Options.outputExampleFreq 1.0 -log.msPerLine 10 -inputPaths sample-data/sents_Brown_order.raw -inputFormat raw -execDir tag14072023_normalK30.out -Options.inductionType normal")

dilute_val = 0
order = 0
norm = False
anchor1 = True
UNCCOPAnchorIndex = 0
UNCAUXAnchorIndex = 0
CAUXAnchorIndex = 0
CCOPAnchorIndex = 0
PREPAnchorIndex = 0
ARTAnchorIndex = 0
IRPASTAnchorIndex = 0
IR3AnchorIndex = 0
nounsNum = 0
verbsNum = 0

base_script = f'scala -cp induction.jar induction.Induction -create -modelType hmm -Options.K 10 -Options.stage2.numIters 1 -Options.stage2.online True -Options.onlinePerm False -Options.stage2.miniBatches True -Options.stage2.miniBatchSize 1 -Options.outputExampleFreq 1 -log.msPerLine 10 -log.maxIndLevel 100 -inputPaths sample-data/sents_Brown_order.raw -inputFormat raw -Options.inductionType morph  -Options.diluteValue {dilute_val}  -Options.order {order} -Options.normalize {norm} -execDir tag19072023_MorphBrownStages_dilute{dilute_val}_order{order}_normalize{norm}.out '

# for order in range(0,1): # 101:
#     for dilute_val in range(1,3):
#         for norm in [False]: # for norm in [True, False]:


# for dilute_val in range(1,10):
#     os.system(f"""scala -cp induction.jar induction.Induction -create -modelType hmm -Options.K 10 -Options.stage2.numIters 1 -Options.stage2.online True -Options.onlinePerm False -Options.stage2.miniBatches True -Options.stage2.miniBatchSize 1 -Options.outputExampleFreq 1 -log.msPerLine 10 -log.maxIndLevel 100 -inputPaths sample-data/sents_Brown_order.raw -inputFormat raw -Options.inductionType morph -Options.diluteValue {dilute_val} -execDir outputs/25072023/tag25072023_MorphBrownStages_dilute{dilute_val}_order0.out -overwrite True""")

# Baseline results

baseline_script = """scala -cp induction.jar induction.Induction -create -modelType hmm -Options.K 10 -Options.stage2.numIters 1 -Options.stage2.online True -Options.onlinePerm False -Options.stage2.miniBatches True -Options.stage2.miniBatchSize 1 -inputFormat raw -Options.inductionType normal -Options.outputExampleFreq 1 -log.msPerLine 10 -log.maxIndLevel 100 """
#
# os.system((baseline_script + """-inputPaths sample-data/all_sents_sorted_length.raw -execDir outputs/25072023/baseline/tag25072023_sortedLen.out -overwrite True"""))
# os.system((baseline_script + """-inputPaths sample-data/all_sents_reverseSorted_length.raw -execDir outputs/25072023/baseline/tag25072023_reverseSortedLen.out -overwrite True"""))
# os.system((baseline_script + """-inputPaths sample-data/all_sents_shuffled.raw -execDir outputs/25072023/baseline/tag25072023_shuffled.out -overwrite True"""))
# os.system((baseline_script + """-inputPaths sample-data/all_sents_sorted_logprobs.raw -execDir outputs/25072023/baseline/tag25072023_sortedLogprobs.out -overwrite True -Options.numOutputParams 100"""))
# os.system((baseline_script + """-inputPaths sample-data/all_sents_reverseSorted_logprobs.raw -execDir outputs/25072023/baseline/tag25072023_reverseSortedLogprobs.out -overwrite True"""))
# os.system((baseline_script + """-inputPaths sample-data/sents_Brown_order.raw -inputFormat raw -execDir outputs/25072023/baseline/tag25072023_BrownOrderNoStages.out -overwrite True"""))
# os.system((baseline_script + """-inputPaths sample-data/sents_Brown_shuffled.raw -inputFormat raw -execDir outputs/25072023/baseline/tag25072023_BrownShuffledNoStages.out -overwrite True"""))
# os.system((baseline_script + """-inputPaths sample-data/all_sents_sorted_logprobs_norm.raw -execDir outputs/25072023/baseline/tag25072023_sortedLogprobsNorm.out -overwrite True"""))
# os.system((baseline_script + """-inputPaths sample-data/all_sents_reverseSorted_logprobs_norm.raw -execDir outputs/25072023/baseline/tag25072023_reverseSortedLogprobsNorm.out -overwrite True """))
# os.system((baseline_script + """  -inputPaths sample-data/sents_ReverseBrown_order.raw -execDir outputs/25072023/baseline/BrownReverseNoStages.out -overwrite True """))

# os.system((baseline_script + """-inputPaths sample-data/all_sents_shuffled.raw -execDir outputs/26032024/shuffled.out -overwrite True -testInputPaths sample-data/test_sents.raw"""))

# FREQUENCY
# os.system((baseline_script + """-inputPaths sample-data/all_sents_sorted_freq.raw -execDir outputs/25072023/baseline/tag25072023_sortedFreq.out -overwrite True"""))
# os.system((baseline_script + """-inputPaths sample-data/all_sents_reverseSorted_freq.raw -execDir outputs/25072023/baseline/tag25072023_reverseSortedFreq.out -overwrite True"""))

#OUTPUTS FOR TEST DATA ACCURACY
# os.system((baseline_script + """ -inputPaths sample-data/sents_Brown_shuffled_plusTest.raw -execDir outputs/23042024/shuffledNormalTest.out -overwrite True -trainEnd 295245 -testStart 295245 -testEnd 296695 -outputCurrentState True"""))
# os.system((baseline_script + """ -inputPaths sample-data/sents_Brown_order_plusTest.raw -execDir outputs/23042024/orderedNormalTest.out -overwrite True -trainEnd 295245 -testStart 295245 -testEnd 296695 -outputCurrentState True"""))
# os.system((baseline_script + """ -Options.inductionType morph -inputPaths sample-data/sents_Brown_shuffled_plusTest.raw -inputFormat raw -execDir outputs/23042024/BrownShuffledNoStagesWithGradualUnlocking.out -overwrite True -Options.priming False -trainEnd 295245 -testStart 295245 -testEnd 296695 -outputCurrentState True"""))
# os.system((baseline_script + """ -Options.inductionType morph -inputPaths sample-data/sents_Brown_shuffled_plusTest.raw -inputFormat raw -execDir outputs/23042024/BrownShuffledStagesNoGradualUnlocking.out -overwrite True -Options.gradualUnlocking False -trainEnd 295245 -testStart 295245 -testEnd 296695 -outputCurrentState True"""))
# os.system((baseline_script + """ -Options.inductionType morph -inputPaths sample-data/sents_Brown_order_plusTest.raw -inputFormat raw -execDir outputs/23042024/BrownOrderedNoStagesWithGradualUnlocking.out -overwrite True -Options.priming False -trainEnd 295245 -testStart 295245 -testEnd 296695 -outputCurrentState True"""))
# os.system((baseline_script + """ -Options.inductionType morph -inputPaths sample-data/sents_Brown_order_plusTest.raw -inputFormat raw -execDir outputs/23042024/BrownOrderedStagesNoGradualUnlocking.out -overwrite True -Options.gradualUnlocking False -trainEnd 295245 -testStart 295245 -testEnd 296695 -outputCurrentState True"""))
# os.system((baseline_script + """ -Options.inductionType morph -inputPaths sample-data/sents_Brown_shuffled_plusTest.raw -execDir outputs/23042024/shuffledMorphTest.out -overwrite True -trainEnd 295245 -testStart 295245 -testEnd 296695 -outputCurrentState True"""))
# os.system((baseline_script + """ -Options.inductionType morph -inputPaths sample-data/sents_Brown_order_plusTest.raw -execDir outputs/23042024/orderedMorphTest.out -overwrite True -trainEnd 295245 -testStart 295245 -testEnd 296695 -outputCurrentState True"""))



#UNLOCKING
# os.system((baseline_script + """ -Options.inductionType morph -inputPaths sample-data/sents_Brown_shuffled.raw -inputFormat raw -execDir outputs/25072023/baseline/BrownShuffledNoStagesWithGradualUnlocking.out -overwrite True -Options.priming False"""))
# os.system((baseline_script + """ -Options.inductionType morph -inputPaths sample-data/sents_Brown_shuffled.raw -inputFormat raw -execDir outputs/25072023/baseline/tag25072023_BrownShuffledStagesNoGradualUnlocking.out -overwrite True -Options.gradualUnlocking False"""))

# # # Baseline staged morph
# os.system(f"""scala -cp induction.jar induction.Induction -create -modelType hmm -Options.K 10 -Options.stage2.numIters 1 -Options.stage2.online True -Options.onlinePerm False -Options.stage2.miniBatches True -Options.stage2.miniBatchSize 1 -Options.outputExampleFreq 1 -log.msPerLine 10 -log.maxIndLevel 100 -inputPaths sample-data/sents_Brown_order.raw -inputFormat raw -Options.inductionType morph -Options.diluteValue 2 -execDir outputs/25072023/baseline/tag25072023_BrownOrderStages_dilute2.out -overwrite True""")

# # Baseline staged MLU
# os.system(f"""scala -cp induction.jar induction.Induction -create -modelType hmm -Options.K 10 -Options.stage2.numIters 1 -Options.stage2.online True -Options.onlinePerm False -Options.stage2.miniBatches True -Options.stage2.miniBatchSize 1 -Options.outputExampleFreq 1 -log.msPerLine 10 -log.maxIndLevel 100 -inputPaths sample-data/sents_Brown_order.raw -inputFormat raw -Options.inductionType mlu -Options.diluteValue 2 -execDir outputs/25072023/baseline/tag25072023_BrownOrderMLUStages_dilute2.out -overwrite True""")
#
# # Baseline staged morph but shuffled examples
# os.system(f"""scala -cp induction.jar induction.Induction -create -modelType hmm -Options.K 10 -Options.stage2.numIters 1 -Options.stage2.online True -Options.onlinePerm False -Options.stage2.miniBatches True -Options.stage2.miniBatchSize 1 -Options.outputExampleFreq 1 -log.msPerLine 10 -log.maxIndLevel 100 -inputPaths sample-data/sents_Brown_shuffled.raw -inputFormat raw -Options.inductionType morph -Options.diluteValue 2 -execDir outputs/25072023/baseline/tag25072023_BrownOrderShuffledStages_dilute2.out -overwrite True""")
#
#
# #Alternative cluster groupings:
# os.system(f"""scala -cp induction.jar induction.Induction -create -modelType hmm -Options.K 11 -Options.grouping 1 -Options.stage2.numIters 1 -Options.stage2.online True -Options.onlinePerm False -Options.stage2.miniBatches True -Options.stage2.miniBatchSize 1 -Options.outputExampleFreq 1 -log.msPerLine 10 -log.maxIndLevel 100 -inputPaths sample-data/sents_Brown_order.raw -inputFormat raw -Options.inductionType morph -Options.diluteValue 2 -execDir outputs/27072023/tag27072023_BrownOrderStages_dilute2_grouping1.out -overwrite True""")
# os.system(f"""scala -cp induction.jar induction.Induction -create -modelType hmm -Options.K 10 -Options.grouping 2 -Options.stage2.numIters 1 -Options.stage2.online True -Options.onlinePerm False -Options.stage2.miniBatches True -Options.stage2.miniBatchSize 1 -Options.outputExampleFreq 1 -log.msPerLine 10 -log.maxIndLevel 100 -inputPaths sample-data/sents_Brown_order.raw -inputFormat raw -Options.inductionType morph -Options.diluteValue 2 -execDir outputs/27072023/tag27072023_BrownOrderStages_dilute2_grouping2.out -overwrite True""")
# os.system(f"""scala -cp induction.jar induction.Induction -create -modelType hmm -Options.K 10 -Options.grouping 3 -Options.stage2.numIters 1 -Options.stage2.online True -Options.onlinePerm False -Options.stage2.miniBatches True -Options.stage2.miniBatchSize 1 -Options.outputExampleFreq 1 -log.msPerLine 10 -log.maxIndLevel 100 -inputPaths sample-data/sents_Brown_order.raw -inputFormat raw -Options.inductionType morph -Options.diluteValue 2 -execDir outputs/27072023/tag27072023_BrownOrderStages_dilute2_grouping3.out -overwrite True""")
# os.system(f"""scala -cp induction.jar induction.Induction -create -modelType hmm -Options.K 9 -Options.grouping 4 -Options.stage2.numIters 1 -Options.stage2.online True -Options.onlinePerm False -Options.stage2.miniBatches True -Options.stage2.miniBatchSize 1 -Options.outputExampleFreq 1 -log.msPerLine 10 -log.maxIndLevel 100 -inputPaths sample-data/sents_Brown_order.raw -inputFormat raw -Options.inductionType morph -Options.diluteValue 2 -execDir outputs/27072023/tag27072023_BrownOrderStages_dilute2_grouping4.out -overwrite True""")

# #PREP
# os.system(f"""scala -cp induction.jar induction.Induction -create -modelType hmm -Options.K 10 -Options.grouping 5 -Options.order 5555 -Options.stage2.numIters 1 -Options.stage2.online True -Options.onlinePerm False -Options.stage2.miniBatches True -Options.stage2.miniBatchSize 1 -Options.outputExampleFreq 1 -log.msPerLine 10 -log.maxIndLevel 100 -inputPaths sample-data/sents_Brown_order.raw -inputFormat raw -Options.inductionType morph -Options.diluteValue 2 -execDir outputs/27072023/tag27072023_BrownOrderStages_dilute2_grouping5_order555.out -overwrite True """)
#
#
# #Minibatch exp:
# for mb in [5,25,50,75,100]:
#     os.system(f"""scala -cp induction.jar induction.Induction -create -modelType hmm -Options.K 10 -Options.stage2.numIters 1 -Options.stage2.online True -Options.onlinePerm False -Options.stage2.miniBatches True -Options.stage2.miniBatchSize {mb} -Options.outputExampleFreq 1 -log.msPerLine 10 -log.maxIndLevel 100 -inputPaths sample-data/sents_Brown_order.raw -inputFormat raw -Options.inductionType morph -Options.diluteValue 2 -execDir outputs/27072023/minibatches/tag27072023_BrownOrderStages_dilute2_mb{mb}.out -overwrite True """)

# #ONLINE EM 100 iters:
# baseline_script100 = """scala -cp induction.jar induction.Induction -create -modelType hmm -Options.K 10 -Options.stage2.numIters 100 -Options.stage2.online True -Options.onlinePerm False -Options.stage2.miniBatches True -Options.stage2.miniBatchSize 1 -inputFormat raw -Options.inductionType normal -Options.outputExampleFreq 1 -log.msPerLine 10 -log.maxIndLevel 100 """
# os.system((baseline_script100 + """-inputPaths sample-data/all_sents_shuffled.raw -execDir outputs/27072023/iters100/tag27072023_shuffled.out -overwrite True"""))


# Reduced number of examples:
# for sen in [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000]:
#     os.system(f"""scala -cp induction.jar induction.Induction -create -modelType hmm -Options.K 10 -Options.stage2.numIters 1 -Options.stage2.online True -Options.onlinePerm False -Options.stage2.miniBatches True -Options.stage2.miniBatchSize 1 -Options.outputExampleFreq 1 -log.msPerLine 10 -log.maxIndLevel 100 -inputPaths sample-data/sents_Brown_order.raw -inputFormat raw -Options.inductionType morph -Options.diluteValue 2 -Options.stagingExamplesNum {sen} -execDir outputs/27072023/numExamples/tag27072023_BrownOrderStages_dilute2_numEx{sen}.out -overwrite True""")
#
# # Reduced fraction of examples:
# for sef in [0.9, 0.5, 0.25, 0.1, 0.05, 0.01, 0.001, 0.0001, 0.00001]: #[0.9, 0.5, 0.25, 0.1, 0.05, 0.01, 0.001]
#     os.system(f"""scala -cp induction.jar induction.Induction -create -modelType hmm -Options.K 10 -Options.stage2.numIters 1 -Options.stage2.online True -Options.onlinePerm False -Options.stage2.miniBatches True -Options.stage2.miniBatchSize 1 -Options.outputExampleFreq 1 -log.msPerLine 10 -log.maxIndLevel 100 -inputPaths sample-data/sents_Brown_order.raw -inputFormat raw -Options.inductionType morph -Options.diluteValue 2 -Options.stagingExamplesFraction {sef} -execDir outputs/27072023/numExamples/tag27072023_BrownOrderStages_dilute2_fracEx{sef}.out -overwrite True""")


#Only nouns and verbs intialized
# os.system(f"""scala -cp induction.jar induction.Induction -create -modelType hmm -Options.K 10 -Options.stage2.numIters 1 -Options.stage2.online True -Options.onlinePerm False -Options.stage2.miniBatches True -Options.stage2.miniBatchSize 1 -Options.outputExampleFreq 1 -log.msPerLine 10 -log.maxIndLevel 100 -inputPaths sample-data/sents_Brown_order.raw -inputFormat raw -Options.inductionType morph -Options.onlyNounVerb True -execDir outputs/01082023/tag01082023_BrownOrderStages_onlyNounsVerbs.out -overwrite True""")

# const #TODO

# SINGLE ANCHOR 1 experiment
anchor1 = True
PREPAnchorIndex = 0  # in because all the first examples are with in
dilute_val = 2
order = 5555
grouping = 5
nounsNum = 10
verbsNum = 10


UNCCOPAnchorIndex = 3
UNCAUXAnchorIndex = 1
CAUXAnchorIndex = 0
CCOPAnchorIndex = 2
ARTAnchorIndex = 0
IRPASTAnchorIndex = 12
IR3AnchorIndex = 1

stages5555= ["ING","PREP", "PLU", "IRPAST", "POS", "UNCCOP", "ART", "RPAST", "R3", "IR3", "UNCAUX", "CCOP", "CAUX"]

order5555 =",".join(stages5555)

anchor1_script = f"""scala -cp induction.jar induction.Induction -create -modelType hmm -Options.K 10 -Options.stage2.numIters 1 -Options.stage2.online True -Options.onlinePerm False -Options.stage2.miniBatches True -Options.stage2.miniBatchSize 1 -Options.outputExampleFreq 1 -log.msPerLine 200 -log.maxIndLevel 100  -inputPaths sample-data/sents_Brown_order_plusTest.raw -inputFormat raw -Options.inductionType morph   -Options.diluteValue {dilute_val} -Options.order {order} -Options.grouping {grouping} -Options.anchor1 {anchor1}   -Options.PREPAnchorIndex {PREPAnchorIndex} -Options.orderStr {order5555} -overwrite True -trainEnd 295245 -testStart 295245 -testEnd 296695 -outputCurrentState True"""
execdir_script = f""" -execDir outputs/06012025/MorphBrownStages_dilute{dilute_val}_order{order}_grouping{grouping}_PREPAnchorIndex{PREPAnchorIndex}_n{nounsNum}_v{verbsNum}"""
#
# dilute2_order5555_grouping5_PREPAnchorIndex0_n10_v10_3_1_0_2_0_12_1.out/"
# os.system((anchor1_script + f""" -Options.UNCCOPAnchorIndex {UNCCOPAnchorIndex} -Options.UNCAUXAnchorIndex {UNCAUXAnchorIndex} -Options.CAUXAnchorIndex {CAUXAnchorIndex} -Options.CCOPAnchorIndex {CCOPAnchorIndex} -Options.ARTAnchorIndex {ARTAnchorIndex} -Options.IRPASTAnchorIndex {IRPASTAnchorIndex} -Options.IR3AnchorIndex {IR3AnchorIndex}""" + execdir_script + f"""_{UNCCOPAnchorIndex}_{UNCAUXAnchorIndex}_{CAUXAnchorIndex}_{CCOPAnchorIndex}_{ARTAnchorIndex}_{IRPASTAnchorIndex}_{IR3AnchorIndex}""" + """.out"""))


# dr_anchor1_n10_v10_3_1_0_2_0_12_1 = "/Users/milamarcheva/Desktop/unsupervised-modeling/outputs/02082023/anchor1/tag02082023_MorphBrownStages_dilute2_order5555_grouping5_PREPAnchorIndex0_n10_v10_3_1_0_2_0_12_1.out/"

all_lists = []
for UNCCOPAnchorIndex in range(0, 5):
    for UNCAUXAnchorIndex in range(0, 5):
        for CAUXAnchorIndex in range(0, 3):
            for CCOPAnchorIndex in range(0, 3):
                for ARTAnchorIndex in range(0, 2):
                    for IRPASTAnchorIndex in range(0, 13):
                        for IR3AnchorIndex in range(0, 3):
                            l = [UNCCOPAnchorIndex, UNCAUXAnchorIndex, CAUXAnchorIndex, CCOPAnchorIndex, ARTAnchorIndex, IRPASTAnchorIndex, IR3AnchorIndex]
                            all_lists.append(l)

# print(len(all_lists))
# print(all_lists[10000:10020])
#
selected_lists = random.sample(all_lists, 200)

sl_count = 0
for sl in selected_lists:
    print(f"{sl_count}/200")
    UNCCOPAnchorIndex = sl[0]
    UNCAUXAnchorIndex = sl[1]
    CAUXAnchorIndex = sl[2]
    CCOPAnchorIndex = sl[3]
    ARTAnchorIndex = sl[4]
    IRPASTAnchorIndex = sl[5]
    IR3AnchorIndex = sl[6]

    #TODO amend so the test happens
    os.system((anchor1_script + f""" -Options.UNCCOPAnchorIndex {UNCCOPAnchorIndex} -Options.UNCAUXAnchorIndex {UNCAUXAnchorIndex} -Options.CAUXAnchorIndex {CAUXAnchorIndex} -Options.CCOPAnchorIndex {CCOPAnchorIndex} -Options.ARTAnchorIndex {ARTAnchorIndex} -Options.IRPASTAnchorIndex {IRPASTAnchorIndex} -Options.IR3AnchorIndex {IR3AnchorIndex}""" + execdir_script + f"""_{UNCCOPAnchorIndex}_{UNCAUXAnchorIndex}_{CAUXAnchorIndex}_{CCOPAnchorIndex}_{ARTAnchorIndex}_{IRPASTAnchorIndex}_{IR3AnchorIndex}""" + """.out"""))

    sl_count+=1


# #ONLINE EM 1000 iters:
# baseline_script100 = """scala -cp induction.jar induction.Induction -create -modelType hmm -Options.K 10 -Options.stage2.numIters 1000 -Options.stage2.online True -Options.onlinePerm False -Options.stage2.miniBatches True -Options.stage2.miniBatchSize 1 -inputFormat raw -Options.inductionType normal -Options.outputExampleFreq 1 -log.msPerLine 10 -log.maxIndLevel 100 """
# os.system((baseline_script100 + """-inputPaths sample-data/all_sents_shuffled.raw -execDir outputs/27072023/iters100/tag27072023_shuffled_1000iters.out -overwrite True"""))


# random orders of staging
# for order in range(601, 1001):
#     os.system(base_script +  f' -Options.order {order} -execDir outputs/28072023/tag28072023_MorphBrownStages_dilute2_order{order}.out')

# Baseline staged morph but shuffled examples: 100 trials
# for i in [500]: in range(1, 501)
#     os.system(f"""scala -cp induction.jar induction.Induction -create -modelType hmm -Options.K 10 -Options.stage2.numIters 1 -Options.stage2.online True -Options.onlinePerm True -Options.onlinePermRandom {i} -Options.stage2.miniBatches True -Options.stage2.miniBatchSize 1 -Options.outputExampleFreq 1 -log.msPerLine 10 -log.maxIndLevel 100 -inputPaths sample-data/sents_Brown_shuffled.raw -inputFormat raw -Options.inductionType morph -Options.diluteValue 2 -execDir outputs/31072023/tag31072023_BrownOrderShuffledStages_dilute2_rseed_{i}.out -overwrite True""")

# os.system(f"""scala -cp induction.jar induction.Induction -create -modelType hmm -Options.K 10 -Options.stage2.numIters 1 -Options.stage2.online True -Options.onlinePerm False  -Options.stage2.miniBatches True -Options.stage2.miniBatchSize 1 -Options.outputExampleFreq 1 -log.msPerLine 10 -log.maxIndLevel 100 -inputPaths sample-data/sents_Brown_order.raw -inputFormat raw -Options.inductionType morph -Options.diluteValue 2 -execDir outputs/01082023/tag01082023_BrownOrderStages_dilute2.out -overwrite True""")

#Using only a number of the nouns and verbs
#
# for n in [1, 10]: # [1] range(10, 75, 10)
#     for v in [1, 10]: # [1]range(5, 53, 10)
#         os.system(f"""scala -cp induction.jar induction.Induction -create -modelType hmm -Options.K 10 -Options.stage2.numIters 1 -Options.stage2.online True -Options.onlinePerm False  -Options.stage2.miniBatches True -Options.stage2.miniBatchSize 1 -Options.outputExampleFreq 1 -log.msPerLine 10 -log.maxIndLevel 100 -inputPaths sample-data/sents_Brown_order.raw -inputFormat raw -Options.inductionType morph -Options.diluteValue 2 -Options.nounsNum {n} -Options.verbsNum {v} -execDir outputs/01082023/reducedNounsVerbs/tag01082023_BrownOrderStages_dilute2_nouns{n}_verbs{v}.out -overwrite True""")


#Within stage randomisation MLU staging
stages = ["ING", "IN", "ON", "PLU", "IRPAST", "POS", "UNCCOP", "ART", "RPAST", "R3", "IR3", "UNCAUX", "CCOP", "CAUX"]
#
stages_MLU = {2: list(permut(["ING", "IN", "ON", "PLU"])), 3: list(permut(["IRPAST", "POS", "UNCCOP"])), 4:  list(permut(["ART", "RPAST", "R3"])), 5:  list(permut(["IR3", "UNCAUX", "CCOP", "CAUX"]))}
#
# orders_MLU = []
# for o2 in stages_MLU[2]:
#     for o3 in stages_MLU[3]:
#         for o4 in stages_MLU[4]:
#             for o5 in stages_MLU[5]:
#                 orders_MLU.append(",".join(o2+o3+o4+o5))
#
#
# random_subset_orders_MLU = random.sample(orders_MLU, 1000)
#
# shuffled_orders = []
# for i in range(30000):
#     l = stages.copy()
#     random.shuffle(l)
#     if l not in orders_MLU:
#         shuffled_orders.append(",".join(l))
#
# # print('shuffled_orders', len(shuffled_orders))
#
# base = f"""scala -cp induction.jar induction.Induction -create -modelType hmm -Options.K 10 -Options.stage2.numIters 1 -Options.stage2.online True -Options.onlinePerm False -Options.stage2.miniBatches True -Options.stage2.miniBatchSize 1 -Options.outputExampleFreq 1000 -log.msPerLine 100 -log.maxIndLevel 100  -inputPaths sample-data/sents_Brown_order.raw -inputFormat raw -Options.inductionType morph -Options.diluteValue 2 """
# #
# print(random_subset_orders_MLU[0])
# print(shuffled_orders[0])

#
# for i in range(len(random_subset_orders_MLU)):
#     iter = f"""-Options.orderStr "{random_subset_orders_MLU[i]}" -execDir outputs/21032024/MLUStages_order{i}.out -overwrite True -log.stdout false"""
#     os.system(base+iter)
#
# for i in range(1000):
#     iter = f"""-Options.orderStr "{shuffled_orders[i]}" -execDir outputs/21032024/randomStages_order{i}.out -overwrite True -log.stdout false"""
#     os.system(base+iter)

base = f"""scala -cp induction.jar induction.Induction -create -modelType hmm -Options.K 10 -Options.stage2.numIters 1 -Options.stage2.online True -Options.onlinePerm False -Options.stage2.miniBatches True -Options.stage2.miniBatchSize 1 -Options.outputExampleFreq 1000 -log.msPerLine 100 -log.maxIndLevel 100  -inputPaths sample-data/sents_Brown_order.raw -inputFormat raw -Options.inductionType morph -Options.diluteValue 2 """

# Mean order of acqusition of the morphemes
meanOrderOfAcquisition = ",".join(stages)
iter = f"""-Options.orderStr "{meanOrderOfAcquisition}" -execDir outputs/26032024/acc.out -testInputPaths  sample-data/test_sents.raw  -overwrite True -outputCurrentState True -Options.testStart 295245 -Options.testEnd 295248  """ #-outputFullPred True -outputCurrentState True
# os.system(base+iter)



# Staging order by frequency
# brownFrequencyOrder = "ART,CCOP,UNCCOP,ING,PLU,IRPAST,IN,CAUX,POS,ON,UNCAUX,R3,RPAST,IR3"
# myFrequencyOrder = "ART,CCOP,IRPAST,ING,UNCCOP,PLU,CAUX,IN,UNCAUX,ON,RPAST,IR3,R3,POS"
# iter = f"""-Options.orderStr "{myFrequencyOrder}" -execDir outputs/26032024/myFrequencyOrder.out -overwrite True -log.stdout false"""
# os.system(base+iter)

# Run induction with tagged data
#
# base = f"""scala -cp induction.jar induction.Induction -create -modelType hmm -Options.K 10 -Options.stage2.numIters 1 -Options.stage2.online True -Options.onlinePerm False -Options.stage2.miniBatches True -Options.stage2.miniBatchSize 1 -Options.outputExampleFreq 1000 -log.msPerLine 10 -log.maxIndLevel 100  -inputPaths sample-data/sample-data/all_sents_shuffled.raw -inputFormat raw -Options.inductionType normal -execDir outputs/26032024/allexamplesNoStaging.out -overwrite True """
# os.system(base)
