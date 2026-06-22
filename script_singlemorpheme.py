import os
import random
from itertools import permutations as permut

random.seed(10)

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

# Baseline results

baseline_script = """scala -cp induction.jar induction.Induction -create -modelType hmm -Options.K 10 -Options.stage2.numIters 1 -Options.stage2.online True -Options.onlinePerm False -Options.stage2.miniBatches True -Options.stage2.miniBatchSize 1 -inputFormat raw  """

#OUTPUTS FOR TEST DATA ACCURACY
# os.system((baseline_script + """ -inputPaths sample-data/sents_Brown_shuffled_plusTest.raw -execDir outputs/23042024/shuffledNormalTest.out -overwrite True -trainEnd 295245 -testStart 295245 -testEnd 296695 -outputCurrentState True"""))
# os.system((baseline_script + """ -inputPaths sample-data/sents_Brown_order_plusTest.raw -execDir outputs/23042024/orderedNormalTest.out -overwrite True -trainEnd 295245 -testStart 295245 -testEnd 296695 -outputCurrentState True"""))
# os.system((baseline_script + """ -Options.inductionType morph -inputPaths sample-data/sents_Brown_shuffled_plusTest.raw -inputFormat raw -execDir outputs/23042024/BrownShuffledNoStagesWithGradualUnlocking.out -overwrite True -Options.priming False -trainEnd 295245 -testStart 295245 -testEnd 296695 -outputCurrentState True"""))
# os.system((baseline_script + """ -Options.inductionType morph -inputPaths sample-data/sents_Brown_shuffled_plusTest.raw -inputFormat raw -execDir outputs/23042024/NEWBrownShuffledStagesNoGradualUnlocking.out -overwrite True -Options.gradualUnlocking False -trainEnd 295245 -testStart 295245 -testEnd 296695 -outputCurrentState True"""))
# os.system((baseline_script + """ -Options.inductionType morph -inputPaths sample-data/sents_Brown_order_plusTest.raw -inputFormat raw -execDir outputs/23042024/BrownOrderedNoStagesWithGradualUnlocking.out -overwrite True -Options.priming False -trainEnd 295245 -testStart 295245 -testEnd 296695 -outputCurrentState True"""))
# os.system((baseline_script + """ -Options.inductionType morph -inputPaths sample-data/sents_Brown_order_plusTest.raw -inputFormat raw -execDir outputs/23042024/BrownOrderedStagesNoGradualUnlocking.out -overwrite True -Options.gradualUnlocking False -trainEnd 295245 -testStart 295245 -testEnd 296695 -outputCurrentState True"""))
# os.system((baseline_script + """ -Options.inductionType morph -inputPaths sample-data/sents_Brown_shuffled_plusTest.raw -execDir outputs/23042024/shuffledMorphTest.out -overwrite True -trainEnd 295245 -testStart 295245 -testEnd 296695 -outputCurrentState True"""))
# os.system((baseline_script + """ -Options.inductionType morph -inputPaths sample-data/sents_Brown_order_plusTest.raw -execDir outputs/23042024/orderedMorphTest.out -overwrite True -trainEnd 295245 -testStart 295245 -testEnd 296695 -outputCurrentState True"""))



# SINGLE ANCHOR 1 experiment
anchor1 = True
PREPAnchorIndex = 0  # in because all the first examples are with in
dilute_val = 2
order = 5555
grouping = 5
nounsNum = 10
verbsNum = 10

stages5555= ["ING","PREP", "PLU", "IRPAST", "POS", "UNCCOP", "ART", "RPAST", "R3", "IR3", "UNCAUX", "CCOP", "CAUX"]

order5555 =",".join(stages5555)

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


selected_lists = random.sample(all_lists, 500)

#SINGLE MORPHEME PRIMING BASELINES
base_anchor1_script = f"""scala -cp induction.jar induction.Induction -create -modelType hmm -Options.K 10 -Options.stage2.numIters 1 -Options.stage2.online True -Options.onlinePerm False -Options.stage2.miniBatches True -Options.stage2.miniBatchSize 1 -inputFormat raw -inputPaths sample-data/sents_Brown_shuffled_plusTest.raw -Options.outputExampleFreq 1000 -log.msPerLine 100 -log.maxIndLevel 100 -overwrite True -Options.gradualUnlocking False -trainEnd 295245 -testStart 295245 -testEnd 296695 -outputCurrentState True -Options.inductionType morph -Options.diluteValue {dilute_val} -Options.orderStr {order5555} -Options.grouping {grouping} -Options.anchor1 {anchor1} -Options.PREPAnchorIndex {PREPAnchorIndex}  """

#for random nonmcompliant 2000 exp
# execdir_script = f""" -execDir outputs/out2025/noncompliant/20250505_MorphBrownStagesOrderedNoUnlocking_dilute{dilute_val}_order{order}_grouping{grouping}_PREPAnchorIndex{PREPAnchorIndex}_n{nounsNum}_v{verbsNum}"""
execdir_script = f""" -execDir outputs/out2025/line4/20250509_MorphBrownStagesNoUnlocking_dilute{dilute_val}_order{order}_grouping{grouping}_PREPAnchorIndex{PREPAnchorIndex}_n{nounsNum}_v{verbsNum}"""

selected_lists10 = random.sample(selected_lists, 10)

#Within stage randomisation MLU staging
stages = ["ING", "PREP", "PLU", "IRPAST", "POS", "UNCCOP", "ART", "RPAST", "R3", "IR3", "UNCAUX", "CCOP", "CAUX"]

stages_MLU = {2: list(permut(["ING", "PREP", "PLU"])), 3: list(permut(["IRPAST", "POS", "UNCCOP"])), 4:  list(permut(["ART", "RPAST", "R3"])), 5:  list(permut(["IR3", "UNCAUX", "CCOP", "CAUX"]))}

orders_MLU = []
for o2 in stages_MLU[2]:
    for o3 in stages_MLU[3]:
        for o4 in stages_MLU[4]:
            for o5 in stages_MLU[5]:
                orders_MLU.append(",".join(o2+o3+o4+o5))

random_subset_orders_MLU = random.sample(orders_MLU, 200)

shuffled_orders = []
for i in range(30000):
    l = stages.copy()
    random.shuffle(l)
    if l not in orders_MLU:
        shuffled_orders.append(",".join(l))

shuffled_orders200 = shuffled_orders[:200]

sl_count = 0

for sl in selected_lists:
    # print(f"{sl_count}/500")
    UNCCOPAnchorIndex = sl[0]
    UNCAUXAnchorIndex = sl[1]
    CAUXAnchorIndex = sl[2]
    CCOPAnchorIndex = sl[3]
    ARTAnchorIndex = sl[4]
    IRPASTAnchorIndex = sl[5]
    IR3AnchorIndex = sl[6]

    os.system((base_anchor1_script + f""" -Options.UNCCOPAnchorIndex {UNCCOPAnchorIndex} -Options.UNCAUXAnchorIndex {UNCAUXAnchorIndex} -Options.CAUXAnchorIndex {CAUXAnchorIndex} -Options.CCOPAnchorIndex {CCOPAnchorIndex} -Options.ARTAnchorIndex {ARTAnchorIndex} -Options.IRPASTAnchorIndex {IRPASTAnchorIndex} -Options.IR3AnchorIndex {IR3AnchorIndex} """ + execdir_script + f"""_{UNCCOPAnchorIndex}_{UNCAUXAnchorIndex}_{CAUXAnchorIndex}_{CCOPAnchorIndex}_{ARTAnchorIndex}_{IRPASTAnchorIndex}_{IR3AnchorIndex}""" + """.out"""))

#UNCOMMENT TO DO RANDOM 2000 or MLU 2000
# for sl in selected_lists10:
#     # print(f"{sl_count}/500")
#     UNCCOPAnchorIndex = sl[0]
#     UNCAUXAnchorIndex = sl[1]
#     CAUXAnchorIndex = sl[2]
#     CCOPAnchorIndex = sl[3]
#     ARTAnchorIndex = sl[4]
#     IRPASTAnchorIndex = sl[5]
#     IR3AnchorIndex = sl[6]
#
#     for i in range(len(shuffled_orders200)):
#         os.system((base_anchor1_script + f""" -Options.UNCCOPAnchorIndex {UNCCOPAnchorIndex} -Options.UNCAUXAnchorIndex {UNCAUXAnchorIndex} -Options.CAUXAnchorIndex {CAUXAnchorIndex} -Options.CCOPAnchorIndex {CCOPAnchorIndex} -Options.ARTAnchorIndex {ARTAnchorIndex} -Options.IRPASTAnchorIndex {IRPASTAnchorIndex} -Options.IR3AnchorIndex {IR3AnchorIndex} -Options.orderStr "{shuffled_orders200[i]}" """ + execdir_script + f"""_{UNCCOPAnchorIndex}_{UNCAUXAnchorIndex}_{CAUXAnchorIndex}_{CCOPAnchorIndex}_{ARTAnchorIndex}_{IRPASTAnchorIndex}_{IR3AnchorIndex}_nonMLUStages_order{i}""" + """.out"""))

    # sl_count+=1



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