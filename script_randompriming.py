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

#SINGLE MORPHEME PRIMING BASELINES
base_anchor1_script = f"""scala -cp induction.jar induction.Induction -create -modelType hmm -Options.K 10 -Options.stage2.numIters 1 -Options.stage2.online True -Options.onlinePerm False -Options.stage2.miniBatches True -Options.stage2.miniBatchSize 1 -inputFormat raw -inputPaths sample-data/sents_Brown_shuffled_plusTest.raw -Options.outputExampleFreq 1000 -log.msPerLine 100 -log.maxIndLevel 100 -overwrite True -Options.gradualUnlocking False -trainEnd 295245 -testStart 295245 -testEnd 296695 -outputCurrentState True -Options.inductionType morph -Options.diluteValue {dilute_val} -Options.orderStr {order5555} -Options.grouping {grouping} -Options.anchor1 {anchor1} -Options.PREPAnchorIndex {PREPAnchorIndex}  """
execdir_script = f""" -execDir outputs/out2025/randompriming/20251103_MorphBrownStagesNoUnlocking_dilute{dilute_val}_order{order}_grouping{grouping}_PREPAnchorIndex{PREPAnchorIndex}_n{nounsNum}_v{verbsNum}"""

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
