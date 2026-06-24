import os
import random
from itertools import permutations as permut

random.seed(10)

dilute_val = 0
order = 0
norm = False
anchor1 = True
nounsNum = 0
verbsNum = 0
BGPLUAnchorIndex = 0
BGDEFAnchorIndex = 0
BGPRESAnchorIndex = 0
BGINFAnchorIndex = 0
BGPASTAnchorIndex = 0
BGPREPAnchorIndex = 0


baseline_script = """scala -cp induction.jar induction.Induction -create -modelType hmm -Options.K 8 -Options.stage2.numIters 1 -Options.stage2.online True -Options.onlinePerm False -Options.stage2.miniBatches True -Options.stage2.miniBatchSize 1 -inputFormat raw -Options.inductionType normal -Options.outputExampleFreq 1 -log.msPerLine 10 -log.maxIndLevel 100 """

# os.system((baseline_script + """-inputPaths sample-data/sents_normtok_bg_cds_copyrightsafe.raw -execDir outputs/bg/11062025_copyrightok.out -overwrite True"""))

#sample-data/sents_morphtok_bg_cds_copyrightsafe.raw

order0 = "ART,RPAST"

os.system(baseline_script + f""" -inputPaths sample-data/sents_morphtok_bg_cds_copyrightsafe.raw -inputFormat raw -Options.inductionType bg -Options.onlyNounVerb False -execDir outputs/bg/17062025_copyrightok_morphrok_moprhprim.out -overwrite True -Options.orderStr {order0}""")

# os.system(f"""scala -cp induction.jar induction.Induction -create -modelType hmm -Options.K 10 -Options.stage2.numIters 1 -Options.stage2.online True -Options.onlinePerm False -Options.stage2.miniBatches True -Options.stage2.miniBatchSize 1 -Options.outputExampleFreq 1 -log.msPerLine 10 -log.maxIndLevel 100 -inputPaths sample-data/sents_Brown_order.raw -inputFormat raw -Options.inductionType morph -Options.onlyNounVerb True -execDir outputs/bg/01082023_BrownOrderStages_onlyNounsVerbs.out -overwrite True""")

