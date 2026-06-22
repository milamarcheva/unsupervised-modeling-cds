This file contains annotation guidelines for the Functional Category Induction (FCI) dataset. For more information please refer to the paper. 

There are 8 labels used for annotation:
 - N is used to label all nouns and pronouns
 - V is used to label all verbs
 - DIV is used to label the nominal plural number morphemes ‘и’, 
 - DET is used to label nominal definitess morphemes, "та", "то", "те", "ят", "я", "а", "ът"
 - MOOD is used to label inferential mood morphemes, "ла", "ил", "ила", "ли"
 - T is used to label tense, more specifically past "ше", "х"; and present verbal number and person: "ш", "е", "и"; and future "ще"
 - P is used to label prepositions, "на", "за"
 - X is used to label any token that cannot be labelled with one of the above 9 labels;

In the process of labelling, distributional analysis is used to determine the correct label for a token which might be associated with multiple labels. 

Some cases where the label is difficult to determine are listed below with justification for our choice of label and some examples: 
 - numbers: N if nominal — (e.g. i need a new one N) or X otherwise (e.g. if used as prenominal)
 - adjectives: X, sometimes they appear in a position with omitted noun, but we still label them as adjectives
 - adverbs: X, not yet a recognised category in literature at this stage
 - be verb (съм): labelled as V
 - да (particle akin to verbal to): labelled as X
 - negative particle не: X
 - auxiliaries: V as bulgarian literature on acquisition does not differentiate between auxiliary and other verb use
 - preterminals (such as adjectives and numerals): X, as DET is used for the definiteness morphological marker
 - к- words (equivalent to wh-words): N when they can be substituted with a noun, otherwise X


Bulgarian HMM morpheme inventory used in bg_morphtok.py

nominal_number
NOUN plural/count suffixes: "ове", "та", "и", "е", "а", "я"
ADJ plural suffix: "и"
DET/PRON plural suffix: "и"
PROPN/NUM/non-active participial nominal-agreement suffix: "и"
Irregular plural lexical bases such as деца, очи, ръце, хора are not tokenised for nominal_number.

nominal_definiteness
Article suffixes: "ите", "ото", "ата", "ият", "ия", "ът", "ят", "та", "то", "те", "а", "я"
Reduced-definite fallback suffixes: "та", "то", "те"

present_verbal_person_number
Tokenised present endings: "м", "ш", "е", "и", "ем", "им", "ме", "те", "ете", "ите", "ат", "ят", "т"
Non-tokenised present 3sg endings such as а/я are not included here.

inferential_mood
Inferential suffixes: "л", "ла", "ло", "ли", "ал", "ала", "ало", "али", "ел", "ела", "ело", "ели", "ил", "ила", "ило", "или", "ял", "яла", "яло", "яли"

past_tense
Past-tense suffixes: "х", "ха", "хме", "хте", "ше", "и"
The suffix "и" only counts when it is actually tokenised as past tense.

future_particle
Particle: "ще"

prepositions
Prepositions: "на", "за"

