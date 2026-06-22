This file contains annotation guidelines for the Functional Category Induction (FCI) dataset. For more information please refer to the paper. 

There are 10 labels used for annotation:
 - N is used to label all nouns and pronouns
 - V is used to label all lexical verbs, including irregular past tense verbs
 - ASP is used to label the progressive aspect morpheme ‘ing’
 - P is used to label prepositions
 - DIV is used to label the regular plural noun morpheme ‘s’
 - DET is used to label determiners
 - T is used to label tense, more specifically regular past ‘ed’, 3rd person regular ‘s’, 3rd person present irregular verbs
 - AUX is used to label auxiliary verbs, uncontractible and contractible
 - C is used to label copular verbs, uncontractible and contractible
 - X is used to label any token that cannot be labelled with one of the above 9 labels;

In the process of labelling, distributional analysis is used to determine the correct label for a token which might be associated with multiple labels. 

Some cases where the label is difficult to determine are listed below with justification for our choice of label and some examples: 
 - weekdays, “here”, “there”: N if nominal or X if adverbial
 - numbers: N if nominal — (e.g. i need a new one N) or X otherwise
 - determiners — N if pronominal or DET if regular determiner
 - possessive pronouns —  N if used as determiner
 - auxiliary verbs: in Brown’s order only “be” verbs are considered, but we label all auxiliary verbs as AUX; this includes in addition to “be”, modal verbs, auxiliary uses of “do” (e.g. did AUX you help mummy with the shoes), auxiliary uses of “have” (e.g. i have AUX never seen a striped watch)
 - negated AUX and C: in acquisitions literature, the negated contractible auxiliary and contractible copular are normally considered as part of the auxiliary or copular, but as only the positive examples are counted by Brown, we label the negating particle as X
- idiomatic expressions: X
- wh- words: X when they introduce a clause or are adverbial (e.g. where X is the little man); N when they can be substituted for a noun (e.g. what N is your name); DET (e.g. what DET animals did you see at the zoo); “where”, “when”, “why” are adverbial so X,  “that”, “who”, “what” are sometimes nominal so they can be labelled as N is appropriate according to the distributional analysis
 - copular or auxiliary in past tense — C or AUX (and not T or V)
 - let’s: ’s is labelled as N because it is “us” contracted
 - future tense, will or `ll: T, even though future tense is not included in Brown’s order
 - existential there: X
 - irregular 3rd person verb, “has”, “does”, “goes”: T if they are used as lexical verbs, and AUX if they are used as auxiliary verbs 
 - “did”, “do”, etc.:AUX when used as auxiliary verbs in a question
 - “go” in future tense “going to”, “use” in past tense “used to”: V as they are not true auxiliaries

