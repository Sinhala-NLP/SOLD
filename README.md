# SOLD - A Benchmark for Sinhala Offensive Language Identification

In this repository, we introduce the {S}inhala {O}ffensive {L}anguage {D}ataset **(SOLD)** and present multiple experiments on this dataset. **SOLD** is a manually annotated dataset containing 10,000 posts from Twitter annotated as offensive and not offensive at both sentence-level and token-level. **SOLD** is the largest offensive language dataset compiled for Sinhala. We also introduce **SemiSOLD**, a larger dataset containing more than 145,000 Sinhala tweets, annotated following a semi-supervised approach.

:warning: This repository contains texts that may be offensive and harmful.

## Annotation
We use an annotation scheme split into two levels deciding (a) Offensiveness of a tweet (sentence-level) and (b) Tokens that contribute to the offence at sentence-level (token-level).

### Sentence-level 
Our sentence-level offensive language detection follows level A in OLID [(Zampieri et al., 2019)](https://aclanthology.org/N19-1144/). We asked annotators to discriminate between the following types of tweets:
* Offensive (OFF): Posts containing any form of non-acceptable language (profanity) or a targeted offence, which can be veiled or direct. This includes insults, threats, and posts containing profane language or swear words.
* Not Offensive (NOT): Posts that do not contain offense or profanity.

Each tweet was annotated with one of the above labels, which we used as the labels in sentence-level offensive language identification.

### Token-level


## Data
Dataset is released in HuggingFace. 