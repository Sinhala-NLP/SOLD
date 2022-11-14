# SOLD - A Benchmark for Sinhala Offensive Language Identification

In this repository, we introduce the {S}inhala {O}ffensive {L}anguage {D}ataset **(SOLD)** and present multiple experiments on this dataset. **SOLD** is a manually annotated dataset containing 10,000 posts from Twitter annotated as offensive and not offensive at both sentence-level and token-level. **SOLD** is the largest offensive language dataset compiled for Sinhala. We also introduce **SemiSOLD**, a larger dataset containing more than 145,000 Sinhala tweets, annotated following a semi-supervised approach.

:warning: This repository contains texts that may be offensive and harmful.

## Annotation
We use an annotation scheme split into two levels deciding (a) Offensiveness of a tweet (sentence-level) and (b) Tokens that contribute to the offence at sentence-level (token-level).

### Sentence-level 
Our sentence-level offensive language detection follows level A in OLID [(Zampieri et al., 2019)](https://aclanthology.org/N19-1144/). We asked annotators to discriminate between the following types of tweets:
* **Offensive (OFF)**: Posts containing any form of non-acceptable language (profanity) or a targeted offence, which can be veiled or direct. This includes insults, threats, and posts containing profane language or swear words.
* **Not Offensive (NOT)**: Posts that do not contain offense or profanity.

Each tweet was annotated with one of the above labels, which we used as the labels in sentence-level offensive language identification.

### Token-level
To provide a human explanation of labelling, we collect rationales for the offensive language. Following HateXplain [(Mathew et al., 2021)](https://ojs.aaai.org/index.php/AAAI/article/view/17745), we define a rationale as a specific text segment that justifies the human annotator’s decision of the sentence-level labels. Therefore, We ask the annotators to highlight particular tokens in a tweet that supports their judgement about the sentence-level label (offensive, not offensive). Specifically, if a tweet is offensive, we guide the annotators to highlight tokens from the text that supports the judgement while including non-verbal expressions such as emojis and morphemes that are used to convey the intention as well. We use this as token-level offensive labels in SOLD.


![Alt text](images/SOLD_Annotation.png?raw=true "Annotation Process")

## Data
Dataset is released in HuggingFace. It can be loaded in to pandas dataframes using the following code. 

```python
from datasets import Dataset
from datasets import load_dataset

sold_train = Dataset.to_pandas(load_dataset('sinhala-nlp/SOLD', split='train'))
```

SemiSOLD == can be == loaded to a pandas dataframe using the following code. 

```python
from datasets import Dataset
from datasets import load_dataset

semi_sold = Dataset.to_pandas(load_dataset('sinhala-nlp/SemiSOLD', split='train'))
```


## Acknowledgments
We want to acknowledge Janitha Hapuarachchi, Sachith Suraweera, Chandika Udaya Kumara and Ridmi Randima, the team of volunteer annotators that provided their free time and eﬀorts to help us produce SOLD.