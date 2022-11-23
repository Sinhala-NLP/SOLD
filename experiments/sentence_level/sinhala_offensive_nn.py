import argparse
import pandas as pd
import statistics
from datasets import Dataset
from datasets import load_dataset
from sklearn.model_selection import train_test_split

from experiments.sentence_level.offensivenn_config import args
from offensive_nn.offensive_nn_model import OffensiveNNModel
from offensive_nn.util.label_converter import encode, decode
from offensive_nn.util.print_stat import print_information

parser = argparse.ArgumentParser(
    description='''evaluates multiple models  ''')
parser.add_argument('--model_name', required=False, help='model name', default=None)
parser.add_argument('--model_type', required=False, help='model type', default="cnn2D")  # lstm or cnn2D
parser.add_argument('--augment', required=False, help='augment', default="false")
parser.add_argument('--std', required=False, help='standard deviation', default="0.01")
parser.add_argument('--augment_type', required=False, help='type of the data augmentation', default="off")

arguments = parser.parse_args()

sold_train = Dataset.to_pandas(load_dataset('sinhala-nlp/SOLD', split='train'))
sold_test = Dataset.to_pandas(load_dataset('sinhala-nlp/SOLD', split='test'))

trn_data = sold_train.rename(columns={'label': 'labels'})
tst_data = sold_test.rename(columns={'label': 'labels'})

# load training data
train = trn_data[['text', 'labels']]
test = tst_data[['text', 'labels']]

train['labels'] = encode(train["labels"])
test['labels'] = encode(test["labels"])

test_sentences = test['text'].tolist()

train_df, eval_df = train_test_split(train, test_size=0.1, random_state=args["manual_seed"])
if arguments.augment == "true":
    print("Downloading SemiSOLD")
    semi_sold = Dataset.to_pandas(load_dataset('sinhala-nlp/SemiSOLD', split='train'))
    std = float(arguments.std)
    augment_type = arguments.augment_type
    complete_df = []
    for index, row in semi_sold.iterrows():
        model_scores = [row['xlmr'], row['xlmt'], row['sinbert']]
        model_std = statistics.stdev(model_scores)
        if model_std < std:
            model_average = statistics.mean(model_scores)
            label = "OFF" if model_average > 0.5 else "NOT"
            complete_df.append([row['text'], label])

    df = pd.DataFrame(complete_df, columns=["text", "labels"])
    df['labels'] = encode(df["labels"])
    if augment_type == "off":
        filtered_df = df.loc[df['labels'] == 1]
    else:
        filtered_df = df

    print("Augmenting {} records".format(filtered_df.shape[0]))
    train_df = train_df.append(filtered_df)
    train_df = train_df.sample(frac=1, random_state=args["manual_seed"]).reset_index(drop=True)

model = OffensiveNNModel(model_type_or_path=arguments.algorithm, embedding_model_name_or_path=arguments.model_name,
                         train_df=train_df,
                         args=args, eval_df=eval_df)
model.train_model()
print("Finished Training")
model = OffensiveNNModel(model_type_or_path=args["best_model_dir"])
predictions, raw_outputs = model.predict(test_sentences)

test['predictions'] = predictions
test['predictions'] = decode(test['predictions'])
test['labels'] = decode(test['labels'])

print_information(test, "predictions", "labels")
