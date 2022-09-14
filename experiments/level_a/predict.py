import argparse
import os

import pandas as pd

from deepoffense.common.deepoffense_config import TEMP_DIRECTORY, SUBMISSION_FOLDER, \
    RESULT_FILE
from deepoffense.util.print_stat import print_information

if not os.path.exists(TEMP_DIRECTORY): os.makedirs(TEMP_DIRECTORY)
if not os.path.exists(os.path.join(TEMP_DIRECTORY, SUBMISSION_FOLDER)): os.makedirs(
    os.path.join(TEMP_DIRECTORY, SUBMISSION_FOLDER))

parser = argparse.ArgumentParser(
    description='''evaluates multiple models  ''')

hard_label = 'NOT'
parser.add_argument('--test', required=False, help='test file', default='data/SOLD_test.tsv')
arguments = parser.parse_args()

tst_data = pd.read_csv(arguments.test, sep="\t")
tst_data = tst_data.rename(columns={'content': 'text', 'Class': 'labels'})
test = tst_data[['text', 'labels']]

test['predictions'] = hard_label

print_information(test, "predictions", "labels")
test.to_csv(os.path.join(TEMP_DIRECTORY, RESULT_FILE), header=True, sep='\t', index=False, encoding='utf-8')
