import json
import configparser
from datasets.InputExample import SeqLabelingInputExample
import itertools

def load_data(fname):
    examples = []
    with open(fname) as f:
        for line in f:
            data = json.loads(line)
            examples.append(SeqLabelingInputExample(guid=data['did'], text=data['seq'], label=data['labels']))
    return examples