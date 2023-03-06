from torch.utils.data import Dataset
from typing import List
import torch
import numpy as np
import logging
from tqdm import tqdm
from datasets.InputExample import SeqLabelingInputExample
from transformers import PreTrainedTokenizer
import itertools

from torch.utils.data import DataLoader


class NERDataset(Dataset):

    def __init__(self, examples: List[SeqLabelingInputExample], tokenizer: PreTrainedTokenizer, max_seq_len: int,
                 label2id=None, show_progress_bar: bool = None, split_seqs: bool = None):
        """
        Create a new SentencesDataset with the tokenized texts and the labels as Tensor
        """
        if show_progress_bar is None:
            show_progress_bar = (
                        logging.getLogger().getEffectiveLevel() == logging.INFO or logging.getLogger().getEffectiveLevel() == logging.DEBUG)
        self.show_progress_bar = show_progress_bar
        self.label2id = label2id
        self.tokenizer = tokenizer
        self.max_len = max_seq_len
        self.split_seqs = split_seqs
        self.padding_label = -1
        self.label_special_tokens = False
        self.convert_input_examples(examples, tokenizer, max_seq_len)

    def convert_input_examples(self, examples: List[SeqLabelingInputExample], tokenizer: PreTrainedTokenizer,
                               max_seq_len: int):
        """
        Converts input examples to a SmartBatchingDataset usable to train the model with
        SentenceTransformer.smart_batching_collate as the collate_fn for the DataLoader
        smart_batching_collate as collate_fn is required because it transforms the tokenized texts to the tensors.
        :param examples:
            the input examples for the training
        :param model
            the Sentence BERT model for the conversion
        :return: a SmartBatchingDataset usable to train the model with SentenceTransformer.smart_batching_collate as the collate_fn
            for the DataLoader
        """

        iterator = examples

        if self.show_progress_bar:
            iterator = tqdm(iterator, desc="Convert dataset")
        logging.info('Sequences longer than {} will be split: {}'.format(self.max_len, self.split_seqs))
        if self.label2id is None:
            # get all labels in the dataset
            all_labels = list(set(list(itertools.chain.from_iterable([example.label for example in examples]))))
            for elm in all_labels:
                if elm.startswith('B-') and 'I-' + elm.strip('B-') not in set(all_labels):
                    all_labels.append('I-' + elm.strip('B-'))

            if self.label_special_tokens is not False:
                # add special labels. all special tokens will be labeled with this label
                all_labels.extend(['CLS', 'SEP', 'X'])
                all_labels.sort()
                self.label2id = {label: idx for idx, label in enumerate(all_labels)}
                self.id2label = {val: key for key, val in self.label2id.items()}
                self.num_predictable_labels = len(self.id2label)
            else:
                all_labels.sort()
                self.label2id = {label: idx for idx, label in enumerate(all_labels)}
                self.id2label = {val: key for key, val in self.label2id.items()}

                self.label2id['X'] = self.padding_label
                self.label2id['CLS'] = self.padding_label
                self.label2id['SEP'] = self.padding_label
                self.id2label[self.padding_label] = 'X'
                self.num_predictable_labels = len(self.id2label) - 1

            logging.info('Label set: {}'.format(self.label2id))
            ignored_labels = [elm for elm, idx in self.label2id.items() if idx == self.padding_label]
            logging.info('{} labels will be ignored during training: {}'.format(len(ignored_labels), ignored_labels))

        tokenized_seqs = []
        extended_labels = []
        c = 0
        for ex_index, example in enumerate(iterator):

            subtoks = []
            # labels need to be extended. a subtoken that starts with '#' gets label 'X'
            sublabels = []
            for tid, tok in enumerate(example.seq):
                for sid, subtok in enumerate(tokenizer.tokenize(tok)):
                    subtoks.append(subtok)
                    if sid == 0:
                        sublabels.append(example.label[tid])
                    elif not subtok.startswith('##'):
                        sublabels.append(example.label[tid])
                    else:
                        sublabels.append('X')

            # split the sequence if it is longer than max length. make sure to not split within an entity or within a word
            if len(subtoks) > max_seq_len - 2:

                c += 1
                if self.split_seqs:
                    smaller_subtoks, smaller_sublabels = self.split_seq(subtoks, sublabels)
                    i = 0
                    for subtoks, sublabels in zip(smaller_subtoks, smaller_sublabels):
                        i += 1
                else:
                    smaller_subtoks, smaller_sublabels = [subtoks[:max_seq_len - 2]], [sublabels[:max_seq_len - 2]]

            else:
                smaller_subtoks, smaller_sublabels = [subtoks], [sublabels]

            for subtoks, sublabels in zip(smaller_subtoks, smaller_sublabels):
                assert len(subtoks) == len(sublabels)
                # add [CLS] and [SEP] tokens

                subtoks = [self.tokenizer.cls_token] + subtoks + [self.tokenizer.sep_token]
                sublabels = ['CLS'] + sublabels + ['SEP']

                if ex_index < 5:
                    logging.info('Ex {}'.format(ex_index))
                    logging.info('Input seq: {}'.format(subtoks))
                    logging.info('--> {}'.format(tokenizer.convert_tokens_to_ids(subtoks)))
                    logging.info('Labels: {}'.format(sublabels))
                    logging.info('--> {}'.format([self.label2id[l] for l in sublabels]))

                tokenized_seqs.append(tokenizer.convert_tokens_to_ids(subtoks))
                extended_labels.append([self.label2id[l] for l in sublabels])

        self.seqs = tokenized_seqs
        self.labels = extended_labels
        logging.info(
            'Found {} sequences longer than max_len of {}. Splitting: {}'.format(c, self.max_len, self.split_seqs))

    def split_seq(self, subtoks, sublabels):
        """
        split sequences longer than max_length. make sure to not split within word or within entity
        :param subtoks:
        :param sublabels:
        :return:
        """

        smaller_subtoks, smaller_sublabels = [], []
        abs_end_positon = self.max_len - 2
        abs_start_position = 0
        while abs_start_position < len(subtoks):

            t_chunk = subtoks[abs_start_position:abs_end_positon]
            l_chunk = sublabels[abs_start_position:abs_end_positon]
            start_label = '[]'
            if abs_end_positon < len(sublabels):
                start_label = sublabels[abs_end_positon]

            if start_label == 'X' or start_label.startswith('I-'):
                while start_label == 'X' or start_label.startswith('I-'):
                    start_label = l_chunk.pop()
                    t_chunk.pop()
            # this is to have a stopping criterion in case the entity is as long as the seq len (should never happen in a realistic setup)
            if len(t_chunk) == 0:
                break
            smaller_subtoks.append(t_chunk)
            smaller_sublabels.append(l_chunk)
            abs_start_position = sum([len(elm) for elm in smaller_subtoks])
            abs_end_positon = abs_start_position + self.max_len - 2

        return smaller_subtoks, smaller_sublabels

    def __getitem__(self, item):
        return self.seqs[item], self.labels[item]

    def __len__(self):
        return len(self.seqs)

    def collate(self, data):

        seqs = [elm[0] for elm in data]
        labels = [elm[1] for elm in data]
        max_len = np.max([len(seq) for seq in seqs])
        padded_seqs = []
        padded_attention_masks = []
        padded_labels = []

        for seq, label in zip(seqs, labels):
            valid_length = np.sum([1 for elm in seq if elm != self.tokenizer.pad_token_id])
            seq = seq[:valid_length]
            label = label[:valid_length]

            assert self.tokenizer.pad_token_id not in seq

            # pad the sequence and labels to max len, build attention mask
            attention_mask = [1] * len(seq)
            padded_seq = seq
            padded_label = label

            while len(padded_seq) < max_len:
                padded_seq.append(self.tokenizer.convert_tokens_to_ids(self.tokenizer.pad_token))
                padded_label.append(self.padding_label)
                attention_mask.append(0)
                # assert len([1 for elm in attention_mask if elm == 0]) == len([1 for elm in padded_label if elm == -1])

            # assert len([1 for elm in attention_mask if elm == 0]) == len([1 for elm in padded_label if elm == -1])

            padded_seqs.append(padded_seq)
            padded_labels.append(padded_label)
            padded_attention_masks.append(attention_mask)

            assert len(padded_seq) == len(padded_label) == len(attention_mask)
            # assert len([1 for elm in attention_mask if elm == 0]) == len([1 for elm in padded_label if elm == -1])

        return {'input_ids': torch.LongTensor(padded_seqs),
                'attention_mask': torch.LongTensor(padded_attention_masks),
                'labels': torch.LongTensor(padded_labels)}


if __name__ == "__main__":
    from readers import ner_reader
    import configparser
    from transformers import BertTokenizer
    import random

    cfg = '../config.cfg'
    random.seed(5)
    np.random.seed(5)
    torch.manual_seed(5)
    config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
    config.read(cfg)

    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
    ds = 'smcalflow_cs_seqlabeling'
    train_data = NERDataset(ner_reader.load_data(config.get('Files', '{}_valid'.format(ds)), ds=ds), tokenizer=tokenizer,
                            max_seq_len=128, split_seqs=True)
    train_dataloader = DataLoader(train_data, shuffle=False, batch_size=1, collate_fn=train_data.collate)
    for toks, labels in train_data:
        for t, l in zip(toks, labels):
            print(tokenizer.convert_ids_to_tokens(t), l)
    for key, val in train_data.id2label.items():
        print(key, val)