"""
Code for transforming semantic parsing datasets into sequences labels with slot values
"""

import re
from utils import load_jsonl, write_jsonl


def get_first_type(s):
    pattern = re.compile(':([^ ]+)')
    m = re.findall(pattern, s)
    if len(m) > 0:
        return m[0]
    else:
        return None


def assign_labels(utterance, alignments):
    utterance = utterance.split()
    labels = ['O'] * len(utterance)
    for a in alignments:
        t = get_first_type(' '.join(a['target_tokens']))
        assert t != None
        for i, idx in enumerate(range(a['source_tokens_idx'][0], a['source_tokens_idx'][1])):
            if i == 0:
                labels[idx] = 'B-' + t.upper()
            else:
                labels[idx] = 'I-' + t.upper()
    assert len(utterance) == len(labels)
    return utterance, labels

if __name__=="__main__":
    split = 'test'
    alignment_file = '/home/mareike/data/smcalflow_cs/data/smcalflow_cs/calflow.orgchart.event_create/source_domain_with_target_num8/{}.alignment.jsonl'.format(split)
    data_file = '/home/mareike/data/smcalflow_cs/data/smcalflow_cs/calflow.orgchart.event_create/source_domain_with_target_num8/{}.jsonl'.format(split)
    fname_out = '/home/mareike/data/multiskill/seq_labeling/source_domain_with_target_num0_from8_seqlabeling_{}.jsonl'.format(split)

    alignments = load_jsonl(alignment_file)
    data = load_jsonl(data_file)

    data_out = []
    for i, elm in enumerate(data):
        # filter out compositional training examples
        if split in ['train', 'valid']:
            if 'events_with_orgchart' not in elm['tags']:
                a = alignments[i]['alignments']
                utterance = elm['utterance']
                did = elm['dialogue_id']
                utterance, labels = assign_labels(utterance, a)
                new_elm = {'did': did,
                       'seq': utterance,
                       'labels': labels
                       }
                data_out.append(new_elm)
        else:
            a = alignments[i]['alignments']
            utterance = elm['utterance']
            did = elm['dialogue_id']
            utterance, labels = assign_labels(utterance, a)
            new_elm = {'did': did,
                       'seq': utterance,
                       'labels': labels
                       }
            data_out.append(new_elm)

    print(len(data))
    print(len(data_out))

    write_jsonl(fname_out, data_out)

