from torch import nn
from transformers import BertForTokenClassification, BertConfig, BertTokenizer, FlaubertTokenizer, FlaubertForTokenClassification
import test_config

class NERModel(nn.Sequential):

    def __init__(self, num_labels: int, model_name_or_path: str = None):
        super(NERModel, self).__init__()
        self.model = self.load_model(model_name_or_path, num_labels)

    def load_model(self, model_name_or_path, num_labels):
        if model_name_or_path == 'bert-base-multilingual-cased':
            model = BertForTokenClassification.from_pretrained(model_name_or_path, num_labels=num_labels)
        elif model_name_or_path == 'bert-base-cased':
            model = BertForTokenClassification.from_pretrained(model_name_or_path, num_labels=num_labels)
        elif model_name_or_path == 'clinical-bert':
            model = BertForTokenClassification.from_pretrained('emilyalsentzer/Bio_ClinicalBERT', num_labels=num_labels)
        elif model_name_or_path == 'beto-bert':
            model = BertForTokenClassification.from_pretrained('dccuchile/bert-base-spanish-wwm-cased', num_labels=num_labels)
        elif model_name_or_path == 'romanian-bert' or model_name_or_path=='romanian-bert_upper':
            model = BertForTokenClassification.from_pretrained('dumitrescustefan/bert-base-romanian-cased-v1',
                                                               num_labels=num_labels)
        elif model_name_or_path == 'biobert' or model_name_or_path=='biobert_upper':
            model = BertForTokenClassification.from_pretrained('dmis-lab/biobert-base-cased-v1.1',
                                                               num_labels=num_labels)
        elif model_name_or_path == 'flaubert':
            model = FlaubertForTokenClassification.from_pretrained('flaubert/flaubert_base_cased',
                                                               num_labels=num_labels)
        elif model_name_or_path == 'pt-biobert':
            model = BertForTokenClassification.from_pretrained('pucpr/biobertpt-bio',
                                                               num_labels=num_labels)
        elif model_name_or_path == 'pt-biobert_lower':
            model = BertForTokenClassification.from_pretrained('pucpr/biobertpt-bio',
                                                               num_labels=num_labels)
        elif model_name_or_path == 'pt-bert':
            model = BertForTokenClassification.from_pretrained('neuralmind/bert-base-portuguese-cased',
                                                               num_labels=num_labels)


        elif model_name_or_path == 'from_config':
            #test_config.test_config['num_labels'] = num_labels
            bert_config = BertConfig.from_dict(test_config.test_config)
            bert_config.num_labels = num_labels
            model = BertForTokenClassification(bert_config)
        else:
            model = BertForTokenClassification.from_pretrained(model_name_or_path,
                                                               num_labels=num_labels)
        return model

def load_tokenizer(model_name_or_path):
    if model_name_or_path == 'bert-base-multilingual-cased':
        tokenizer = BertTokenizer.from_pretrained(model_name_or_path)
    elif model_name_or_path == 'bert-base-cased':
        tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    elif model_name_or_path == 'clinical-bert':
        tokenizer = BertTokenizer.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')
    elif model_name_or_path == 'beto-bert':
        tokenizer = BertTokenizer.from_pretrained('dccuchile/bert-base-spanish-wwm-cased')
    elif model_name_or_path == 'romanian-bert':
        tokenizer = BertTokenizer.from_pretrained('dumitrescustefan/bert-base-romanian-cased-v1')
    elif model_name_or_path == 'romanian-bert_upper':
        tokenizer = BertTokenizer.from_pretrained('dumitrescustefan/bert-base-romanian-cased-v1',  do_lower_case = False)
    elif model_name_or_path == 'biobert':
        tokenizer = BertTokenizer.from_pretrained('dmis-lab/biobert-base-cased-v1.1')
    elif model_name_or_path == 'biobert_upper':
        tokenizer = BertTokenizer.from_pretrained('dmis-lab/biobert-base-cased-v1.1', do_lower_case = False)
    elif model_name_or_path == 'from_config':
        tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    elif model_name_or_path == 'adx':
        tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
    elif model_name_or_path == 'ptbiobert':
        tokenizer = BertTokenizer.from_pretrained('dmis-lab/biobert-base-cased-v1.1')
    elif model_name_or_path == 'flaubert':
        tokenizer = FlaubertTokenizer.from_pretrained('flaubert/flaubert_base_cased')
    elif model_name_or_path == 'pt-biobert':
        tokenizer = BertTokenizer.from_pretrained('pucpr/biobertpt-bio',  do_lower_case=False)
    elif model_name_or_path == 'pt-biobert_lower':
        tokenizer = BertTokenizer.from_pretrained('pucpr/biobertpt-bio', do_lower_case=True)
    elif model_name_or_path == 'pt-bert':
        tokenizer = BertTokenizer.from_pretrained('neuralmind/bert-base-portuguese-cased', do_lower_case=False)
    return tokenizer

if __name__=="__main__":
    name = 'pt-biobert_lower'
    model = NERModel(model_name_or_path=name, num_labels=10)
    tokenizer = load_tokenizer(name)
    #tokenizer.config.do_lower_case = False

    #print(tok.lower_case)
    toks = tokenizer.tokenize('HELLO')
    for tok in toks:
        i = tokenizer.convert_tokens_to_ids(tok)
        print(tokenizer.convert_ids_to_tokens(i))
