from seqeval.metrics import classification_report, f1_score
from sklearn.metrics import classification_report as prf_report
from sklearn.metrics import f1_score as classification_f1_score

def compute_seqacc(preds, golds, id2label_map):
    y_true, y_pred = [], []

    def trim(predict, label):

        temp_1 = []
        temp_2 = []
        for j, m in enumerate(predict):
            if j == 0:
                continue
            if id2label_map[label[j]] != 'X':
                temp_1.append(id2label_map[label[j]])
                temp_2.append(id2label_map[m])
        #temp_1.pop()
        #temp_2.pop()
        y_true.append(temp_1)
        y_pred.append(temp_2)

    for predict, label in zip(preds, golds):

        trim(predict, label)
    report = classification_report(y_true, y_pred, digits=4)
    f1 = f1_score(y_true, y_pred)
    return f1, report

def compute_prf(preds, golds):
    report = prf_report(golds, preds, digits=4)
    f1 = classification_f1_score(golds, preds, average='micro')
    return f1, report



if __name__=="__main__":
    label_map = {'ANAT': 0, 'CHEM': 1, 'CLS': 2, 'DEVI': 3, 'DISO': 4, 'GEOG': 5, 'I-ANAT': 6, 'I-CHEM': 7, 'I-DEVI': 8,
     'I-DISO': 9, 'I-GEOG': 10, 'I-LIVB': 11, 'I-OBJC': 12, 'I-PHEN': 13, 'I-PHYS': 14, 'I-PROC': 15, 'LIVB': 16,
     'O': 17, 'OBJC': 18, 'PHEN': 19, 'PHYS': 20, 'PROC': 21, 'SEP': 22, 'X': 23}
    rev = {val:key for key, val in label_map.items()}
    predictions=  [[0,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23], [2, 17, 23, 23, 17, 17, 1, 23, 23, 17, 17, 18, 23, 23, 17, 17, 23, 23, 23, 23, 17, 17, 0, 23, 23, 23, 23, 23, 17, 16, 17, 23, 21, 23, 23, 23, 17, 17, 23, 17, 23, 1, 17, 22]]
    golds = [[0,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23], [2, 17, 23, 23, 17, 17, 1, 23, 23, 17, 17, 18, 23, 23, 17, 17, 23, 23, 23, 23, 17, 17, 0, 23, 23, 23, 23, 23, 17, 16, 17, 23, 21, 23, 23, 23, 17, 17, 23, 17, 23, 1, 17, 22]]

    report = compute_seqacc(predictions, golds, rev)
    print(report[1])