from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import auc
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay



def accuracy(target, pred):
    correct = pred.round()==target
    correct = correct.sum()
    total = pred.shape[0]
    return (correct/total, correct, total)


def f1(target, pred, class_unbalance):
    f1 = f1_score(target, pred.round())
    weights = [x * (1/class_unbalance) if x == 1 else x + 1 for x in target]
    weighted_f1 = f1_score(target, pred.round(), sample_weight=weights)
    return f1, weighted_f1

def roc_curves(target, pred):
    fpr, tpr, thresholds = roc_curve(target, pred)
    roc_auc = roc_auc_score(target, pred)
    display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc)
    return (fpr, tpr, roc_auc, thresholds, display)

def pr_curves(target, pred):
    precision, recall, threshold = precision_recall_curve(target, pred)
    auc_pr = auc(recall, precision)
    display = PrecisionRecallDisplay(precision=precision, recall=recall)
    return (precision, recall, auc_pr, threshold, display)

def conf_matrix(target, pred):
    conf_matrix = confusion_matrix(target, pred.round())
    return conf_matrix

def sensitivity(tp, fn):
    return tp/(tp+fn)

def specificity(tn, fp):
    return tn/(tn+fp)