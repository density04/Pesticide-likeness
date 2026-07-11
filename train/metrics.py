import numpy as np
from sklearn.metrics import roc_curve
from sklearn.metrics import auc as auc_ 
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import math

def rnn_metrics(score_test, score_norm):
    score_test=list(np.array(score_test)[np.where(score_test)!=0].squeeze())
    y_true = np.r_[np.ones(len(score_test)),np.zeros(len(score_norm))]

    # Model-predicted probabilities
    y_scores = np.r_[score_test,score_norm]
    y_pre=np.array([1 if i>70 else 0 for i in y_scores])
    # Calculate FPR, TPR, and thresholds
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    tn,fp,fn,tp=confusion_matrix(y_true,y_pre).ravel()
    # Calculate AUC
    roc_auc = auc_(fpr, tpr)
    bacc=balanced_accuracy_score(y_true, y_pre, adjusted=False)
    f1 = f1_score(y_true, y_pre)
    mcc=matthews_corrcoef(y_true, y_pre)
    pre=precision_score(y_true, y_pre)
    rec=recall_score(y_true, y_pre)
    acc=accuracy_score(y_true,y_pre)
    q_=accuracy_score(y_true[np.where(y_true==0)[0]],y_pre[np.where(y_true==0)[0]])
    sp=tn/(tn+fp)
    return (roc_auc,bacc,f1,mcc,pre,rec,acc,q_,sp,y_true,y_scores)




def gcn_metrics(score_test,label):
    y_true = np.array(label)
    y_scores = np.array(score_test)
    y_pre=np.array([1 if i>0.5 else 0 for i in y_scores])
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    tn,fp,fn,tp=confusion_matrix(y_true,y_pre).ravel()
    roc_auc = auc_(fpr, tpr)
    bacc=balanced_accuracy_score(y_true, y_pre, adjusted=False)
    f1 = f1_score(y_true, y_pre)
    mcc=matthews_corrcoef(y_true, y_pre)
    pre=precision_score(y_true, y_pre)
    rec=recall_score(y_true, y_pre)
    acc=accuracy_score(y_true,y_pre)
    q_=accuracy_score(y_true[np.where(y_true==0)[0]],y_pre[np.where(y_true==0)[0]])
    sp=tn/(tn+fp)
    return (roc_auc,bacc,f1,mcc,pre,rec,acc,q_,sp,y_true,y_scores)

def dcan_metrics(score_test,label):
    fpr, tpr, thresholds = roc_curve(label,score_test)
    auc = auc_(fpr, tpr)
    y_true = np.array(label)
    y_pre=np.array([1 if i>0.5 else 0 for i in score_test])
    tn,fp,fn,tp=confusion_matrix(y_true,y_pre).ravel()
    roc_auc = auc_(fpr, tpr)
    bacc=balanced_accuracy_score(y_true, y_pre, adjusted=False)
    f1 = f1_score(y_true, y_pre)
    mcc=matthews_corrcoef(y_true, y_pre)
    pre=precision_score(y_true, y_pre)
    rec=recall_score(y_true, y_pre)
    acc=accuracy_score(y_true,y_pre)
    q_=accuracy_score(y_true[np.where(y_true==0)[0]],y_pre[np.where(y_true==0)[0]])
    sp=tn/(tn+fp)

    return (roc_auc, bacc, pre, rec, f1, mcc, sp, q_, acc)