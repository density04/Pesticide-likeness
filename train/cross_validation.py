import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from models import RNNLM, GCNModel, DGCAN
import torch
import numpy as np 
import random
from models.RNN import rnnlm_trainer
from models.GCN import gcn_trainer
from sklearn.model_selection import KFold
from train.metrics import *
import pickle
import dataset.DGCAN_Dataset as pp

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

np.seterr(invalid='ignore')
torch.manual_seed(123)
torch.cuda.manual_seed(123)
np.random.seed(123)
random.seed(123)
torch.backends.cudnn.deterministic = True
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using %s"%device)



pos_data=list(open("data/ins.txt"))
neg_data=list(open("data/neg.txt"))[:len(pos_data)]
train_data=pos_data+neg_data
answer = [1 for _ in range(len(pos_data))] + [0 for _ in range(len(neg_data))]
whole_data=(np.array(train_data),np.array(answer))
dcan_data=pp.get_dcan_dataset(pos_data,neg_data)




kf = KFold(n_splits=10, shuffle=True, random_state=42)


metric_list=['acc','auc','bacc','f1','mcc','pre','rec','q_','sp','y_true','y_scores']
for metrics_names in metric_list:
    exec('rnn_'+metrics_names+'_'+'=[]')
for metrics_names in metric_list:
    exec('gcn_'+metrics_names+'_'+'=[]')
for metrics_names in metric_list:
    exec('dcan_'+metrics_names+'_'+'=[]')


k=1
for train_index, test_index in kf.split(whole_data[0]):
    gcn_train_data=(np.array(whole_data[0])[train_index],np.array(whole_data[1])[train_index])
    gcn_test_data=(np.array(whole_data[0])[test_index],np.array(whole_data[1])[test_index])
    gcn_model = GCNModel(input_size=31, hidden_size1=32, hidden_size2=256, n_layers=4, n_heads=4, dropout=0.2)
    gcn_model=gcn_trainer(gcn_model)
    gcn_model=gcn_model.to(device)
    gcn_train_dataloader = gcn_model.construct_dataloader(train_data=gcn_train_data, batch_size=100)
    gcn_optimizer = torch.optim.Adam(gcn_model.parameters(), lr=0.001)
    for epoch in range(300):
        print('\rFold %d GCN epoch %d'%(k,epoch),end='')
        train_loss, _ = gcn_model.train_epoch(gcn_train_dataloader, None, gcn_optimizer, gradient_clip_val=1.0)
        for param_group in gcn_optimizer.param_groups :
            param_group['lr'] = 0.01 * (0.998** epoch)  # lr * (lr_decay' ^ epoch)

    gcn_score_test=gcn_model.test(gcn_test_data[0])
    gcn_test_label=gcn_test_data[1]
    auc,bacc,f1,mcc,pre,rec,acc,q_,sp,y_true,y_scores=gcn_metrics(gcn_score_test,gcn_test_label)
 
    for metrics_names in metric_list:
        exec('gcn_'+metrics_names+'_'+'.append(eval(metrics_names))')
    print("Fold %d GCN"%(k))
    print("auc:%f,bacc:%f,f1:%f,mcc:%f,pre:%f,rec:%f,acc:%f,q_:%f,sp:%f"%(auc,bacc,f1,mcc,pre,rec,acc,q_,sp))
    fm='gcn_model_ins__'+str(k)+'.pth'
    torch.save(gcn_model.state_dict(), fm)



    rnn_model = RNNLM(input_size=66,stereo=True,hidden_size=1024,n_layers=4,dropout=0.2)
    rnn_model=rnnlm_trainer(rnn_model)
    rnn_model=rnn_model.to(device)
    rnn_model.load_state_dict(torch.load('para/save.pt'))
    rnn_train_data=gcn_train_data[0][np.where(gcn_train_data[1]==1)]
    rnn_test_posdata=gcn_test_data[0][np.where(gcn_test_data[1]==1)]
    rnn_norm_data=gcn_test_data[0][np.where(gcn_test_data[1]==0)]
    rnn_train_dataloader = rnn_model.construct_dataloader(train_data=rnn_train_data, batch_size=100)
    rnn_optimizer = torch.optim.Adam(rnn_model.parameters(), lr=0.0001)

    for epoch in range(300) :
        print('\rFold %d RNN epoch %d'%(k,epoch),end='')
        train_loss, _ = rnn_model.train_epoch(rnn_train_dataloader, None, rnn_optimizer, gradient_clip_val=1.0)
    rnn_score_test=rnn_model.test(rnn_test_posdata)
    rnn_score_norm=rnn_model.test(rnn_norm_data)
    auc,bacc,f1,mcc,pre,rec,acc,q_,sp,y_true,y_scores=rnn_metrics(rnn_score_test,rnn_score_norm)
    for metrics_names in metric_list:
        exec('rnn_'+metrics_names+'_'+'.append(eval(metrics_names))')
    print("Fold %d RNN"%(k))
    print("auc:%f,bacc:%f,f1:%f,mcc:%f,pre:%f,rec:%f,acc:%f,q_:%f,sp:%f"%(auc,bacc,f1,mcc,pre,rec,acc,q_,sp))
    fm='rnn_model_ins__'+str(k)+'.pth'
    torch.save(rnn_model.state_dict(), fm)


    DCAN_train_data=np.array(dcan_data,dtype=object)[train_index]
    DCAN_val_data=gcn_test_data
    DCAN_train_data=pp.to_gpu(DCAN_train_data,device)
    decay_interval=25
    lr_decay=0.85
    DCAN_model = DGCAN.MolecularGraphNeuralNetwork(5000,dim=52, layer_hidden=4, layer_output=10, dropout=0.45).to(device)
    DCAN_trainer = DGCAN.Trainer(DCAN_model, lr=3e-4, batch_train=8)
    # DCAN_tester = DGCAN.Tester(DCAN_model, batch_test=8)
    for epoch in range(150):
        print('\rFold %d DCAN epoch %d'%(k,epoch),end='')
        if epoch % decay_interval == 0:
            DCAN_trainer.optimizer.param_groups[0]['lr'] *= lr_decay
        prediction_train, loss_train, train_res = DCAN_trainer.train(DCAN_train_data)

#DCAN
    test_res = DCAN_model.test(data=DCAN_val_data[0])

    auc,bacc, pre, rec, f1, mcc, sp, q_, acc=dcan_metrics(test_res,DCAN_val_data[1])

    for metrics_names in metric_list:
        exec('dcan_'+metrics_names+'_'+'.append(eval(metrics_names))')
    print("Fold %d DCAN"%(k))
    print("auc:%f,bacc:%f,f1:%f,mcc:%f,pre:%f,rec:%f,acc:%f,q_:%f,sp:%f"%(auc,bacc,f1,mcc,pre,rec,acc,q_,sp))
    fm='dcan_model_ins__'+str(k)+'.pth'
    torch.save(DCAN_model.state_dict(), fm)


    
    # Write the recorded lists into dictionaries
    gcn_metric={i:eval('gcn_'+i+'_') for i in metric_list}
    rnn_metric={i:eval('rnn_'+i+'_') for i in metric_list}
    dcan_metric={i:eval('dcan_'+i+'_') for i in metric_list}
    with open('multi_rnn_ins2.pkl', 'wb') as f:
        pickle.dump(rnn_metric, f)
    with open('multi_gcn_ins2.pkl', 'wb') as f:
        pickle.dump(gcn_metric, f)
    with open('multi_dcan_ins2.pkl', 'wb') as f:
        pickle.dump(dcan_metric, f)
    k+=1