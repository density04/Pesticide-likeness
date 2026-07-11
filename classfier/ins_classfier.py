import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import torch
from models import RNNLM, GCNModel, DGCAN  #Importing models runs __init__.py and exposes RNNLM from it
import numpy as np
import dataset.DGCAN_Dataset as pp
import pandas as pd

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

def ins_classifer(filepath):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pos_data=list(open("data/ins1.txt"))
    neg_data=list(open("data/neg.txt"))[:len(pos_data)]
    dcan_data=pp.get_dcan_dataset(pos_data,neg_data) # Set the atom dictionary

    data=list(open(filepath))

    gcn_model = GCNModel(input_size=31, hidden_size1=32, hidden_size2=256, n_layers=4, n_heads=4, dropout=0.2)
    gcn_model.load_state_dict(torch.load('models/para/gcn_ins.pth'))
    gcn_model=gcn_model.to(device)
    gcn_prediction_raw=gcn_model.test(data)

    rnn_model = RNNLM(input_size=66,stereo=True,hidden_size=1024,n_layers=4,dropout=0.2)
    rnn_model.load_state_dict(torch.load('models/para/rnn_ins.pth'))
    rnn_model=rnn_model.to(device)
    rnn_prediction_raw=rnn_model.test(data)

    DCAN_model = DGCAN.MolecularGraphNeuralNetwork(5000,dim=52, layer_hidden=4, layer_output=10, dropout=0.45)
    DCAN_model.load_state_dict(torch.load('models/para/dcan_ins.pth'))
    DCAN_model=DCAN_model.to(device)
    dcan_prediction_raw=DCAN_model.test(data=data)
    
    
    
    result=pd.DataFrame({'smiles':data,'rnn':np.zeros(len(data)),'gcn':np.zeros(len(data)),'dcan':np.zeros(len(data))})
    rnn_prediction_raw=np.array(rnn_prediction_raw)
    gcn_prediction_raw=np.array(gcn_prediction_raw)
    dcan_prediction_raw=np.array(dcan_prediction_raw)

    rnn_train_min=0
    rnn_train_max=94.76294217959096
    gcn_train_min=4.77988323837053e-05
    gcn_train_max=0.9999998807907104
    dcan_train_max=1
    dcan_train_min=0
    result['rnn']=(rnn_prediction_raw-rnn_train_min)/(rnn_train_max-rnn_train_min)
    result['gcn']=(gcn_prediction_raw-gcn_train_min)/(gcn_train_max-gcn_train_min)
    result['dcan']=(dcan_prediction_raw-dcan_train_min)/(dcan_train_max-dcan_train_min)

    ol=pd.DataFrame([])
    ol['rnn']=[1 if i>0.5 else 0 for i in result['rnn']]
    ol['gcn']=[1 if i>0.5 else 0 for i in result['gcn']]
    ol['dcan']=[1 if i>0.5 else 0 for i in result['dcan']]
    ol['result']=[1 if sum(ol.iloc[i,:])>=2 else 0 for i in range(len(ol['rnn']))]
    result['type']=ol['result']
    for i in range(result.shape[0]):
        for j in range(1,4):
            result.iloc[i,j]=np.max([0,result.iloc[i,j]])
            result.iloc[i,j]=np.min([1,result.iloc[i,j]])
    points=[]
    for i in range(ol.shape[0]):
        if ol.iloc[i,3]==1:
            points.append(np.dot(ol.iloc[i,0:3],result.iloc[i,1:4])/sum(ol.iloc[i,0:3])) 
        else:
            points.append(np.dot((1-ol.iloc[i,0:3]),result.iloc[i,1:4])/sum(1-ol.iloc[i,0:3]))
    result['final_points']=points
    return result

if __name__=='__main__':
    filepath="data/1.txt"
    res=ins_classifer(filepath)
    print(res)