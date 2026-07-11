import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from models import RNNLM, GCNModel, DGCAN
import torch
import numpy as np 
import random
from models.RNN import rnnlm_trainer
from models.GCN import gcn_trainer
from train.metrics import *
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



pos_data=list(open("data/ins.txt")) # Change the training set here
neg_data=list(open("data/neg.txt"))[:len(pos_data)]
train_data=pos_data+neg_data
answer = [1 for _ in range(len(pos_data))] + [0 for _ in range(len(neg_data))]
whole_data=(np.array(train_data),np.array(answer))
dcan_data=pp.get_dcan_dataset(pos_data,neg_data)


gcn_train_data=(np.array(whole_data[0]),np.array(whole_data[1]))
gcn_test_data=(np.array(whole_data[0]),np.array(whole_data[1]))
gcn_model = GCNModel(input_size=31, hidden_size1=32, hidden_size2=256, n_layers=4, n_heads=4, dropout=0.2)
gcn_model=gcn_trainer(gcn_model)
gcn_model=gcn_model.to(device)
gcn_train_dataloader = gcn_model.construct_dataloader(train_data=gcn_train_data, batch_size=100)
gcn_optimizer = torch.optim.Adam(gcn_model.parameters(), lr=0.001)
for epoch in range(300):
    train_loss, _ = gcn_model.train_epoch(gcn_train_dataloader, None, gcn_optimizer, gradient_clip_val=1.0)
    for param_group in gcn_optimizer.param_groups :
        param_group['lr'] = 0.01 * (0.998** epoch)  # lr * (lr_decay' ^ epoch)
torch.save(gcn_model.state_dict(), 'gcn_model_all_fun.pth')


rnn_model = RNNLM(input_size=66,stereo=True,hidden_size=1024,n_layers=4,dropout=0.2)
rnn_model=rnnlm_trainer(rnn_model)
rnn_model=rnn_model.to(device)
rnn_model.load_state_dict(torch.load('models/para/save.pt'))
rnn_train_data=gcn_train_data[0][np.where(gcn_train_data[1]==1)]
rnn_test_posdata=gcn_test_data[0][np.where(gcn_test_data[1]==1)]
rnn_norm_data=gcn_test_data[0][np.where(gcn_test_data[1]==0)]
rnn_train_dataloader = rnn_model.construct_dataloader(train_data=rnn_train_data, batch_size=100)
rnn_optimizer = torch.optim.Adam(rnn_model.parameters(), lr=0.0001)
for epoch in range(300) :
    train_loss, _ = rnn_model.train_epoch(rnn_train_dataloader, None, rnn_optimizer, gradient_clip_val=1.0)
torch.save(rnn_model.state_dict(), 'rnn_model_all_fun.pth')


DCAN_train_data=np.array(dcan_data,dtype=object)
DCAN_val_data=gcn_test_data
DCAN_train_data=pp.to_gpu(DCAN_train_data,device)
DCAN_model = DGCAN.MolecularGraphNeuralNetwork(5000,dim=52, layer_hidden=4, layer_output=10, dropout=0.45).to(device)
DCAN_trainer = DGCAN.Trainer(DCAN_model, lr=3e-4, batch_train=8)
for epoch in range(150):
    if epoch % 25 == 0: # Decay every 25 epochs
        DCAN_trainer.optimizer.param_groups[0]['lr'] *= 0.85 # Decay
    prediction_train, loss_train, train_res = DCAN_trainer.train(DCAN_train_data)
torch.save(DCAN_model.state_dict(), 'dcan_model_all_fun.pth')