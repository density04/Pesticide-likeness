# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 18:04:58 2022

@author:Jinyu-Sun
"""

# coding=utf-8
import timeit  # Import the timing module
import sys  # Import the system module
import os
import numpy as np  # Import the NumPy library
import math  # Import the math library
import torch  # Import the PyTorch library
import torch.nn as nn  # Import the PyTorch neural network module
import torch.nn.functional as F  # Import PyTorch functional neural network module
import torch.optim as optim  # Import the PyTorch optimizer module
import pickle  # Import the pickle module for serialization
from sklearn.metrics import roc_auc_score, roc_curve  # Import functions for ROC AUC score and ROC curve calculation
from sklearn.metrics import confusion_matrix  # Import the confusion matrix calculation function
# import preprocess as pp  # Import the custom preprocessing module
import pandas as pd  # Import the Pandas library for data processing
import matplotlib.pyplot as plt  # Import the Matplotlib library for plotting
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import dataset.DGCAN_Dataset as pp


# Check whether GPU is available and use it if possible
if torch.cuda.is_available():
    device = torch.device('cuda')  # Use GPU
else:
    device = torch.device('cpu')  # Use CPU

torch.cuda.empty_cache()  # Clear the CUDA cache


class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()  # Initialize the parent class
        self.dropout = dropout  # Dropout rate
        self.concat = concat  # Whether to concatenate
        self.in_features = in_features  # Input feature dimension
        self.out_features = out_features  # Output feature dimension
        self.alpha = alpha  # Negative slope of LeakyReLU
        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))  # Weight matrix

        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))  # Attention coefficients
        torch.nn.init.xavier_uniform_(self.W, gain=2.0)  # Initialize weights with Xavier initialization
        torch.nn.init.xavier_uniform_(self.W, gain=1.9)  # Reinitialize with Xavier initialization
        self.leakyrelu = nn.LeakyReLU(self.alpha)  # LeakyReLU activation function

    def forward(self, input, adj):  # Define the forward pass
        """
        input: input features [N, in_features], where N is the number of nodes
        adj: graph adjacency matrix with shape [N, N]; nonzero values indicate edges
        """
        # h = torch.mm(input.cpu(), self.W.cpu())  # Calculate the linear transformation of node features [N, out_features]
        h = torch.mm(input, self.W) 
        N = h.size()[0]  # Number of nodes in the graph
        # Calculate attention coefficients
        a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1).view(N, -1,
                                                                                          2 * self.out_features)  # [N, N, 2*out_features]
        # e = self.leakyrelu(torch.matmul(a_input.cpu(), self.a.cpu()).squeeze(2))  # Calculate attention weights
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))
        zero_vec = -9e10 * torch.ones_like(e)  # Create a vector with very small values
        attention = torch.where(adj > 0, e, zero_vec)  # Generate attention weights from the adjacency matrix
        # If the corresponding adjacency entry is greater than 0, keep that attention weight; otherwise set it to a very small value
        attention = F.softmax(attention, dim=1)  # Apply softmax normalization to attention weights
        attention = F.dropout(attention, self.dropout, training=self.training)  # Apply dropout
        h_prime = torch.matmul(attention, h)  # Calculate weighted features
        if self.concat:
            return F.elu(h_prime)  # Return after ELU activation
        else:
            return h_prime  # Return weighted features


class GAT(nn.Module):
    def __init__(self, nfeat, nhid, dropout, alpha, nheads):
        super(GAT, self).__init__()  # Initialize the parent class
        """
        n_heads indicates how many GAT layers are used; these layers are concatenated like a self-attention mechanism
        They extract features from different subspaces.
        """
        self.dropout = dropout  # Dropout rate
        # Create multiple graph attention layers
        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in
                           range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)  # Add each attention layer to the model

        self.out_att = GraphAttentionLayer(nhid, 56, dropout=dropout, alpha=alpha, concat=False)  # Output GAT layer
        self.nheads = nheads  # Number of heads

    def forward(self, x, adj):  # Define the forward pass
        x = F.dropout(x, self.dropout, training=self.training)  # Apply dropout
        z = torch.zeros_like(self.attentions[1](x, adj))  # Initialize weighted features
        # Aggregate the outputs of all attention layers
        for att in self.attentions:
            z = torch.add(z, att(x, adj))  # Accumulate each attention layer output
        x = z / self.nheads  # Calculate the average across heads
        x = F.dropout(x, self.dropout, training=self.training)  # Apply dropout again
        x = F.elu(self.out_att(x, adj))  # Pass through the final GAT layer and ELU activation
        return F.softmax(x, dim=1)  # Return the softmax output


class MolecularGraphNeuralNetwork(nn.Module):
    def __init__(self, N_fingerprints, dim, layer_hidden, layer_output, dropout):
        super(MolecularGraphNeuralNetwork, self).__init__()  # Initialize the parent class
        self.layer_hidden = layer_hidden  # Number of hidden layers
        self.layer_output = layer_output  # Number of output layers
        self.embed_fingerprint = nn.Embedding(N_fingerprints, dim)  # Define the fingerprint embedding layer
        self.W_fingerprint = nn.ModuleList([nn.Linear(dim, dim) for _ in range(layer_hidden)])  # Define multiple linear layers

        self.W_output = nn.ModuleList([nn.Linear(56, 56) for _ in range(layer_output)])  # Define the linear layers in the output layer
        self.W_property = nn.Linear(56, 2)  # Define the property prediction layer

        self.dropout = dropout  # Dropout rate
        self.alpha = 0.25  # Negative slope of LeakyReLU
        self.nheads = 2  # Number of attention heads
        self.attentions = GAT(dim, dim, dropout, alpha=self.alpha, nheads=self.nheads).to(device)  # Initialize the GAT layer and move it to the device

    def pad(self, matrices, pad_value):
        """Pad the matrix list for batch processing.
        For example, given the matrix list [A, B, C],
        we get a new matrix [A00, 0B0, 00C], where 0 is the padding value.
        """
        shapes = [m.shape for m in matrices]  # Get the matrix shapes
        M, N = sum([s[0] for s in shapes]), sum([s[1] for s in shapes])  # Calculate the size of the padded matrix
        zeros = torch.FloatTensor(np.zeros((M, N))).to(device)  # Create an all-zero padding matrix
        pad_matrices = pad_value + zeros  # Initialize the padding matrix
        i, j = 0, 0  # Initialize indices
        for k, matrix in enumerate(matrices):
            m, n = shapes[k]  # Get each matrix shape
            pad_matrices[i:i + m, j:j + n] = matrix  # Fill the new matrix with the current matrix
            i += m  # Update the row index
            j += n  # Update the column index
        return pad_matrices  # Return the padded matrix

    def update(self, matrix, vectors, layer):
        hidden_vectors = torch.relu(self.W_fingerprint[layer](vectors))  # Calculate hidden vectors

        return hidden_vectors + torch.matmul(matrix, hidden_vectors)  # Return updated vectors

    def sum(self, vectors, axis):
        sum_vectors = [torch.sum(v, 0) for v in torch.split(vectors, axis)]  # Sum vectors along the axis
        return torch.stack(sum_vectors)  # Return the stacked vectors

    def gnn(self, inputs):
        """Concatenate or pad each input item for batch processing."""
        Smiles, fingerprints, adjacencies, molecular_sizes = inputs  # Unpack inputs
        fingerprints = torch.cat(fingerprints)  # Concatenate fingerprints
        # fingerprints=fingerprints.cpu()
        adj = self.pad(adjacencies, 0)  # Pad adjacency matrices
        """GNN layer for updating fingerprint vectors."""
        fingerprint_vectors = self.embed_fingerprint(fingerprints)  # Generate fingerprint vectors through the embedding layer

        for l in range(self.layer_hidden):  # Iterate over each hidden layer
            # hs = self.update(adj.cpu(), fingerprint_vectors.cpu(), l)  # Update fingerprint vectors
            hs = self.update(adj, fingerprint_vectors, l) 
            fingerprint_vectors = F.normalize(hs, 2, 1)  # Apply L2 normalization to the updated vectors
        """Attention layer"""
        # molecular_vectors = self.attentions(fingerprint_vectors.cpu(), adj.cpu())  # Obtain molecular vectors through the GAT layer
        molecular_vectors = self.attentions(fingerprint_vectors, adj)
        """Obtain molecular vectors by summing or averaging fingerprint vectors."""
        molecular_vectors = self.sum(molecular_vectors, molecular_sizes)  # Sum by molecular size
        return Smiles, molecular_vectors  # Return SMILES and molecular vectors

    def mlp(self, vectors):
        """Regressor based on a multilayer perceptron."""
        for l in range(self.layer_output):  # Iterate over output layers
            vectors = torch.relu(self.W_output[l](vectors))  # Activate output vectors
        outputs = torch.sigmoid(self.W_property(vectors))  # Get the final output through sigmoid
        return outputs  # Return outputs

    def forward_classifier(self, data_batch, train):
        inputs = data_batch[:-1]  # Get input data
        correct_labels = torch.cat(data_batch[-1])  # Get correct labels

        if train:  # If this is training mode
            Smiles, molecular_vectors = self.gnn(inputs)  # Run the GNN forward pass
            predicted_scores = self.mlp(molecular_vectors)  # Run the MLP forward pass
            '''Loss function'''
            loss = F.cross_entropy(predicted_scores, correct_labels)  # Calculate cross-entropy loss
            predicted_scores = predicted_scores.to('cpu').data.numpy()  # Move predicted scores to CPU and convert them to a NumPy array
            predicted_scores = [s[1] for s in predicted_scores]  # Extract the prediction scores for the second class
            correct_labels = correct_labels.to('cpu').data.numpy()  # Move correct labels to CPU and convert them to a NumPy array
            return Smiles, loss, predicted_scores, correct_labels  # Return SMILES, loss, predicted scores, and correct labels
        else:  # If this is test mode
            with torch.no_grad():  # Do not calculate gradients
                Smiles, molecular_vectors = self.gnn(inputs)  # Run the GNN forward pass
                predicted_scores = self.mlp(molecular_vectors)  # Run the MLP forward pass
                # loss = F.cross_entropy(predicted_scores.cpu(), correct_labels.cpu())  # Calculate cross-entropy loss
                loss = F.cross_entropy(predicted_scores, correct_labels)
            predicted_scores = predicted_scores.to('cpu').data.numpy()  # Move predicted scores to CPU and convert them to a NumPy array
            predicted_scores = [s[1] for s in predicted_scores]  # Extract the prediction scores for the second class
            correct_labels = correct_labels.to('cpu').data.numpy()  # Move correct labels to CPU and convert them to a NumPy array
            
            return Smiles, loss, predicted_scores, correct_labels  # Return SMILES, loss, predicted scores, and correct labels
    def test(self, data) :
        self.eval()
        data=pp.create_testdataset(data,device=device)
        DCAN_tester = Tester(self, batch_test=8)
        dcan_prediction_raw = DCAN_tester.test_classifier(data)
        return dcan_prediction_raw


class Trainer(object):
    def __init__(self, model, lr, batch_train):
        self.model = model  # Store the model
        self.batch_train = batch_train  # Store the training batch size
        self.lr = lr  # Store the learning rate
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)  # Use the Adam optimizer

    def train(self, dataset):
        np.random.shuffle(dataset)  # Shuffle the dataset
        N = len(dataset)  # Get the dataset size
        # N = dataset.shape[0]
        loss_total = 0  # Initialize total loss
        SMILES, P, C = '', [], []  # Initialize SMILES, prediction, and correct-label lists
        for i in range(0, N, self.batch_train):  # Iterate over the dataset by batch size
            data_batch = list(zip(*dataset[i:i + self.batch_train]))  # Get the current batch data
            Smiles, loss, predicted_scores, correct_labels = self.model.forward_classifier(data_batch,
                                                                                           train=True)  # Run the forward pass and calculate loss
            SMILES += ' '.join(Smiles) + ' '  # Concatenate SMILES
            P.append(predicted_scores)  # Store predicted scores
            C.append(correct_labels)  # Store correct labels
            self.optimizer.zero_grad()  # Zero gradients
            loss.backward()  # Backpropagate
            self.optimizer.step()  # Update parameters
            loss_total += loss.item()  # Accumulate loss
        tru = np.concatenate(C)  # Concatenate all correct labels
        pre = np.concatenate(P)  # Concatenate all predicted scores
        AUC = roc_auc_score(tru, pre)  # Calculate the AUC score
        SMILES = SMILES.strip().split()  # Remove extra spaces and split SMILES
        pred = [1 if i > 0.15 else 0 for i in pre]  # Generate predicted labels from the threshold
        predictions = np.stack((tru, pred, pre))  # Stack prediction results
        return AUC, loss_total, predictions  # Return AUC, loss, and prediction results


class Tester(object):
    def __init__(self, model, batch_test):
        self.model = model  # Store the model
        self.batch_test = batch_test  # Store the test batch size

    def test_classifier(self, dataset):
        N = len(dataset)  # Get the dataset size
        loss_total = 0  # Initialize total loss
        SMILES, P, C = '', [], []  # Initialize SMILES, prediction, and correct-label lists
        for i in range(0, N, self.batch_test):  # Iterate over the dataset by batch size
            data_batch = list(zip(*dataset[i:i + self.batch_test]))  # Get the current batch data
            (Smiles, loss, predicted_scores, correct_labels) = self.model.forward_classifier(data_batch,
                                                                                             train=False)  # Run the forward pass
            SMILES += ' '.join(Smiles) + ' '  # Concatenate SMILES
            loss_total += loss.item()  # Accumulate loss
            P.append(predicted_scores)  # Store predicted scores
            C.append(correct_labels)  # Store correct labels
        SMILES = SMILES.strip().split()  # Remove extra spaces and split SMILES
        tru = np.concatenate(C)  # Concatenate all correct labels
        pre = np.concatenate(P)  # Concatenate all predicted scores
        pred = [1 if i > 0.15 else 0 for i in pre]  # Generate predicted labels from the threshold
        # AUC = roc_auc_score(tru, pre)  # Calculate the AUC score
        cnf_matrix = confusion_matrix(tru, pred)  # Calculate the confusion matrix
        predictions = np.stack((tru, pred, pre))  # Stack prediction results
        return  predictions 
        # return predicted_scores







class Tester(object):
    def __init__(self, model, batch_test):
        self.model = model  # Store the model
        self.batch_test = batch_test  # Store the test batch size

    def test_classifier(self, dataset):
        N = len(dataset)  # Get the dataset size
        loss_total = 0  # Initialize total loss
        SMILES, P, C = '', [], []  # Initialize SMILES, prediction, and correct-label lists
        for i in range(0, N, self.batch_test):  # Iterate over the dataset by batch size
            data_batch = list(zip(*dataset[i:i + self.batch_test]))  # Get the current batch data
            (Smiles, loss, predicted_scores, correct_labels) = self.model.forward_classifier(data_batch,
                                                                                             train=False)  # Run the forward pass
            SMILES += ' '.join(Smiles) + ' '  # Concatenate SMILES
            loss_total += loss.item()  # Accumulate loss
            P.append(predicted_scores)  # Store predicted scores
            C.append(correct_labels)  # Store correct labels
        SMILES = SMILES.strip().split()  # Remove extra spaces and split SMILES
        tru = np.concatenate(C)  # Concatenate all correct labels
        pre = np.concatenate(P)  # Concatenate all predicted scores
        pred = [1 if i > 0.15 else 0 for i in pre]  # Generate predicted labels from the threshold
        # AUC = roc_auc_score(tru, pre)  # Calculate the AUC score
        cnf_matrix = confusion_matrix(tru, pred)  # Calculate the confusion matrix
        predictions = np.stack((tru, pred, pre))  # Stack prediction results
        return  predictions[2,:]
        # return predicted_scores
