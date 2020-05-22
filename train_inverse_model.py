from dataset import train_loader,valid_loader
import torch
import torch.nn as nn
import numpy as np
from torch.optim import Adam
from models import CategoricalMLP
import pickle

input_dims = (4,)
hidden_dims = (64,64,64)
output_dims = (4,)
mlp = CategoricalMLP(input_dims + hidden_dims + output_dims)
MSELoss = nn.MSELoss(reduction = 'sum')
optimizer = Adam(mlp.parameters(), lr = 0.001)

train_losses = []
for epoch in range(300):
    print("Epoch %s"%epoch)
    epochLosses = []
    for batch_ndx, sample in enumerate(train_loader):
        optimizer.zero_grad()
        input = np.concatenate((sample['obj1'] , sample['obj2']),axis=1)
        input = torch.as_tensor(input,dtype=torch.float32)
        predictions = mlp(input)
        expectations = torch.as_tensor(sample['push'], dtype = torch.float32)
        loss = MSELoss(predictions,expectations)
        epochLosses.append(loss.item())
        loss.backward()
        optimizer.step()
    epochLoss = np.mean(epochLosses)
    train_losses.append(epochLoss)
    print("Epoch LOSS: %s"%epochLoss)

validationLosses = []
for batch_ndx, sample in enumerate(valid_loader):
    input = np.concatenate((sample['obj1'] , sample['obj2']),axis=1)
    input = torch.as_tensor(input,dtype=torch.float32)
    predictions = mlp(input)
    expectations = torch.as_tensor(sample['push'], dtype = torch.float32)
    loss = MSELoss(predictions,expectations)
    validationLosses.append(loss.item())
validationLoss = np.mean(validationLosses)
print("Validation MSE Loss: %s"%validationLoss)

d = {'train_loss':train_losses,'test_loss':validationLoss}

with open('./logs/train_inverse_logs.pickle', 'wb') as f:
    pickle.dump(d, f, protocol=pickle.HIGHEST_PROTOCOL)

torch.save(mlp.state_dict(),"./models/inverse_model.pt")