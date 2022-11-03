# Ambuj Kumar Pandit - Photometric Redshift prediction using mixed model input
#-----------------------------------------------------------------------------------------------
from tqdm import tqdm
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
import torchvision
from torchvision import datasets, models, transforms
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
from torch.autograd import Variable


# run parameters.
BATCH_SIZE = 64
learning_rate = 3e-4
NUM_EPOCHS = 200
NUM_WORKER = 2
save_path = "./logs/"
run_id = 10 #experiment id
#-------------------------------------------------------
from utils import MyDataset

transform = transforms.Compose([transforms.ToTensor(),transforms.CenterCrop(32),transforms.RandomHorizontalFlip(),transforms.RandomVerticalFlip()])

file = np.load("../data/cube_part1.npy") 
labels = np.load("../data/labels_part1.npy")
dataset = MyDataset(file, labels, mm=True, transform=transform)
#del file
#del labels
# TODO: Reduce the ram requirements by closing the files.

print("Data loading success!")

train_count = int(0.8 * len(dataset))
test_count = len(dataset) - train_count
train_dataset, test_dataset = torch.utils.data.random_split(dataset, (train_count, test_count))
train_dataset_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKER)
test_dataset_loader  = torch.utils.data.DataLoader(test_dataset , batch_size=BATCH_SIZE, shuffle=False,num_workers=NUM_WORKER)
dataloader = {'train': train_dataset_loader, 'test': test_dataset_loader}

print(f"train: {len(train_dataset)} test: {len(test_dataset)} samples. ")

#check gpu support.
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
if device :
    print("Using CUDA.")
#---------------------------------------------------------------------
# Define model and loss.
from utils import Metrics,format_,weight_init
from models import *

model = A4()

from torchinfo import summary
summary(model,batch_dim=BATCH_SIZE)

model = model.cuda() if use_cuda else model
model = model.apply(weight_init)
optimizer = optim.Adam(model.parameters(),lr=learning_rate, weight_decay=5e-4)
criterion = nn.HuberLoss(delta=0.1,reduction='sum')
scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=10, gamma=0.1)
print("All initialisation done.")

#------------------------------------------------------------------------------
# Main loop
def run(model, dataloader, epochs):
    # main loop
    min_loss = 10000 # setinel value.
    for epoch in range(epochs):
        print(f"Experiment no: {run_id} Epoch: {epoch+1}")
        fname = save_path+"images/run"+str(run_id)

        #Training Loop.
        model.train()
        running_loss = 0.0
        train_metrics = [] # store the logs
        for i, data in enumerate(tqdm(dataloader["train"])):
            inputs,extra,labels = data
            inputs,extra, labels = inputs.to(device), extra.to(device),labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()
            outputs = model(inputs,extra).squeeze()  #debug.
            loss = criterion(outputs, labels)  #implement a custom loss.
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            #Collect the data.
            na = outputs.detach().to('cpu').numpy()
            nb = labels.detach().to('cpu').numpy()
            result = Metrics(na,nb)
            result["loss"] = loss.item()
            train_metrics.append(result)

        #generate report.
        ta,tb,tc,td,te = format_(train_metrics, fname+"train"+str(epoch+1))
        print(f'Training | nmad: {round(tc,6)}, bias: {round(td,6)}, loss: {round(ta,9)}, MAE: {round(tb,6)}, outliers %: {round(te,3)}')

        #Testing Loop.
        model.eval()
        running_loss=0.0
        test_metrics = [] #store the logs
        for i,data in enumerate(tqdm(dataloader["test"])):
            inputs,extra,labels = data
            inputs,extra,labels = inputs.to(device), extra.to(device), labels.to(device)
            outputs = model(inputs,extra).squeeze()  #debug.
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            # collect the data.
            na = outputs.detach().to('cpu').numpy()
            nb = labels.detach().to('cpu').numpy()
            result = Metrics(na,nb)
            result["loss"] = loss.item()
            test_metrics.append(result)
        #generate report.
        va,vb,vc,vd,ve = format_(test_metrics, fname+"test"+str(epoch+1))
        print(f'Testing | nmad: {round(vc,6)}, bias: {round(vd,6)}, loss: {round(va,9)}, MAE: {round(vb,6)}, outliers %: {round(ve,3)}')

        #Saving the model.
        if (va <= min_loss):
            print("Saving the model.")
            torch.save(model.state_dict(), save_path +"run"+str(run_id)+".pth")
            min_loss = va

#------------------------------------------------------------------------------

#----------------------------------------------------------------------------

run(model,dataloader,NUM_EPOCHS)
print("All Done!")
