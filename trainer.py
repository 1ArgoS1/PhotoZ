#-----------------------------------------------------------------------------------------------
from tqdm import tqdm
import argparse
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
import torchvision
from torchvision import datasets, models, transforms
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
from torch.autograd import Variable

#------------------------------------------------------
# Run Parameters Initialisation
#------------------------------------------------------

argparser = argparse.ArgumentParser(description='Process hyper-parameters')
argparser.add_argument('--lr', type=float, default=1e-3, help='training rate')
argparser.add_argument('--epochs', type=int, default=200, help='Number of epochs')
argparser.add_argument('--num_worker', type=int, default=2, help='Number of workers feeding data')
argparser.add_argument('--id',       type=int, default='1331', help='File prefix for marking output')
argparser.add_argument('--batch_size',  type=int, default=64,   help='Batch Size of input')
argparser.add_argument('--save_path', type=str,   default='./logs/', help='Directory for logging')
argparser.add_argument('--step_lr',  type=int, default=15,   help='Step LR epoch decay')

args = argparser.parse_args()

BATCH_SIZE = args.batch_size
learning_rate = args.lr
NUM_EPOCHS = args.epochs
NUM_WORKER = args.num_worker
save_path = args.save_path
run_id = args.id
step_lr = args.step_lr
print(f"Batch size : {BATCH_SIZE}, Learning rate : {learning_rate}")

#-------------------------------------------------------
# Dataset Initialisation
#-------------------------------------------------------
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
from utils import MyDataset

transform = transforms.Compose([transforms.ToTensor(),transforms.CenterCrop(32),transforms.RandomHorizontalFlip(),transforms.RandomVerticalFlip()])

file = np.load("../data/cube.npy", mmap_mode='r', fix_imports=True)
labels = np.load("../data/labels.npy", mmap_mode='r', fix_imports=True)
dataset = MyDataset(file, labels, mm=False, transform=transform)
#del file
#del labels
# TODO: Reduce the ram requirements by closing the files.update : DONE !!

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
#---------------------------------------------------------------------

from utils import Metrics, format_, weight_init
from models import *

# CoATNet model
model = A5([32,32],5,[2, 3, 5, 3, 5],[128, 128, 256, 256, 256],['T','C','T','C']) 

#print model summary.
from torchinfo import summary
summary(model,input_size=(BATCH_SIZE,5,32,32))

model = model.cuda() if use_cuda else model
model = model.apply(weight_init)   # weight initialisation 
optimizer = optim.Adam(model.parameters(),lr=learning_rate, weight_decay=5e-4)
criterion = nn.HuberLoss(delta=0.1,reduction="sum")   # Todo: try with `mean` also. 
scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=step_lr, gamma=0.5) 
# Todo: Decay lr s.t. metrics improve.  
print("All initialisation done.")

#------------------------------------------------------------------------------
# Training/Testing loop
#------------------------------------------------------------------------------

def run(model, dataloader, epochs):
    min_loss = 10000 # setinel value.
    for epoch in range(epochs):
        print(f"Experiment no: {run_id} Epoch: {epoch+1}")
        fname = save_path+"images/run"+str(run_id)


        #Training Loop.
        model.train()
        running_loss = 0.0
        train_metrics = [] # store the logs

        for i, data in enumerate(tqdm(dataloader["train"])):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()
            outputs = model(inputs).squeeze()  #debug.
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
        ta,tb,tc,td,te = format_(train_metrics,fname+"train"+str(epoch+1))
        print(f'Training | nmad: {round(tc,6)}, bias: {round(td,6)}, loss: {round(ta,9)}, MAE: {round(tb,6)}, outliers %: {round(te,3)}')

        #Testing Loop.
        model.eval()
        running_loss=0.0
        test_metrics = [] #store the logs

        for i,data in enumerate(tqdm(dataloader["test"])):
            inputs,labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs).squeeze()  #debug.
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            # collect the data.
            na = outputs.detach().to('cpu').numpy()
            nb = labels.detach().to('cpu').numpy()
            result = Metrics(na,nb)
            result["loss"] = loss.item()
            test_metrics.append(result)
        #generate report.
        va,vb,vc,vd,ve = format_(test_metrics,fname+"test"+str(epoch+1))
        print(f'Testing | nmad: {round(vc,6)}, bias: {round(vd,6)}, loss: {round(va,9)}, MAE: {round(vb,6)}, outliers %: {round(ve,3)}')

        #Saving the model.
        if va <= min_loss:
            print("Saving the model.")
            torch.save(model.state_dict(), save_path +"run"+str(run_id)+".pth")
            min_loss = va

#------------------------------------------------------------------------------

#----------------------------------------------------------------------------

run(model,dataloader,NUM_EPOCHS)
print("All Done!")
