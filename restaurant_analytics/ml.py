import pandas as pd
import torch
import numpy as np
import os
from tqdm import tqdm
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
from torchvision.utils import save_image



print(torch.__version__)

cd = os.path.dirname(os.path.abspath(__file__))
file_name = cd+"\\data.xlsx"
fp = os.path.join(cd, file_name)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class NN_model(nn.Module):
    ''' a simple neural net to train and test data on '''
    def __init__(self, input_dim=6, output_dim=1):
        super(NN_model, self).__init__()

        self.hiden1 = nn.Linear(input_dim,50)
        self.act1 = nn.Sigmoid()
        self.hidden2 = nn.Linear(50,10)
        self.act2 = nn.Softmax(dim=1)
        self.out = nn.Linear(10,output_dim)
        self.out_act = nn.Sigmoid()

    def forward(self, x):
        x = self.hiden1(x)
        x = self.act1(x)
        x = self.hidden2(x)
        x = self.act2(x)
        x = self.out(x)
        x = self.out_act(x)

        return x
    
#reading data from the file
df = pd.read_excel(fp,usecols="A,C:H")
n = int(len(df) * 0.85)

x = df.drop(columns="count")    # taking all but the target as inputs
x['date'] = x["date"].dt.day_of_year    # converting dates to day of the year
y = df["count"]         # target values
y = np.array(y/y.max(), dtype=np.float32)    #normalizing target values

# splitting data to train on bottom 85% and test the rest
x_train, x_test = x.iloc[:n], x.iloc[n:]
y_train, y_test = y[:n], y[n:]

# converting dataframe to tensors and moving to proper device for max efficiency
X_train, X_test = torch.tensor(x_train.values, dtype=torch.float32).to(device), torch.tensor(x_test.values, dtype=torch.float32).to(device)
Y_train, Y_test = torch.as_tensor(y_train).unsqueeze(1).to(device), torch.as_tensor(y_test).unsqueeze(1).to(device)


# converting tensor to tensordataset
train_data = TensorDataset(X_train, Y_train)
test_data = TensorDataset(X_test, Y_test)

# converting tensordataset to dataloader
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

print("successfully converted data to tensor")
print("now starting to analyze data using simple model with Adam")


#################### time to train the model on the data

model = NN_model(input_dim=6, output_dim=1)
model.to(device)

# simple set up of variables
optimizer = optim.Adam(params=model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()
num_epochs = 100

for epoch in range(num_epochs):
    model.train()
    running_loss = 0
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        # prediction on batch data
        predicted = model(data)  
        # compute loss and gradient
        loss = loss_fn(predicted, target)   
        loss.backward()

        optimizer.step()
        running_loss += loss.item()
    if epoch%5 == 0:
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {running_loss / len(train_loader)}')
# print(f"{data}\n{target}")
######################## time to evaluate the model on unseen data

model.eval()
# predefine some variables for testing
samples = 0
correct = 0
tot_loss = 0
with torch.no_grad():
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        # print(f"input size{data.shape}\tlabel size{target.shape}")
        # predict values
        predicted = model(data)
        print("predicted vs. actual")
        for i in range(len(predicted)):
            print(f"{int(predicted[i]*265)},\t\t{int(target[i]*265)}")
        # compute loss
        loss = loss_fn(predicted, target)
        tot_loss += loss.item()

        correct += (predicted == target).sum().item()
        samples += target.size(0)

average_loss = tot_loss / len(test_loader)
accuracy = correct / samples

print(f'Average Loss: {average_loss:.4f}')
print(f'Accuracy: {accuracy * 100:.2f}%')