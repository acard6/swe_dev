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
import matplotlib.pyplot as plt



print(torch.__version__)

cd = os.path.dirname(os.path.abspath(__file__))
file_name = cd+"\\data.xlsx"
fp = os.path.join(cd, file_name)
pred_file_name = cd + "\\predictions.xlsx"
pred_file = os.path.join(cd, pred_file_name)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class NN_model(nn.Module):
    ''' a simple neural net to train and test data on '''
    def __init__(self, input_dim=6, output_dim=1):
        super(NN_model, self).__init__()

        self.hiden1 = nn.Linear(input_dim,50)
        self.act1 = nn.Softmax(dim=1)
        self.hidden2 = nn.Linear(50,50)
        self.act2 = nn.Sigmoid()
        self.out = nn.Linear(50,output_dim)
        self.out_act = nn.Sigmoid()

    def forward(self, x):
        x = self.hiden1(x)
        x = self.act1(x)
        x = self.hidden2(x)
        x = self.act2(x)
        x = self.out(x)
        x = self.out_act(x)

        return x


def convert_data(size=16):
    '''reading data from the file'''
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
    train_loader = DataLoader(train_data, batch_size=size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=size, shuffle=False)

    print("successfully converted data to tensor")
    print("now starting to analyze data using simple model with Adam")
    return train_loader, test_loader 


def future_data(size=16):
    ''' turns future data to dataloader to pass throught the 
        model and give a prediction on the expected covers'''
    # converting prediction file to dataframe
    df = pd.read_excel(pred_file,usecols="A,C:H")
    x = df.drop(columns="count")
    x['date'] = x['date'].dt.day_of_year
    y = df["count"]

    # converting dataframe to tensor and moving to any accelerator
    x_pred = torch.tensor(x.values, dtype=torch.float32).to(device)
    y_pred = torch.tensor(y.values, dtype=torch.float32).to(device)

    # converting tensor to TensorDataSet then to DataLoader
    pred_data = TensorDataset(x_pred, y_pred)
    pred_loader = DataLoader(pred_data, batch_size=size, shuffle=False)
    print("prediction file converted to DataLoader")
    return pred_loader


def train(model,optimizer, loss_fn, train_loader, epochs=100):
    '''time to train the model on the data'''
    model.to(device)

    # simple set up of variables
    num_epochs = epochs
    losses = []

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
        if epoch%50 == 0:
            print(f'Epoch {epoch}/{num_epochs}, Loss: {running_loss / len(train_loader)}')
        losses.append(running_loss / len(train_loader))
    # print(f"{data}\n{target}")

    torch.save(model.state_dict(), "\\weights.pth")
    return losses


def test(model, loss_fn, test_loader):
    '''time to evaluate the model on unseen data'''
    model.eval()
    # predefine some variables for testing
    samples = 0
    correct = 0
    tot_loss = 0
    with torch.no_grad():
        print("predicted vs. actual  difference")
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            # print(f"input size{data.shape}\tlabel size{target.shape}")
            # predict values
            predicted = model(data)
            for i in range(len(predicted)):
                p = int(predicted[i]*265)
                a = int(target[i]*265)
                print(f"{p}\t\t{a}\t{abs(p-a)}")
            # compute loss
            loss = loss_fn(predicted, target)
            tot_loss += loss.item()

            correct += (abs(predicted - target) <= 0.015).sum().item()
            samples += target.size(0)

    average_loss = tot_loss / len(test_loader)
    accuracy = correct / samples

    print(f'Average Loss: {average_loss:.4f}')
    print(f'Accuracy: {accuracy * 100:.2f}%')


def predict(model, pred_loader):
    '''predict on future days'''
    model.eval()
    # predefine some variables for testing
    with torch.no_grad():
        print("predicting....")
        for data, taget in pred_loader:
            data = data.to(device)
            print(data)
            predicted = model(data)
            for i in range(len(predicted)):
                p = int(predicted[i]*265)
                print(f"{p}")
            # compute loss


def plot_losses(epochs, loss_arr):
    """ plotting losses of the model over epochs"""
    print("plotting losses")
    plt.plot(range(epochs), loss_arr)
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.title("loss over time")
    plt.grid(True)
    plt.show()


def main():
    batch_size = 32
    # inputs for the model to use to operate
    train_loader, test_loader = convert_data(batch_size)
    num_epochs = 500
    model = NN_model()
    optimizer = optim.Adam(params=model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()

    # training the model
    losses = train(model,optimizer, loss_fn, train_loader, num_epochs)

    #testing the model
    test(model, loss_fn, test_loader)

    # plotting losses
    plot_losses(num_epochs, losses)

    # read data and make a prediction
    future_loader = future_data(batch_size)
    predict(model, future_loader)




if __name__ == "__main__":
	main()