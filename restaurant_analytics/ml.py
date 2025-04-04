import pandas as pd
import numpy as np
import os
# from tqdm import tqdm
import torch
import torch.optim as optim
import torch.nn as nn
# import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
# from torchvision import transforms
# from torchvision.utils import save_image
import matplotlib.pyplot as plt
from datetime import datetime, timedelta



# print(torch.__version__)

cd = os.path.dirname(os.path.abspath(__file__))
file_name = cd+"\\data.xlsx"
# file_name = cd+"\\fri-sat.xlsx"
# file_name = cd+"\\sun-thur.xlsx"
fp = os.path.join(cd, file_name)
pred_file_name = cd + "\\predictions.xlsx"
pred_file = os.path.join(cd, pred_file_name)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
date = None
year = 2025
a = 256
b = 32  # use either b=64/c=32 or b=32/c=4
c = 4
d = 1
''' when analyzing sunday-thursday data consider using the parameter of b=64 and c=32 as it appropriates closer to overall data average 
    amongst the different possible parameters used'''
print(f"b: {b}, c: {c}")
class NN_model(nn.Module):
    ''' a simple neural net to train and test data on '''
    def __init__(self, input_dim=6, output_dim=1):
        super(NN_model, self).__init__()
        self.hidden1 = nn.Linear(input_dim,a)
        self.hidden2 = nn.Linear(a,b)
        self.hidden3 = nn.Linear(b,c)
        self.hidden4 = nn.Linear(c,d)
        self.out = nn.Linear(c,output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.hidden1(x))
        x = self.relu(self.hidden2(x))
        x = self.relu(self.hidden3(x))
        # x = self.relu(self.hidden4(x))
        x = self.out(x)

        return x


def convert_data(size=16):
    '''reading data from the file'''
    df = pd.read_excel(fp,usecols="A,C:H")
    percentage = 0.85               # % of data to use as training
    a = 375                         # total amount of values to look at
    # a = 108   #fri-sat
    # a = 267   #sun-thur
    n = int(a * percentage)

    x = df.drop(columns="count")    # taking all but the target as inputs
    x['date'] = x["date"].dt.day_of_year    # converting dates to day of the year
    global date
    date = x["date"][a]

    for column in x.columns: 
        x[column] = x[column]  / x[column].abs().max() 
    y = df["count"]         # target values
    # y = np.array(y/y.max(), dtype=np.float32)    #normalizing target values
    # splitting data to train on bottom 85% and test the rest
    x_train, x_test = x.iloc[:n], x.iloc[n:a]
    y_train, y_test = y[:n], y[n:a]
    x_pred = x.iloc[a:]
    y_pred = y[a:]
    # converting dataframe to tensors and moving to proper device for max efficiency
    X_train, X_test = torch.tensor(x_train.values, dtype=torch.float32).to(device), torch.tensor(x_test.values, dtype=torch.float32).to(device)
    Y_train, Y_test = torch.as_tensor(y_train.values, dtype=torch.float32).unsqueeze(1).to(device), torch.as_tensor(y_test.values, dtype=torch.float32).unsqueeze(1).to(device)
    X_pred, Y_pred = torch.tensor(x_pred.values, dtype=torch.float32).to(device), torch.as_tensor(y_pred.values, dtype=torch.float32).unsqueeze(1).to(device)

    # converting tensor to tensordataset
    train_data = TensorDataset(X_train, Y_train)
    test_data = TensorDataset(X_test, Y_test)
    pred_data = TensorDataset(X_pred, Y_pred)

    # converting tensordataset to dataloader
    train_loader = DataLoader(train_data, batch_size=size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=size, shuffle=False)
    pred_loader = DataLoader(pred_data, batch_size=size, shuffle=False)

    print("successfully converted data to tensor\n")
    return train_loader, test_loader, pred_loader 


def future_data(size=16):
    ''' turns future data to dataloader to pass throught the 
        model and give a prediction on the expected covers'''
    # converting prediction file to dataframe
    df = pd.read_excel(pred_file,usecols="A,C:H")
    x = df.drop(columns="count")
    x['date'] = x['date'].dt.day_of_year
    for column in x.columns: 
        x[column] = x[column]  / x[column].abs().max()
    y = df["count"]

    # converting dataframe to tensor and moving to any accelerator
    x_pred = torch.tensor(x.values, dtype=torch.float32).to(device)
    y_pred = torch.tensor(y.values, dtype=torch.float32).to(device)

    # converting tensor to TensorDataSet then to DataLoader
    pred_data = TensorDataset(x_pred, y_pred)
    pred_loader = DataLoader(pred_data, batch_size=size, shuffle=False)
    print("prediction file converted to DataLoader")
    return pred_loader


def train(model,optimizer, loss_fn, train_loader, epochs=100, scheduler=None):
    '''time to train the model on the data'''
    model.to(device)

    print("Analyzing data with the model:")
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
        if epoch%250 == 0:
            # print("",end=".",flush=True)
            print(f'Epoch {epoch}/{num_epochs}, Loss: {running_loss / len(train_loader)}')
        losses.append(running_loss / len(train_loader))

        if scheduler != None:
            scheduler.step()
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {running_loss / len(train_loader)}')
    # print(f"{data}\n{target}")
    print(f"average loss on training: {np.average(losses)}\n")
    # print(".",end="\n")
    torch.save(model.state_dict(), "\\weights.pth")
    return losses   #returns an array of losses over epochs with "epochs" elements inside


def test(model, loss_fn, test_loader):
    '''time to evaluate the model on unseen data'''
    # print("Testing the model on unseen data")
    model.eval()
    # predefine some variables for testing
    samples = 0
    correct = 0
    tot_loss = 0
    with torch.no_grad():
        # print("predicted vs. actual  difference") 
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            # predict values
            predicted = model(data)
            for i in range(len(predicted)):
                p = int(predicted[i])
                a = int(target[i])
                # if abs(p-a) <= 5:
                #     print(f"{p}\t\t{a}\t{abs(p-a)}")
                # print(f"{p}\t\t{a}\t{abs(p-a)}")
            # compute loss
            loss = loss_fn(predicted, target)
            tot_loss += loss.item()

            correct += (abs(predicted - target) <= 10).sum().item()
            samples += target.size(0)

    average_loss = tot_loss / len(test_loader)
    accuracy = correct / samples

    print(f'Average Loss on testing: {average_loss:.4f}')
    print(f'Accuracy: {accuracy * 100:.2f}%')
    return average_loss, accuracy   #returns float for average loss and accuracy during this test


def predict(model, pred_loader):
    '''predict on future days'''
    model.eval()
    # predefine some variables for testing
    idx = 0
    with torch.no_grad():
        print("predicting....")
        for data, target in pred_loader:
            data = data.to(device)
            # print(data)
            predicted = model(data)
            for i in range(len(predicted)):
                p = int(predicted[i])
                day = int(date+i+32*idx)
                res = datetime(year,1,1) + timedelta(days=day-1)
                res = res.strftime("%m/%d/%Y")
                # res = datetime.strptime(year + "-" + day, "%Y-%j").strftime("%m/%d/%Y")
                print(f"{res}: {p:<3}")
            idx += 1
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
    # run_train_loss = []
    # run_test_loss = []
    # run_accuracy = []

    # inputs for the model to use to operate
    train_loader, test_loader, future_loader = convert_data(batch_size)
    num_epochs = 1000
    loss_fn = nn.L1Loss()
    # for i in range(7):
        # print(f"run {i+1}",end="",flush=True)
    model = NN_model()
    optimizer = optim.Adam(params=model.parameters(), lr=0.05)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10,gamma=0.95)

        # training the model
    train_losses = train(model,optimizer, loss_fn, train_loader, num_epochs, scheduler=scheduler)
        # run_train_loss.append(round(np.average(train_losses),2))
        #testing the model
    test_loss, acc = test(model, loss_fn, test_loader)
        # run_test_loss.append(round(test_loss,2))
        # run_accuracy.append(round((acc*100),2))
    # print("trained and tested model several times. results are in")
    # print("train losses:")
    # print(run_train_loss)
    # print("test losses")
    # print(run_test_loss)
    # print("accuracies:")
    # print(run_accuracy)

    # plotting losses
    # plot_losses(num_epochs, train_losses)

    # read data and make a prediction
    # future_loader = future_data(batch_size)
    predict(model, future_loader)




if __name__ == "__main__":
	main()