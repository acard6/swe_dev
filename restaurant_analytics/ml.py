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
from sklearn.preprocessing import LabelEncoder



# print(torch.__version__)

cd = os.path.dirname(os.path.abspath(__file__))
time = ""
start = 0
if time == "weekend":
    file_name = cd+"\\fri-sat.xlsx"
    LUT = 126     # fri-sat
    correctness = 15
    
elif time == "weekday":
    file_name = cd+"\\sun-thur.xlsx"
    LUT = 315     #sun-thur
    correctness = 10

elif time == "morning":#consider 80% training data for mornings
    file_name = cd+"\\mornings.xlsx"
    LUT = 447     #sun-thur
    correctness = 5
    start = 44
    
else:
    file_name = cd+"\\data.xlsx"
    LUT = 447     # total amount of values to look at
    correctness = 10    # how much the data should be off by
    
fp = os.path.join(cd, file_name)
pred_file_name = cd + "\\predictions.xlsx"
pred_file = os.path.join(cd, pred_file_name)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
date = None
percentage = 0.85               # % of data to use as training
year = 2025
a = 256
b = 64  # use either b=64/c=32 or b=32/c=4
c = 32  # 64x32 for weekday. 32x4 for weekend
d = 1


''' when analyzing sunday-thursday data consider using the parameter of b=64 and c=32 as it appropriates closer to overall data average 
    amongst the different possible parameters used'''
print(f"b: {b}, c: {c}, file used: {time}, trainging on {percentage*100:.0f}% data")


class NN_model(nn.Module):
    ''' a simple neural net to train and test data on '''
    def __init__(self, input_dim=6, output_dim=1, num_hol=30,embed_dim=8):
        super(NN_model, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=num_hol, embedding_dim=embed_dim)

        comb_in_dim = input_dim + embed_dim

        self.hidden1 = nn.Linear(comb_in_dim,a)
        self.hidden2 = nn.Linear(a,b)
        self.hidden3 = nn.Linear(b,c)
        self.out = nn.Linear(c,output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x, holiday_id):
        hol_embed = self.embedding(holiday_id)
        hol_vector = hol_embed.mean(dim=1)

        x = torch.cat([x,hol_vector], dim=1)

        x = self.relu(self.hidden1(x))
        x = self.relu(self.hidden2(x))
        
        x = self.dropout(self.relu(self.hidden3(x)))
        # x = self.relu(self.hidden4(x))
        x = self.out(x)

        return x


def convert_data(size=16):
    '''reading data from the file'''
    df = pd.read_excel(fp,usecols="A,C:I")
    n = int(LUT * percentage)
    
    # Add this: ensure holiday exists and is processed
    df['holiday'] = df['holiday'].fillna('').str.lower().str.split(', ')
    df['holiday'] = df['holiday'].apply(lambda x: [h.strip() for h in x if h.strip()])
    all_holidays = sorted(set(h for sublist in df['holiday'] for h in sublist if h))
    holiday_encoder = LabelEncoder().fit(all_holidays)
    df['holiday_encoded'] = df['holiday'].apply(lambda x: holiday_encoder.transform(x).tolist() if x else [])

    # Pad holiday lists to fixed length (optional: tune max_holidays)
    max_holidays = max(len(h) for h in df['holiday_encoded'])
    padded_holidays = [h + [0] * (max_holidays - len(h)) for h in df['holiday_encoded']]
    holiday_tensor = torch.tensor(padded_holidays, dtype=torch.long)    

    x = df.drop(columns=["count", "holiday", "holiday_encoded"])    # taking all but the target as inputs
    x['date'] = x["date"].dt.day_of_year    # converting dates to day of the year
    global date
    date = x["date"][LUT]

    for column in x.columns: 
        x[column] = (x[column] - x[column].mean()) / x[column].std() 
    y = df["count"]         # target values

    # splitting data to train on bottom 85% and test the rest
    x_train, x_test, x_pred = x.iloc[start:n], x.iloc[n:LUT], x.iloc[LUT:]
    y_train, y_test, y_pred = y[start:n], y[n:LUT], y[LUT:]
    h_train, h_test, h_pred = holiday_tensor[start:n], holiday_tensor[n:LUT], holiday_tensor[LUT:]
    # converting dataframe to tensors and moving to proper device for max efficiency
    X_train, X_test = torch.tensor(x_train.values, dtype=torch.float32).to(device), torch.tensor(x_test.values, dtype=torch.float32).to(device)
    Y_train, Y_test = torch.as_tensor(y_train.values, dtype=torch.float32).unsqueeze(1).to(device), torch.as_tensor(y_test.values, dtype=torch.float32).unsqueeze(1).to(device)
    X_pred, Y_pred = torch.tensor(x_pred.values, dtype=torch.float32).to(device), torch.as_tensor(y_pred.values, dtype=torch.float32).unsqueeze(1).to(device)
    H_train, H_test, H_pred = h_train.to(device), h_test.to(device), h_pred.to(device)

    # converting tensor to tensordataset
    train_data = TensorDataset(X_train, H_train, Y_train)
    test_data = TensorDataset(X_test, H_test, Y_test)
    pred_data = TensorDataset(X_pred, H_pred, Y_pred)

    # converting tensordataset to dataloader
    train_loader = DataLoader(train_data, batch_size=size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=size, shuffle=False)
    pred_loader = DataLoader(pred_data, batch_size=size, shuffle=False)

    print("successfully converted data to tensor")
    return train_loader, test_loader, pred_loader 


def future_data(size=16):
    ''' This function is no longer needed this is implemented in the convert_data function
        turns future data to dataloader to pass throught the 
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
        for data, holiday ,target,  in train_loader:
            data, holiday, target = data.to(device), holiday.to(device), target.to(device)
            optimizer.zero_grad()
            # prediction on batch data
            predicted = model(data, holiday)  
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
    print(f"Average loss on training: {np.average(losses):.2f}")
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
        # print(f"predicted\ttarget\tdifference")            
        for data, holiday, target in test_loader:
            data, holiday, target = data.to(device), holiday.to(device) ,target.to(device)
            # predict values
            predicted = model(data, holiday)

            for i in range(len(predicted)):
                p = int(predicted[i])
                a = int(target[i])
                # if abs(p-a) <= correctness:
                #     print(f"{p}\t\t{a}\t{abs(p-a)}")
                # print(f"{p}\t\t{a}\t{abs(p-a)}")
            # compute loss
            loss = loss_fn(predicted, target)
            tot_loss += loss.item()
            correct += (abs(predicted - target) <= correctness).sum().item()
            samples += target.size(0)

    average_loss = tot_loss / len(test_loader)
    accuracy = correct / samples

    print(f'Average Loss on testing: {average_loss:.2f}')
    print(f'Accuracy: {accuracy * 100:.2f}%')
    return average_loss, accuracy   #returns float for average loss and accuracy during this test


def predict(model, pred_loader):
    '''predict on future days'''
    model.eval()
    # predefine some variables for testing
    idx = 0
    with torch.no_grad():
        print("predicting....")
        for data, holiday, _ in pred_loader:
            data, holiday = data.to(device), holiday.to(device)
            # print(data)
            predicted = model(data, holiday)
            for i in range(len(predicted)):
                p = int(predicted[i])
                day = int(date+i+32*idx)
                res = datetime(year,1,1) + timedelta(days=day-1)
                res = res.strftime("%m/%d/%Y")
                # res = datetime.strptime(year + "-" + day, "%Y-%j").strftime("%m/%d/%Y")
                # print(f"{res}: {p:<3}")
                print(f"{p:<3}")

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

    # inputs for the model to use to operate
    train_loader, test_loader, future_loader = convert_data(batch_size)

    num_epochs = 1000   #for weekday data consider 2000 epochs. weekend either 1500.
    loss_fn = nn.L1Loss()
    model = NN_model()
    optimizer = optim.Adam(params=model.parameters(), lr=0.05, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10,gamma=0.95)

    # training the model
    train_losses = train(model,optimizer, loss_fn, train_loader, num_epochs, scheduler=scheduler)
    #testing the model
    test_loss, acc = test(model, loss_fn, test_loader)


    # plotting losses
    # plot_losses(num_epochs, train_losses)

    # predicting with model
    predict(model, future_loader)




if __name__ == "__main__":
	main()