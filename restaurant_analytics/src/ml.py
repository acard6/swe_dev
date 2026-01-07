import pandas as pd
import numpy as np
import os
# from tqdm import tqdm
import torch
import torch.optim as optim
import torch.nn as nn
# import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, Subset
# from torchvision import transforms
# from torchvision.utils import save_image
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.preprocessing import LabelEncoder
import random

DROPOUT = round(random.uniform(0.2,0.5),3)      
PERCENTAGE = round(random.uniform(0.73,0.87),3)

############################### opening up the necesasry file ############################
this_file = os.path.dirname( os.path.dirname(os.path.abspath(__file__)) )      # parent directory
cd = os.path.join(this_file, "data")
row_size = 1000
time = ""
start = 0
if time == "weekend":
    file_name = cd+"\\fri-sat.xlsx"
    LUT = 144     # fri-sat
    correctness = 15
    
elif time == "weekday":
    file_name = cd+"\\sun-thur.xlsx"
    LUT = 358     #sun-thur
    correctness = 10

elif time == "morning":#consider 80% training data for mornings
    file_name = cd+"\\mornings.xlsx"
    LUT = 650     #sun-thur
    correctness = 5
    start = 44
    
else:
    file_name = cd+"\\data.xlsx"
    LUT = 650     # total amount of values to look at  (some % of the total data being observed)
    correctness = 10    # how much the data should be off by

fp = os.path.join(cd, file_name)
pred_file_name = cd + "\\predictions.xlsx"
pred_file = os.path.join(cd, pred_file_name)
############################### EOL ############################


############################### global var ############################
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = "cpu"
date = None
PRELOAD = False                 # preload the model with the previous ones parameters. Used for sliding window to help reteach on new data
SAVE_MODEL_WEIGHTS = False      # save the model weights for use in future training/testing/predictions
# percentage = 0.84               # % of data to use as training

year = 2025
a = 256
b = 64  # use either b=64/c=32 or b=32/c=4
c = 32  # 64x32 for weekday. 32x4 for weekend
d = 1
############################### EOL ############################

''' when analyzing sunday-thursday data consider using the parameter of b=64 and c=32 as it appropriates closer to overall data average 
    amongst the different possible parameters used'''



class NN_model(nn.Module):
    ''' a simple neural net to train and test data on '''
    def __init__(self, input_dim=6, output_dim=1, num_hol=32,embed_dim=8, dropout=DROPOUT):
        super(NN_model, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=num_hol, embedding_dim=embed_dim)

        comb_in_dim = input_dim + embed_dim

        self.hidden1 = nn.Linear(comb_in_dim,a)
        self.hidden2 = nn.Linear(a,b)
        self.hidden3 = nn.Linear(b,c)
        self.out = nn.Linear(c,output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

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
    '''
    This function converts the data into a tensor that will be passed into a dataloader subset to be used by the model for creating predictions
    Input:  
        |->    size - batch size for the data
    Output:
        |->     TensorDataset of the (inputs, embedded encoding, target values)
    '''
    df = pd.read_excel(fp,usecols="A,C:I")
    # n = int(LUT * Percent)
    
    # Add this: ensure holiday exists and is processed
    df['holiday'] = df['holiday'].fillna('').str.lower().str.split(', ')    # fill in not-a-number/empty cells in the data
    df['holiday'] = df['holiday'].apply(lambda x: [h.strip() for h in x if h.strip()])  # strip text of any additional white space
    all_holidays = sorted(set(h for sublist in df['holiday'] for h in sublist if h))    # sort all the entries of the column as a set
    holiday_encoder = LabelEncoder().fit(all_holidays)                                  # using LabelEncoder func fit all my entries to encode
    df['holiday_encoded'] = df['holiday'].apply(lambda x: holiday_encoder.transform(x).tolist() if x else [])   # conver things to a list

    # Pad holiday lists to fixed length (optional: tune max_holidays)
    max_holidays = max(len(h) for h in df['holiday_encoded'])
    padded_holidays = [h + [0] * (max_holidays - len(h)) for h in df['holiday_encoded']]

    # Label encoder converted to a tensor 
    h_tensor = torch.tensor(padded_holidays, dtype=torch.long, device=device)

    x = df.drop(columns=["count", "holiday", "holiday_encoded"])    # taking all but the target as inputs
    x['date'] = x["date"].dt.day_of_year    # converting dates to day of the year
    global date, row_size
    date = x["date"][LUT]
    row_size = len(x)


    # normalising values based off of mean and std dev
    for column in x.columns: 
        x[column] = (x[column] - x[column].mean()) / x[column].std() 
    y = df["count"]         # target values


    # converting data to tensor
    x_tensor = torch.as_tensor(x.values, dtype=torch.float32).to(device)
    y_tensor = torch.as_tensor(y.values, dtype=torch.float32).to(device).unsqueeze(1)
    
    # converting tensors to tensordataset
    dataset = TensorDataset(x_tensor, h_tensor, y_tensor)

    return dataset     # returns tensor dataset


def dataloader_subset(loader, start, end, size=16, shuffle=False):
    '''
    takes a subset region of a TensorDataset and converts it to a dataloader to be used for either training, testing, or predicting for the model
    Inputs
        |->     Loader: the TensorDataset to be sliced
        |->     Start:  starting point within the tensor 
        |->     End:    end point within the tensor
        |->     size:   batch size for the data(default is 16)
        |->     shuffle:shuffle data based on if training or testing

    Outputs
        |->     Dataloader of that region of the TenserDataset
    '''
    # converting tensordataset to dataloader
    # print(f"start:{start}, end: {end}")
    if end > len(loader):
        end = len(loader)
    if start < 0:
        start = 0
    if start > len(loader):
        return -1
    if start == end:
        return -1
    idx = list(range(start,end))
    
    loader = Subset(loader, idx)
    # print("successfully converted data to tensor")
    return DataLoader(loader, batch_size = size,shuffle = shuffle)   # returns a dataloader from a tensordataset subset with start and end points 


def use_save():
    global PRELOAD
    PRELOAD = True
    # print("using previous pretrained model weights")

def disable_save():
    global PRELOAD
    PRELOAD = False
    # print("no longer using previous model save")

def save_weights():
    global SAVE_MODEL_WEIGHTS
    SAVE_MODEL_WEIGHTS = True
def forget_weights():
    global SAVE_MODEL_WEIGHTS
    SAVE_MODEL_WEIGHTS = False
  
def train(model,optimizer, loss_fn, train_loader, epochs=100, scheduler=None):
    '''
        train on the data and update weights
    Inputs
        |->     Model:          ML model to be used
        |->     Optimizer:      optimization func for the model 
        |->     loss_fn:        loss func for the mode
        |->     train_loader:   DataLoader to train model on
        |->     epochs:         number of runs to be made
        |->     scheduler:      model scheduler

    Outputs
        |->     avg:            model average accuracy 
        |->     predicted:      models output
    '''
    model.to(device)

    # print("Analyzing data with the model:")
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
        # if epoch%250 == 0:
        # #     print("",end=".",flush=True)
        #     print(f'Epoch {epoch}/{num_epochs}, Loss: {running_loss / len(train_loader)}')
        losses.append(running_loss / len(train_loader))

        if scheduler != None:
            scheduler.step()
    # print(f'Epoch {epoch+1}/{num_epochs}, Loss: {running_loss / len(train_loader)}')
    # print(f"{data}\n{target}")
    avg = np.average(losses)
    # print(f"Average loss on training: {avg:.2f}")
    # print(".",end="\n")\
    output = os.path.join(this_file, "weights.pth")
    if (SAVE_MODEL_WEIGHTS):
        torch.save({"model_state": model.state_dict(),}, output)
    return avg   #returns an array of losses over epochs with "epochs" elements inside as well as original model output


def test(model, loss_fn, test_loader):
    '''
        test on the data and see how the model does
    Inputs
        |->     Model:          ML model to be used
        |->     Optimizer:      optimization func for the model 
        |->     loss_fn:        loss func for the mode
        |->     train_loader:   DataLoader to train model on
        |->     epochs:         number of runs to be made
        |->     scheduler:      model scheduler

    Outputs
        |->     avgerage_loss:  model average loss accuracy 
        |->     accuracy:       models accuracy as of its current training
        |->     predicted:      models output
    '''
    # print("Testing the model on unseen data")
    model.eval()
    # predefine some variables for testing
    samples = 0
    correct = 0
    tot_loss = 0
    idx = 0
    with torch.no_grad():
        # print("predicted vs. actual  difference") 
        # print(f"predicted\ttarget\tdifference")            
        for data, holiday, target in test_loader:
            data, holiday, target = data.to(device), holiday.to(device) ,target.to(device)
            # predict values
            predicted = model(data, holiday)

            # for i in range(len(predicted)):
            #     p = int(predicted[i])
            #     a = int(target[i])
                
                # if abs(p-a) <= correctness:
                #    print(f"{p}\t\t{a}\t{abs(p-a)}")
                # print(f"{p}\t\t{a}\t{abs(p-a)}")

                # day = int(date+i+32*idx)
                # res = datetime(year,1,1) + timedelta(days=day-1)
                # res = res.strftime("%m/%d/%Y")
                # print(f"{res}: {p:<3}\t\t{a}")

            # compute loss
            loss = loss_fn(predicted, target)
            tot_loss += loss.item()
            correct += (abs(predicted - target) <= correctness).sum().item()
            samples += target.size(0)
            idx += 1
    if (len(test_loader) == 0):
            return 0,0

    average_loss = tot_loss / len(test_loader)
    if np.isnan(average_loss):
        print(tot_loss)
    accuracy = correct / samples

    # print(f'Average Loss on testing: {average_loss:.2f}')
    # print(f'Accuracy: {accuracy * 100:.2f}%')
    return average_loss, accuracy   #returns float for average loss and accuracy during this test, as well as original model output


def predict(model, pred_loader):
    '''
        makes predictions on the data
    Inputs
        |->     Model:          ML model to be used
        |->     pred_loader:    DataLoader to have the model predict on

    Outputs
        |->     fianl_pred:     models output as a list for easy viewing
    '''
    model.eval()
    # predefine some variables for testing
    pred_arr = []   # array of my prediction to be passed to other functions

    idx = 0
    with torch.no_grad():
        # print("predicting....")
        for data, holiday, _ in pred_loader:
            data, holiday = data.to(device), holiday.to(device)
            # print(data)
            predicted = model(data, holiday)
            pred_arr.append(predicted.cpu().numpy())
         
            # for i in range(len(predicted)):
            #     p = int(predicted[i])   #predicted value
            #     day = int(date+i+32*idx)
            #     res = datetime(year,1,1) + timedelta(days=day-1)
            #     res = res.strftime("%m/%d/%Y")
            #     # res = datetime.strptime(year + "-" + day, "%Y-%j").strftime("%m/%d/%Y")
            #     # print(f"{res}: {p:<3}")
            #     print(f"{p:<3}")
            # idx += 1



    final_pred = np.concatenate(pred_arr, axis=0)
    final_pred = final_pred.flatten()
    return final_pred    # returns the predicted output for the remainider of the data as a list, as well as the original model output


def plot_losses(epochs, loss_arr):
    """ plotting losses of the model over epochs"""
    print("plotting losses")
    plt.plot(range(epochs), loss_arr)
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.title("loss over time")
    plt.grid(True)
    plt.show()


def activate_model(num_epochs=1000, train_loader=None, test_loader=None, future_loader=None, percetage=PERCENTAGE, dropout=DROPOUT, test_mode=True):
    # batch_size = 32
    # print(f"b: {b}, c: {c}, file used: {time}, trainging on {percetage*100:.0f}% data, dropout{dropout}")
    print(f"file: {time}, trainging on {percetage*100:.2f}% data, dropout: {dropout}")

    # num_epochs = 1000   #for weekday data consider 2000 epochs. weekend either 1500.

    loss_fn = nn.L1Loss()
    model = NN_model(dropout=dropout)
    model.to(device)
    optimizer = optim.Adam(params=model.parameters(), lr=0.05, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10,gamma=0.95)

    older_model = None
    if PRELOAD:
        older_model = torch.load(os.path.join(this_file, "weights.pth"), weights_only=True)
        model.load_state_dict(older_model["model_state"])

    # training the model
    train_losses = train(model,optimizer, loss_fn, train_loader, num_epochs, scheduler=scheduler)

    #testing the model
    if test_mode:
        test_loss, acc = test(model, loss_fn, test_loader)
    else:
        test_loss, acc = 0,0


    # plotting losses
    # plot_losses(num_epochs, train_losses)

    # predicting with model
    prediction = predict(model, future_loader)
    return train_losses, test_loss, acc, prediction       # my outputs used for easy viewing/manipulation
    


