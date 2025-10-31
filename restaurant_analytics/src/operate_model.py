import ml as ml
import numpy as np
import time
import torch
import json
from ml import LUT as n     # the number of real data points in the dataset
from ml import PERCENTAGE as PERCENTAGE
from ml import DROPOUT as DROPOUT
import random


############################################################################ values setup for the model
epochs = 1000   # number of epochs in a single run
batch_size = 128 # data batch size
run_model_in_test = True        #if a dataloader subset returns -1(no data in loader) there is no need to test on said data since it is nonexistent

# preload the data once as tensors
dataset = ml.convert_data(batch_size) 

# convert data to loaders using a fixed split. (anywhere between 70/30-85/15 on trainin/testing)
data_size = int(n* PERCENTAGE)     # the real data * seed . for splitting real data into training and testing

# var for normal run
run_normal = 1
runs = 10        # number of runs to be averaged out

#var for expanding window
run_expanding_window = 0
window_runs = 8

#var for sliding window
run_sliding_window = 1
sliding_window_size = 75
FACTOR = 80/100            # 1-overlap%. how much of this data is independent from the following 

TEST_SIZE = int(0.10 * n)


# storing the output of the model to average things out at the end
tr_l = []
tt_l = []
acc =  []
pred = []

number_of_weights = int((n-ml.start)/(sliding_window_size*FACTOR))
slide_weights = np.linspace(0.5, 1.0, number_of_weights)
slide_weights = slide_weights/ slide_weights.sum()

expanding_weights = np.linspace(0.5, 1.0, window_runs)
expanding_weights = expanding_weights/ expanding_weights.sum()

pred_calc = {
    "fixed":[], 
    "expanding":[], 
    "sliding":[]
}

weights = {
    "fixed": .4,
    "expanding": expanding_weights.tolist(),
    "sliding": slide_weights.tolist()
}
future_loader =  ml.dataloader_subset(dataset, n, ml.row_size, batch_size)  
#################################################################################### end of value setup

def main():
    '''
        Small window + small overlap: many independent short-term samples → may miss long-term trends.
        Small window + large overlap: captures short-term patterns, smooth transitions. (Best for a few days ahead)

        Large window + small overlap: fewer samples, but each contains long-term info.
        Large window + large overlap: maximal context, smoothest predictions, heavier computation. (best for 2+ weeks ahead)

        large window -> 100 < points
        small window -> 100 > points

        large overlap -> 50%<
        small overlap -> 50%>
    '''
################################################## different ways to run the model --- timer starts
    start_time = time.time()    # timer for the runs

        ############################## normal
    if (run_normal):
        run_model(runs, use_prior=False, save_weights=False)

            #################### sliding


    if(run_sliding_window):
        # print("starting sliding window")
        run_model(1, save_weights=True)
        sliding_window()

            ######################## expanding

    if (run_expanding_window):
        expanding_window(save_weights=True)
    

    end_time = time.time()
    print(f"elapsed time: {end_time-start_time:.3f} second to run model")
################################################## different ways to run the model --- timer ends


    # avg_tr = sum(tr_l)/max(len(tr_l),1)
    avg_tt = sum(tt_l)/max(len(tt_l),1)
    avg_acc = sum(acc)/max(len(acc),1)


#########  old predictiong
    # printing the entire prediction list
    old = False
    if not old:
        pred_array = np.array(pred)
        l = len(pred_array)
        num_rows = len(pred_array[0]) if (l>0) else 0
        print("Predictions: ")
        for row in range(num_rows):
            val = 0
            for col in range(l):
                val += pred_array[col][row]
            print( round(val/l) )
        print(f"{avg_acc:.2f}")
        print(f"{avg_tt:.2f}")
    # print(f"{avg_tr:.2f}\t :avg train loss")


######### new predicting
    final_pred = weighted_ensemble(pred_calc, weights)
    print("using weighted ensemble, heres those predictions:")
    final_pred = final_pred.tolist()
    for i in (final_pred):
        print(i)


def run_model(lenght=runs, use_prior=False, save_weights=False, save_results=True):
    data_size = int(n* random.uniform(0.67,0.87))         
    train_loader =  ml.dataloader_subset(dataset, ml.start, data_size, batch_size, True)  
    test_loader =  ml.dataloader_subset(dataset, data_size, n, batch_size,)
    
    # print("Normal run")
    if test_loader == -1:
        global run_model_in_test
        run_model_in_test = False
    for i in range(lenght):
        dropout = round(random.uniform(0.2,0.5),3)
        # print(f"normal run on model, test {i}....")

        if use_prior and not (i==0):
            ml.use_save()

        train_l, test_l, accuracy, prediction = ml.activate_model(epochs, train_loader=train_loader, test_loader=test_loader, future_loader=future_loader, test_mode=run_model_in_test, dropout=dropout, percetage=data_size/n)
        
        if save_results:
            tr_l.append(round(train_l,2))
            tt_l.append(round(test_l,2))
            acc.append(round(accuracy*100,2))
            pred.append([int(round(x,0)) for x in prediction])
            pred_calc["fixed"].append(prediction)
        
        print(f"completed run {i+1}")
        print(f"run accuracy: {accuracy*100:.2f}\n")
        if save_weights:
            ml.save_weights()

    print()

def expanding_window(use_prior=False, save_weights=False, save_results=True):
    # ml.disable_save()
    size = max(window_runs-1, 1)
    # print("starting expanding window")
    for i in range(window_runs):
        dropout = round(random.uniform(0.2,0.5),3)    
        window_size = int(n*(i*(0.40)/size+0.5)) # ( (%A+%B)/(runs-1) * i + %B ) * n
        train_loader = ml.dataloader_subset(dataset, ml.start, window_size, batch_size, True)
        test_loader = ml.dataloader_subset(dataset, window_size, window_size+TEST_SIZE, batch_size)
        
        if test_loader == -1:
            global run_model_in_test
            run_model_in_test = False
        
        if use_prior:
            ml.use_save()
        
        train_l, test_l, accuracy, prediction = ml.activate_model(epochs, train_loader=train_loader, test_loader=test_loader, future_loader=future_loader, dropout=dropout, percetage=window_size/n,test_mode=run_model_in_test)

        if save_results:
            pred.append([int(round(x,0)) for x in prediction])
            tt_l.append(round(test_l,2))
            tr_l.append(round(train_l,2))
            acc.append(round(accuracy*100,2))
            pred_calc["expanding"].append(prediction)
        
        print(f"completed window {i+1}/{window_runs}")
        print(f"exp window accuracy: {accuracy*100:.2f}\n")
        
        if save_weights:
            ml.save_weights()
    print()

def sliding_window(size=sliding_window_size, use_prior=False, save_weights=False, save_results=True):
    start = ml.start
    overlap = int(size*FACTOR)
    # ml.disable_save()
    dropout = round(random.uniform(0.2,0.5),3)
  
    for i in range(0,n,overlap):
        end = start + sliding_window_size + i
        if (end) > n:
            break
        train_loader = ml.dataloader_subset(dataset, start+i, end, batch_size, True)
        test_loader = ml.dataloader_subset(dataset, end, end+TEST_SIZE, batch_size)
        if test_loader == -1:
            global run_model_in_test
            run_model_in_test = False
        if train_loader == -1:
            break

        if use_prior:
            ml.use_save()
        train_l, test_l, accuracy, prediction = ml.activate_model(epochs, train_loader=train_loader, test_loader=test_loader, future_loader=future_loader, dropout=dropout, percetage=end/n,test_mode=run_model_in_test)

        if save_results:        
            pred.append([int(round(x,0)) for x in prediction])
            if not (np.isnan(test_l)):
                tt_l.append(round(test_l,2))
            tr_l.append(round(train_l,2))
            acc.append(round(accuracy*100,2))
        
        print(f"sliding window accuracy: {accuracy*100:.2f}\n")
        # print(f"sliding window loss: {test_l:.2f}\n")
        if save_weights:
            ml.save_weights()

    if save_results:
        pred_calc["sliding"].append(prediction)


def weighted_ensemble(predictions, weights):
    reg_preds = [torch.as_tensor(p, dtype=torch.float32) for p in predictions["fixed"]]
    win_preds = [torch.as_tensor(p, dtype=torch.float32) for p in predictions["expanding"]]
    slide_preds = [torch.as_tensor(p, dtype=torch.float32) for p in predictions["sliding"]]

    normalizing = []

    if reg_preds:
        reg_preds = torch.stack(reg_preds).mean(dim=0)
        # reg_weight = weights['fixed'] * reg_preds
        normalizing.append((reg_preds, weights.get('fixed', 1.0)))

    if win_preds:
        ew = torch.as_tensor(weights["expanding"], dtype=torch.float32)
        ew /= ew.sum()
        ew_stack = torch.stack(win_preds)
        ew = (ew.unsqueeze(-1)*ew_stack).sum(dim=0)
        normalizing.append((ew, 1.0))
    

    if slide_preds:
        sw = torch.as_tensor(weights["sliding"], dtype=torch.float32)
        sw /= sw.sum()
        sw_stack = torch.stack(slide_preds)
        sw = (sw.unsqueeze(-1) * sw_stack).sum(dim=0)
        normalizing.append((sw, 1.0))
    
    if normalizing:
        tot_weight = sum(w for _,w in normalizing)
        GOD_SAYS = sum(pred * (w/tot_weight) for pred, w in normalizing)
    else:
        GOD_SAYS = None
    return GOD_SAYS.round().clamp(min=0).int() if GOD_SAYS is not None else None

if __name__ == "__main__":
    main()