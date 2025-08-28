import ml as ml
import numpy as np
import time
import torch
import json
from ml import LUT as n     # the number of real data points in the dataset
from ml import PERCENTAGE as PERCENTAGE
from ml import DROPOUT as DROPOUT
import random

def weighted_ensemble(predictions, weights):
    reg_preds = [torch.as_tensor(p, dtype=torch.float32) for p in predictions["fixed"]]
    win_preds = [torch.as_tensor(p, dtype=torch.float32) for p in predictions["expanding"]]
    slide_preds = [torch.as_tensor(p, dtype=torch.float32) for p in predictions["sliding"]]

    if reg_preds:
        reg_preds = torch.stack(reg_preds).mean(dim=0)
        reg_weight = weights['fixed'] * reg_preds
    else:
        reg_weight = 0

    if win_preds:
        win_preds = torch.stack(win_preds).mean(dim=0)
        win_weight = weights['fixed'] * win_preds
    else:
        win_weight = 0
    

    if slide_preds:
        sw = torch.as_tensor(weights["sliding"], dtype=torch.float32)
        sw /= sw.sum()
        sw_stack = torch.stack(slide_preds)
        sw = (sw.unsqueeze(-1) * sw_stack).sum(dim=0)
    else:
        sw = 0
    
    GOD_SAYS = reg_weight + win_weight + sw
    return GOD_SAYS.round().clamp(min=0).int()


def main():
    # values for the model
    runs = 5       # number of runs to be averaged out
    epochs = 1000   # number of epochs in a single run
    batch_size = 32 # data batch size
    run_model_in_test = True

    # preload the data once as tensors
    dataset = ml.convert_data(batch_size) 
   
   # convert data to loaders using a fixed split. (anywhere between 70/30-85/15 on trainin/testing)
    data_size = int(n* PERCENTAGE)     # the real data * seed . for splitting real data into training and testing

    run_normal = False 
    
    run_expanding_window = False
    window_runs = 3
    
    run_sliding_window = True
    sliding_window_size = 30
    FACTOR = 9/10            # 1-overlap%. how much of this data is independent from the following 

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


    # storing the output of the model to average things out at the end
    tr_l = []
    tt_l = []
    acc =  []
    pred = []

    number_of_weights = int(np.ceil(n/(sliding_window_size*FACTOR)))
    slide_weights = np.linspace(0.5, 1.0, number_of_weights)
    slide_weights = slide_weights/ slide_weights.sum()
    pred_calc = {
        "fixed":[], 
        "expanding":[], 
        "sliding":[]
    }

    weights = {
        "fixed": 0.2,
        "expanding": 0.3,
        "sliding": slide_weights.tolist()
    }

    future_loader =  ml.dataloader_subset(dataset, n, ml.row_size, batch_size)  
################################################## different ways to run the model --- timer starts
    start_time = time.time()    # timer for the runs

        ############################## normal
    if (run_normal):
        data_size = int(n* random.uniform(0.73,0.87))            
        train_loader =  ml.dataloader_subset(dataset, ml.start, data_size, batch_size, True)  
        test_loader =  ml.dataloader_subset(dataset, data_size, n, batch_size,)
        ml.use_save()
        # print("Normal run")
        if test_loader == -1:
            run_model_in_test = False
        for i in range(runs):
            dropout = round(random.uniform(0.2,0.5),3)
            print(f"starting model on test {i}....")
            train_l, test_l, accuracy, prediction = ml.activate_model(epochs, train_loader=train_loader, test_loader=test_loader, future_loader=future_loader, test_mode=run_model_in_test, dropout=dropout)
            tr_l.append(round(train_l,2))
            tt_l.append(round(test_l,2))
            acc.append(round(accuracy*100,2))
            pred.append([int(round(x,0)) for x in prediction])
            # ml_train, ml_test, pred = ml.activate_model(epochs, train_loader=train_loader, test_loader=test_loader, future_loader=future_loader)
            pred_calc["fixed"].append(prediction)
            print(f"completed run {i}\n")


            ######################## expanding

    if (run_expanding_window):
        dropout = round(random.uniform(0.2,0.5),3)    
        size = max(window_runs-1, 1)
        # print("starting expanding window")
        for i in range(window_runs):
            # ml.use_save()
            window_size = int(n*(i*(0.47)/size+0.5)) # ( (%A+%B)/(runs-1) * i + %B ) * n
            train_loader = ml.dataloader_subset(dataset, ml.start, window_size, batch_size, True)
            test_loader = ml.dataloader_subset(dataset, window_size, n, batch_size)
            if test_loader == -1:
                run_model_in_test = False
            train_l, test_l, accuracy, prediction = ml.activate_model(epochs, train_loader=train_loader, test_loader=test_loader, future_loader=future_loader, dropout=dropout, percetage=window_size/n,test_mode=run_model_in_test)

            pred_calc["expanding"].append(prediction)
            pred.append([int(round(x,0)) for x in prediction])
            tt_l.append(round(test_l,2))
            acc.append(round(accuracy*100,2))
            print(f"completed run {i}")
        print()
    
            #################### sliding

    if(run_sliding_window):
        # print("starting sliding window")
        start = ml.start
        overlap = int(sliding_window_size*FACTOR)
        ml.use_save()
        dropout = round(random.uniform(0.2,0.5),3)
        for i in range(0,n,overlap):
            end = start + sliding_window_size + i
            if (start + sliding_window_size+i) > n:
                end = n
            train_loader = ml.dataloader_subset(dataset, start+i, end, batch_size, True)
            test_loader = ml.dataloader_subset(dataset, end, n, batch_size)
            if test_loader == -1:
                run_model_in_test = False

            train_l, test_l, accuracy, prediction = ml.activate_model(epochs, train_loader=train_loader, test_loader=test_loader, future_loader=future_loader, dropout=dropout, percetage=end/n,test_mode=run_model_in_test)
            
            pred_calc["sliding"].append(prediction)
            pred.append([int(round(x,0)) for x in prediction])
            tt_l.append(round(test_l,2))
            acc.append(round(accuracy*100,2))

    end_time = time.time()
    print(f"elapsed time: {end_time-start_time:.3f} second to run model")
################################################## different ways to run the model --- timer ends


    # avg_tr = sum(tr_l)/max(len(tr_l),1)
    avg_tt = sum(tt_l)/max(len(tt_l),1)
    avg_acc = sum(acc)/max(len(acc),1)


#########  old predictiong
    # printing the entire prediction list
    old = False
    if old == False:
        pred = np.array(pred)
        l = len(pred)
        num_rows = len(pred[0]) if (l>0) else 0
        print("Predictions: ")
        for row in range(num_rows):
            val = 0
            for col in range(l):
                val += pred[col][row]
            print( round(val/l) )

        print(f"{avg_acc:.2f}")
        print(f"{avg_tt:.2f}")
    # print(f"{avg_tr:.2f}\t :avg train loss")


######### new predicting
    final_pred = weighted_ensemble(pred_calc, weights)
    # for i in (final_pred.tolist()):
    #     print(i)

if __name__ == "__main__":
    main()