import ml as ml
import numpy as np
import time
import pickle
import json


def main():
    # values for the model
    n = 10       # number of runs to be averaged out
    epochs = 1000   # number of epochs in a single run
    batch_size = 32 # data batch size

    # storing the output of the model to average things out at the end
    tr_l = []
    tt_l = []
    acc =  []
    pred = []

    start_time = time.time()    # timer

    train_loader, test_loader, future_loader = ml.convert_data(batch_size)  #preload the data once
    for i in range(n):
        print(f"starting model on test {i}....")
        train_l, test_l, accuracy, prediction = ml.activate_model(batch_size,epochs, train_loader=train_loader, test_loader=test_loader, future_loader=future_loader)
        tr_l.append(round(train_l,2))
        tt_l.append(round(test_l,2))
        acc.append(round(accuracy*100,2))
        pred.append([int(round(x,0)) for x in prediction])
        print(f"completed run {i}")
    end_time = time.time()
    print(f"elapsed time: {end_time-start_time:.3f} second for {n} runs")

    avg_tr = sum(tr_l)/len(tr_l)
    avg_tt = sum(tt_l)/len(tt_l)
    avg_acc = sum(acc)/len(acc)

    # printing the entire list
    pred = np.array(pred)
    l = len(pred)
    print("Predictions: ")

    for row in range(len(pred[0])):
        val = 0
        for col in range(l):
            # print(pred[col][row],end='\t')
            val += pred[col][row]
        print( round(val/l) )

    # print(f"{avg_acc:.2f}\t :avg accuracy")
    print(f"{avg_acc:.2f}")
    # print(f"{avg_tt:.2f}\t :avg test loss")
    print(f"{avg_tt:.2f}")
    # print(f"{avg_tr:.2f}\t :avg train loss")



if __name__ == "__main__":
    main()