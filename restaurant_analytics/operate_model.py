import ml as ml
import numpy as np
import time
import pickle
import json


def main():
    n = 10

    tr_l = []
    tt_l = []
    acc =  []
    pred = []

    start_time = time.time()
    for i in range(n):
        print(f"starting model on test {i}....")
        train_l, test_l, accuracy, prediction = ml.activate_model(32,1000)
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