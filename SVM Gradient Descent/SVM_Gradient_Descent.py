import csv
import os
import numpy as np
import json

#reference taken from implementation on site  https://github.com/llSourcell/Classifying_Data_Using_a_Support_Vector_Machine/blob/master/support_vector_machine_lesson.ipynb

def SVM(x, y,learning_rate,epochs):
    print(learning_rate,epochs)
    w = np.zeros(len(x[0]))
    b = 0
    # learning_rate = 0.001


    # epochs = 100000
    # epochs = 5000
    errors = []
    epoch_list = []

    for epoch in range(1, epochs):
        # print(epoch)
        error_count = 0
        for i in range(len(x)):
            if (y[i] * (np.dot(w, x[i]) + b) < 1):
                w = w - learning_rate * (((1 / epoch) * w) - (y[i] * x[i]))
                b = b - learning_rate * (-1 * y[i])
                error_count += 1
            else:
                w = w - learning_rate * ((1 / epoch) * w)

            # w=w_new
        errors.append(error_count)
        epoch_list.append(epoch)

    return (w, b)


def test(w, x, y, b):
    error_count = 0
    ctr = 0
    for i in range(len(x)):
        ctr += 1
        # print(np.dot(w, x[i]) + b, y[i])
        if (np.sign(y[i]) != np.sign(np.dot(w, x[i]) + b)):
            error_count += 1

        # print(error_count)
    return (ctr - error_count) / ctr

label_path = "E:\ASU\Sem 2\SML\Project\Data"
fnames=["Train_Labels_10000_HU_HIST.csv","Train_Labels_10000_FLAT_GRAY.csv","Train_Labels_10000_KAZE.csv"]
data = []
# fname="Train_Labels_10000_HU_HIST.csv"
# fname1="Train_Labels_10000_BLUR_HU_HIST.csv"
for fname in fnames:

    data=[]
    with open(os.path.join(label_path, fname)) as csv_file:
        csv_reader = csv.reader(csv_file)
        for row in csv_reader:
            for i in range(len(row)):
                if (i > 0):
                    row[i] = float(row[i])
            data.append(row)

    X = np.zeros([len(data), len(data[0]) - 2])
    Y = np.zeros(len(data))
    for i in range(len(data)):
        X[i] = np.asarray(data[i][1:-1])
        Y[i] = data[i][-1]
        # print(X[i],Y[i])


    Xtrain = X[:9000]
    Xtest = X[9000:]
    Ytrain = Y[:9000]
    Ytest = Y[9000:]


    epochs_list=[100,200,500,1000,5000,10000]
    learning_rate_list=[0.001,0.005,0.01,0.05,0.1,0.5]

    results={}

    ctr=0
    for i in learning_rate_list:
        results[i]={}
        for j in epochs_list:
            ctr+=1
            print(fname)
            w_b = SVM(Xtrain, Ytrain,i,j)
            results[i][j] = test(w_b[0], Xtest, Ytest, w_b[1])
            # results[i][j] = test(w_b[0], Xtrain, Ytrain, w_b[1])
            print(results[i][j])
            # if(ctr==2):
            # break
        # break
    print()
    # print(results)

    temp_fname = fname.split(".")[0]
    store_fname = temp_fname+"_TRAIN.json"
    with open(os.path.join(label_path,store_fname),"w") as json_file:
        json.dump(results,json_file)
    # break