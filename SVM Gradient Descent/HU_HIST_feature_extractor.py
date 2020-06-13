import cv2
import numpy as np
import csv
import os
import mahotas

path = "E:\ASU\Sem 2\SML\Project\Data\Train"  # change path when running on full data
# path1 = "E:\ASU\Sem 2\SML\Project\Data\Train_cropped_32"  # change path when running on full data

label_path = "E:\ASU\Sem 2\SML\Project\Data"
path1 = "E:\ASU\Sem 2\SML\Project\Data\Train_cropped_32_10000"

def save_to_csv(data,fname):

    #Saving features in CSV

    with open(os.path.join(label_path, fname),"w",newline="") as csv_file:      #label_path is the absolute location excluding the filename where the final csv will be stored
        csv_writer = csv.writer(csv_file,delimiter=",")
        for row in data:
            if(int(row[-1])==0):
                row[-1]=-1

            csv_writer.writerow(row)


def feature_to_list(features):
    # pass
    #transforming feature space to list for storing in csv

    data=[]
    with open(os.path.join(label_path, "Train_Labels.csv")) as csv_file:    #label path is the absolute address excluding the file name where the label information of the training data set is stored
        csv_reader = csv.reader(csv_file,delimiter=",")
        for i in features.keys():
            csv_file.seek(0)
            key=i.split(".")[0]
            temp=[]
            for row in csv_reader:
                if(key in row[0]):
                    temp=[row[0]]
                    # for j in features[i]:
                    #     temp.append(j)
                    temp.extend(features[i])
                    temp.append(row[1])
                    data.append(temp)
                    break

    # for row in data:
    #     print(row)

    save_to_csv(data,"Train_Labels_10000_HU_HIST_BIN_1.csv")

def extract_features_1():

    # Extract features
    features={}
    for f in os.listdir(path1):

        # image = cv2.imread(os.path.join(path1, f), -1)              #path1 is the absolute address of the location where the cropped images are stored
        # image = cv2.imread(os.path.join(path1, f), 0)
        img = cv2.imread(os.path.join(path1, f), 0)
        (threshold, bin_img) = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        HU = cv2.HuMoments(cv2.moments(bin_img)).flatten()

        hist = cv2.calcHist(bin_img, [0], None, [2], [0, 256])

        hist = cv2.normalize(hist, hist)
        hist = hist.flatten()

        features[f]=[]
        features[f].extend(HU)
        features[f].extend(hist)
        # print(features[f])


    feature_to_list(features)




extract_features_1()