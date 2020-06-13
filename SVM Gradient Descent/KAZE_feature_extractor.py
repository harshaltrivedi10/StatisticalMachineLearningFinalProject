import cv2
import numpy as np
import csv
import random
import os

label_path = "E:\ASU\Sem 2\SML\Project\Data"

def save_to_csv(data,fname):

    #Saving features in CSV
    print(len(data))
    with open(os.path.join(label_path, fname),"w",newline="") as csv_file:
        csv_writer = csv.writer(csv_file,delimiter=",")
        for row in data:
            if(int(row[-1])==0):
                row[-1]=-1

            csv_writer.writerow(row)


def feature_to_list(features):
    # pass
    #transforming feature space to list for storing in csv
    # print(features)
    data=[]
    with open(os.path.join(label_path, "Train_Labels.csv")) as csv_file:
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

    # save_to_csv(data,"Train_Labels_100_KAZE.csv")
    print(data)
    # save_to_csv(data, "Train_Labels_5000_KAZE.csv")
    save_to_csv(data, "Train_Labels_10000_KAZE.csv")


def extract_features(vector_size=32):
    path = "E:\ASU\Sem 2\SML\Project\Data\Train"                #change path when running on full data
    # path1 = "E:\ASU\Sem 2\SML\Project\Data\Train_cropped_32_100"    #change path when running on full data
    # path1 = "E:\ASU\Sem 2\SML\Project\Data\Train_cropped_32_5000"
    path1 = "E:\ASU\Sem 2\SML\Project\Data\Train_cropped_32_10000"


    # Extract features
    features={}
    for f in os.listdir(path1):
        # print(f)
        image = cv2.imread(os.path.join(path1, f), -1)
        try:

            alg = cv2.KAZE_create()
            # Finding image keypoints
            kps = alg.detect(image)
            # for i in kps:
            #     print("KPS: ", i.response)
            kps = sorted(kps, key=lambda x: -x.response)[:vector_size]
            kps, dsc = alg.compute(image, kps)
            dsc = dsc.flatten()
            size = vector_size * 64
            if(len(dsc<size)):
                dsc = np.concatenate((dsc,np.zeros(size-len(dsc))))
            # print(kps)
            # print(dsc)
            # print()

        except AttributeError as e:
            dsc = np.zeros(vector_size * 64)

        features[f] = dsc
    feature_to_list(features)


extract_features()