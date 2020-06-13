from PIL import Image
import cv2
import os

def crop():
    #create empty folder in which the cropped images to be saved
    path = "E:\ASU\Sem 2\SML\Project\Data\Train"                #change path when running on full data
    path1 = "E:\ASU\Sem 2\SML\Project\Data\Train_cropped_32"    #change path when running on full data

    ctr=0
    number_of_instances = 10000
    for filename in os.listdir(path):
        print(ctr)
        if(ctr<number_of_instances):                     #remove line when running on full data
            # print(ctr)                  #remove line when running on full data
            img1 = cv2.imread(os.path.join(path, filename),-1)
            img2 = img1[31:63,31:63]

            cv2.imwrite(os.path.join(path1 + "_" +str(number_of_instances),filename + "32_cropped.tif"),img2)

            ctr+=1                      #remove line when running on full data
        else:                           #remove line when running on full data
            break                       #remove line when running on full data
        # cv2.waitKey(0)

crop()