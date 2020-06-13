import matplotlib.pyplot as plt
import json
import os

label_path = "E:\ASU\Sem 2\SML\Project\Data"
fnames=["Train_Labels_10000_HU_HIST.json", "Train_Labels_10000_FLAT_GRAY.json","Train_Labels_10000_KAZE.json"]
# colors={"0.001":'r',"0.005":'yellow',"0.01":'m',"0.05":'k',"0.1":'c',"0.5":'b'}
colors={0:'r', 1:'g', 2:'b'}
label=["HU HIST","FLAT","KAZE",]
data=[]
for fname in fnames:

    with open(os.path.join(label_path,fname)) as json_file:
        data.append(json.load(json_file))



print()
for j in data[0]:
    plt.figure()
    print(j)
    for i in range(len(data)):
        print(i)
        xdata = []
        ydata = []
        for k in data[i][j]:

            xdata.append(k)
            ydata.append(data[i][j][k])
        print(xdata)
        print(ydata)

        plt.title("Test Accuracy for Learning rate = " + str(j))
        plt.xlabel("1/lambda")
        plt.ylabel("Accuracy")
        plot_data = plt.plot(xdata,ydata,colors[i],label = label[i])
        plt.legend(loc='best')
        # plt.draw()
plt.show()
# plt.close()    # break
