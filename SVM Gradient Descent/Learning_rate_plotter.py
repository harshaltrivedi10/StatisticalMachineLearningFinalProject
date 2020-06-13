import matplotlib.pyplot as plt
import json
import os

label_path = "E:\ASU\Sem 2\SML\Project\Data"
fnames=["Train_Labels_10000_HU_HIST.json", "Train_Labels_10000_FLAT_GRAY.json","Train_Labels_10000_KAZE.json"]
colors={"0.001":'r',"0.005":'yellow',"0.01":'m',"0.05":'k',"0.1":'c',"0.5":'b'}
label=["HU HIST","FLAT","KAZE",]

# data=[]
for fname in range(len(fnames)):
    plt.figure()
    with open(os.path.join(label_path,fnames[fname])) as json_file:
        # data.append(json.load(json_file)
        data = json.load(json_file)
        print()
        for i in data:
            xdata = []
            ydata = []
            print(i)
            for j in data[i]:
                xdata.append(int(j))
                ydata.append(data[i][j])
            print(xdata)
            print(ydata)

            plt.title(fnames[fname])
            plt.xlabel("1/lambda")
            plt.ylabel("Accuracy")
            plot_data = plt.plot(xdata,ydata,colors[i],label = i)
            plt.legend(loc='best')
plt.show()
plt.close()    # break
