"""

references taken from:
    1. Mathieu's log (September 19th, 2010) Retrieved April 1, 2019, from: https://web.archive.org/web/20140429090836/http://www.mblondel.org/journal/2010/09/19/support-vector-machines-in-python/
    2. Mathieu Blondel (September 2010) Retrieved April 1, 2019, from: https://gist.github.com/mblondel/586753
    3. Sentdex (May 13th 2016) Retrieved April 1, 2019, from: https://www.youtube.com/channel/UCfzlCWGWYyIQ0aLC5w48gBQ 


Author-Vishal Kumar
ASU ID - 1215200480
"""

import numpy as np
import cvxopt as co
from numpy import linalg
import pandas as pd

#reading data

colnames=['id']
for i in range(39):
    colnames.append('feat_'+str(i+1))
complete_data_x = pd.read_csv('final_features_32_latest.csv',names=colnames, header=None)
complete_data_x = complete_data_x[:10000]
complete_data_x['id'] = complete_data_x['id'].astype(str).str[:-18]
keep_y = list(complete_data_x['id'])

complete_data_y = pd.read_csv('train_labels.csv')
complete_data_y = complete_data_y.loc[complete_data_y['id'].isin(keep_y)]

complete_data_x = complete_data_x.sort_values(by=['id'])
complete_data_y = complete_data_y.sort_values(by=['id'])

complete_data_y.loc[complete_data_y['label'] == 0 , 'label'] = -1              

co.solvers.options['show_progress'] = False
co.solvers.options['maxiters'] = 200 


#kernel Class
class kernel():
    
    def linear_kernel(self, x1, x2):
        return np.dot(x1, x2)

    def polynomial_kernel_3(self, x, y):
        return (1 + np.dot(x, y)) ** 3

    def polynomial_kernel_2(self, x, y):
        return (1 + np.dot(x, y)) ** 2   

    def gaussian_kernel(self, x, y, sigma=5.0):
        return np.exp(-linalg.norm(x-y)**2 / (2 * (sigma ** 2)))
    
#SVM Class
class svm():
    def __init__(self, kernel_name = kernel().linear_kernel, c=0):
        self.c = c
        self.kernel_name = kernel_name
        
    def train_model(self, train_x, train_y):
        #calculate parameters rquired for cvoxpt libraray to work
        
        no_samples, no_features = train_x.shape
        kernel_matrix = np.zeros((no_samples, no_samples))
        for index in range(no_samples):
            for second_index in range(no_samples):
                kernel_matrix[index, second_index] = self.kernel_name(train_x[index],train_x[second_index])
                
        P = co.matrix(np.outer(train_y,train_y) * kernel_matrix)
        q = co.matrix(np.ones(no_samples) * -1)
        A = co.matrix(train_y, (1,no_samples), 'd')
        #A=A.astype(double)
        b = co.matrix(0.0)
    
        if self.c == 0:
            G = co.matrix(np.diag(np.ones(no_samples) * -1))
            h = co.matrix(np.zeros(no_samples))
        else:
            tmp1 = np.diag(np.ones(no_samples) * -1)
            tmp2 = np.identity(no_samples)
            G = co.matrix(np.vstack((tmp1, tmp2)))
            tmp1 = np.zeros(no_samples)
            tmp2 = np.ones(no_samples) * self.c
            h = co.matrix(np.hstack((tmp1, tmp2)))
            
        result = co.solvers.qp(P, q, G, h, A, b)
        l_multipliers = np.ravel(result['x'])
        
        support_vectors = l_multipliers > 1e-05
        index_sup_vec = np.arange(len(l_multipliers))[support_vectors]
        self.l_multipliers = l_multipliers[support_vectors]
        self.support_vectors_x = train_x[support_vectors]
        self.support_vectors_y = train_y[support_vectors]
        
        self.b = 0
        for n in range(len(self.l_multipliers)):
            self.b += self.support_vectors_y[n]
            self.b -= np.sum(self.l_multipliers * self.support_vectors_y * kernel_matrix[index_sup_vec[n],support_vectors])
        self.b /= len(self.l_multipliers)
            
        # Weight vector
        if self.kernel_name == kernel().linear_kernel:
            self.w = np.zeros(no_features)
            for n in range(len(self.l_multipliers)):
                self.w += self.l_multipliers[n] * self.support_vectors_y[n] * self.support_vectors[n]
        else:
            self.w = None
        
    def predict(self, features):
        if self.w is not None:
            return np.sign(np.dot(features, self.w) + self.b)
        else:
            y_predict = 0
            for l_multipliers, support_vectors_y, support_vectors in zip(self.l_multipliers, self.support_vectors_y, self.support_vectors_x):
                y_predict += l_multipliers * support_vectors_y * self.kernel_name(features, support_vectors)
            
            return np.sign(y_predict + self.b)


#Different values to make different test cases
c_list = np.linspace(0.00, 1, 20)
kernel_list = [kernel().linear_kernel, kernel().polynomial_kernel_3, kernel().polynomial_kernel_2, kernel().gaussian_kernel]
data_size = [1000, 2000, 3000, 4000, 5000, 6000, 7000]

final_result = {}

#loop for each test case
for data_frac in data_size:
    
    data_frac = int(data_frac)
    train_x = complete_data_x[:data_frac]
    train_x = train_x.drop(['id'],axis=1)
    train_x = np.array(train_x)
    train_x = np.double(train_x)
    train_y = complete_data_y[:data_frac]
    train_y = list(train_y['label'])
    train_y = np.double(train_y)
    
    test_x = complete_data_x[data_frac:data_frac+500]
    test_x = test_x.drop(['id'],axis=1)
    test_x = np.array(test_x)
    test_x = np.double(test_x)
    test_y = complete_data_y[data_frac:data_frac+500]
    test_y = list(test_y['label'])
    test_y = np.double(test_y)


    for kernel_name1 in kernel_list:
        for c_val in c_list:

        
            try:
                mySvmModel = svm(kernel_name = kernel_name1, c=c_val)
                mySvmModel.train_model(train_x, train_y)
                predict_y = []
                for element in test_x:
                    predict_y.append(mySvmModel.predict(element))
                
                
                
                matches = [1 for i, j in zip(test_y, predict_y) if i == j]
                accuracy = len(matches)/len(predict_y)
                accuracy = accuracy*100
                error = 100 - accuracy
                
                final_result[(str(kernel_name1), str(c_val), str(data_frac))] = [accuracy, error]
                
                print(str(kernel_name1)+str(c_val)+str(data_frac)+'   '+str(accuracy))
            except:
                pass
            
            
df2 = pd.DataFrame.from_dict(final_result, orient='index')
df2.to_csv('output_cancer.csv', sep=',')
