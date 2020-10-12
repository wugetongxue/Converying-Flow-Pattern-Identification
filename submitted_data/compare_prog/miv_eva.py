import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import preprocessing
from pybrain.tools.shortcuts import buildNetwork
from pybrain.structure import *
from pybrain.datasets import ClassificationDataSet
#from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
from sklearn.metrics import accuracy_score
import xlrd
#import matlab.engine
import math
import random as rand
import matplotlib.pyplot as plt
from visualization import plot_imfs
from emd import EMD 
import numpy as np
import os
from scipy.signal import hilbert
from scipy import angle, unwrap
import re
import warnings
warnings.filterwarnings('ignore')

round_train_acc = []
round_test_acc = []

def make_matrix(m, n, fill=0.0):
    mat = []
    for i in range(m):
       mat.append([fill] * n)
    return mat

def dat_get_from_log(path): 
  dat_i = []
  dat_rlst = []
#  with open("../../paper_used_log/nn_input.txt",'rb') as f:
  with open(path,'rb') as f:
    buff = f.read()
#    yy=buff.replace('[',' ')
    num = buff.split()
    for tsk_i in range(0,len(num)):
#    for tsk_i in range(0,4):
        dat = num[tsk_i]
        if dat.find(']')!=-1:
          yy=dat.replace(']','')
          yy_1=yy.replace(',','')
          dat_o = float(yy_1)
          dat_i.append(dat_o)
          dat_rlst.append(dat_i)
          dat_i = []
        elif dat.find('[')!=-1:
          yy=dat.replace('[','')
          yy_1=yy.replace(',','')
          dat_o = float(yy_1)
          dat_i.append(dat_o)
        else:
          yy_1=dat.replace(',','')
          dat_o = float(yy_1)
          dat_i.append(dat_o)

  return dat_rlst 
#    f = open('temp.txt', 'w+')
#    f.write(str(dat_rlst) + os.linesep)
#    f.close()

#for ceemd_max_step_i in range(2,8):
def nn_training(nn_input,nn_input1,nn_input2,nn_input_test1,nn_input_test2,nn_input_test3):
#    nn_input = dat_get_from_log("../../paper_used_log/nn_input.txt") 
#    nn_input1 = dat_get_from_log("../../paper_used_log/nn_input1.txt") 
#    nn_input2 = dat_get_from_log("../../paper_used_log/nn_input2.txt")
    nn_output_expected = [0.0]*len(nn_input)+[1.0]*len(nn_input1)+[2.0]*len(nn_input2)
    X = nn_input+nn_input1+nn_input2     #+nn_input_test
    y = nn_output_expected               #+nn_output_test_expected
#    print X
#    print y

    dataset = ClassificationDataSet(len(X[0]),1,nb_classes=3)
    for i in range(len(y)):
        dataset.addSample(list(X[i]),list([y[i]]))
    dataset.calculateStatistics()
    dataset._convertToOneOfMany()
    dataTrain,dataTest1 = dataset.splitWithProportion(proportion=1) 
    x_train,y_train = dataTrain['input'],dataTrain['target']

#    nn_input_test1 = dat_get_from_log("../../paper_used_log/nn_input_test1.txt") 
#    nn_input_test2 = dat_get_from_log("../../paper_used_log/nn_input_test2.txt") 
#    nn_input_test3 = dat_get_from_log("../../paper_used_log/nn_input_test3.txt") 
    nn_input_test =   nn_input_test1 + nn_input_test2 +nn_input_test3
    nn_output_test_expected = [0.0]*len(nn_input_test1)+[1.0]*len(nn_input_test2)+[2.0]*len(nn_input_test3)

    X = nn_input_test
    y = nn_output_test_expected
#    print X
#    print y
    dataset = ClassificationDataSet(len(X[0]),1,nb_classes=3)
    for i in range(len(y)):
        dataset.addSample(list(X[i]),list([y[i]]))

    dataset.calculateStatistics()
    dataset._convertToOneOfMany()

    dataTrain1,dataTest = dataset.splitWithProportion(proportion=0.0)
    x_test,y_test = dataTest['input'],dataTest['target']
#    print 'Input dim:{} Output dim:{}'.format(dataTrain.indim,dataTrain.outdim)
#    print 'Train: x = {} y = {}'.format(x_train.shape,y_train.shape)
#    print 'Test: x = {} y = {}'.format(x_test.shape,y_test.shape)

#    net = buildNetwork(dataTrain.indim,4,20,dataTrain.outdim,bias = True,hiddenclass = TanhLayer,outclass=SoftmaxLayer) #Two hiddenlayer: hidden0=4,hidden1=20
#    net = buildNetwork(dataTrain.indim,4,5,dataTrain.outdim,bias = True,hiddenclass = TanhLayer,outclass=SoftmaxLayer) #Two hiddenlayer: hidden0=4,hidden1=20
    net = buildNetwork(dataTrain.indim,2,5,dataTrain.outdim,bias = True,hiddenclass = TanhLayer,outclass=SoftmaxLayer) #Two hiddenlayer: hidden0=4,hidden1=20
#    net = buildNetwork(dataTrain.indim,4,15,dataTrain.outdim,bias = True,hiddenclass = TanhLayer) #Two hiddenlayer: hidden0=4,hidden1=20
#    net = buildNetwork(dataTrain.indim,20,dataTrain.outdim, bias = True, hiddenclass =  TanhLayer,outclass = SoftmaxLayer)
    model = BackpropTrainer(net,dataTrain,learningrate=0.00001,momentum=0.99,verbose=True)
#    model.trainUntilConvergence(maxEpochs=200000)
    model.trainUntilConvergence(maxEpochs=10000)
    
#    print net.activateOnDataset(dataTrain)
    predict_train = net.activateOnDataset(dataTrain)
    actual_train = y_train
    predict_train1 =[0.0]*len(predict_train)
    actual_train1 =[0.0] * len(actual_train)
    for tsk_i in range(0,len(predict_train)):
       predict_train1[tsk_i] = np.argmax(predict_train[tsk_i])
       actual_train1[tsk_i] =  np.argmax(actual_train[tsk_i])

    train_acc = accuracy_score(actual_train1,predict_train1)
    predict_test = net.activateOnDataset(dataTest)
    actual_test = y_test
    predict_test1 =[0.0]*len(predict_test)
    actual_test1 =[0.0] * len(actual_test)
    for tsk_i in range(0,len(predict_test)):
       predict_test1[tsk_i] = np.argmax(predict_test[tsk_i]) 
       actual_test1[tsk_i] =  np.argmax(actual_test[tsk_i])
   
#    print actual_test1,predict_test1
    test_acc = accuracy_score(actual_test1,predict_test1)
#    for i in range(0,len(predict_test)):
#       print "predict,test%d"%i,predict_test[i],actual_test[i]

#    round_train_acc = round_train_acc + [train_acc] 
#    round_test_acc  = round_test_acc +[test_acc]

    print 'Train acc = ',round(train_acc,2),' Test acc = ',round(test_acc,2)
    print 'round_train_acc = ',round_train_acc,'round_test_acc = ', round_test_acc

    return net,train_acc,test_acc
#    if round(test_acc,4) >=0.998 and round(train_acc,2)>=0.998: break

def miv_eva_feature_gen(input_feature,feature_position,flag = "imf",chg_rate = 0.1):
#    feature_rlst_add  = input_feature
#    feature_rlst_sub  = input_feature
    feature_rlst_add =  make_matrix(len(input_feature), len(input_feature[0]), )
    feature_rlst_sub =  make_matrix(len(input_feature), len(input_feature[0]), )
    for tsk_i in range(0,len(input_feature)):
       for tsk_j in range(0,len(input_feature[0])):
           feature_rlst_add[tsk_i][tsk_j] = input_feature[tsk_i][tsk_j]
           feature_rlst_sub[tsk_i][tsk_j] = input_feature[tsk_i][tsk_j]

    imf_order = (len(input_feature[0])-1)//2 + 1
    if flag == "imf" : 
        for tsk_i in range(0,len(input_feature)):
           feature_rlst_add[tsk_i][feature_position] = input_feature[tsk_i][feature_position]*(1+chg_rate)
           feature_rlst_add[tsk_i][(feature_position+imf_order)] = input_feature[tsk_i][(feature_position+imf_order)] * (1+chg_rate)
           feature_rlst_sub[tsk_i][feature_position] = input_feature[tsk_i][feature_position] *(1-chg_rate) 
           feature_rlst_sub[tsk_i][(feature_position+imf_order)] = input_feature[tsk_i][(feature_position+imf_order)] *(1-chg_rate) 
    else: 
        for tsk_i in range(0,len(input_feature)):
           feature_rlst_add[tsk_i][feature_position] = input_feature[tsk_i][feature_position] * (1+chg_rate)
           feature_rlst_sub[tsk_i][feature_position] = input_feature[tsk_i][feature_position] * (1-chg_rate) 

           
    return  feature_rlst_add, feature_rlst_sub 

def miv_eva_rlst(net,input_feature_add,input_feature_sub):

    X = input_feature_add
    y = [0.0]*len(input_feature_add)
    dataset = ClassificationDataSet(len(X[0]),1,nb_classes=3)
    for i in range(len(y)):
       dataset.addSample(list(X[i]),list([y[i]]))
    dataset.calculateStatistics()
    dataset._convertToOneOfMany()
    dataTest_feature_add,dataTest1 = dataset.splitWithProportion(proportion=1)
    x_train_add,y_train_add = dataTest_feature_add['input'],dataTest_feature_add['target']
    predict_train = net.activateOnDataset(dataTest_feature_add)
    predict_train_add =[0.0]*len(predict_train)
    for tsk_i in range(0,len(predict_train)):
         predict_train_add[tsk_i] = np.argmax(predict_train[tsk_i])
    
    X = input_feature_sub
    y = [0.0]*len(input_feature_sub)
    dataset = ClassificationDataSet(len(X[0]),1,nb_classes=3)
    for i in range(len(y)):
        dataset.addSample(list(X[i]),list([y[i]]))
    dataset.calculateStatistics()
    dataset._convertToOneOfMany()
    dataTest_feature_sub,dataTest1 = dataset.splitWithProportion(proportion=1)
    x_train_sub,y_train_add = dataTest_feature_sub['input'],dataTest_feature_sub['target']
    predict_train = net.activateOnDataset(dataTest_feature_sub)
    predict_train_sub =[0.0]*len(predict_train)
    for tsk_i in range(0,len(predict_train)):
         predict_train_sub[tsk_i] = np.argmax(predict_train[tsk_i])
    rlst = abs(sum(predict_train_add)-sum(predict_train_sub))
    
#    print rlst
    return rlst

def input_feature_cut(input_feature,position,flag="imf"):

    if flag == "imf":
        input_feature_len_pos = (len(input_feature[0])-1)//2+1
        output_feature_rlst = make_matrix(len(input_feature), (len(input_feature[0])-2), )
        pos = 0
        for tsk_i in range(0,len(input_feature)):
          pos =0
          for tsk_j in range(0,len(input_feature[0])):
            if (tsk_j == position) or (tsk_j == input_feature_len_pos) :
                pass
            else:
                output_feature_rlst[tsk_i][pos] = input_feature[tsk_i][tsk_j] 
                pos+=1
    else:
        output_feature_rlst = make_matrix(len(input_feature), (len(input_feature[0])-1), )
        pos = 0
        for tsk_i in range(0,len(input_feature)):
          pos =0
          for tsk_j in range(0,len(input_feature[0])):
            if (tsk_j == position):
                pass
            else:
                output_feature_rlst[tsk_i][pos] = input_feature[tsk_i][tsk_j] 
                pos+=1
                
    return output_feature_rlst 

def eva_feature_cpy(input_feature):
    feature_rlst =  make_matrix(len(input_feature), len(input_feature[0]), )
    for tsk_i in range(0,len(input_feature)):
       for tsk_j in range(0,len(input_feature[0])):
           feature_rlst[tsk_i][tsk_j] = input_feature[tsk_i][tsk_j]

    return feature_rlst

#a = [[0.1,0.2,0.3,0.4,0.5,0.6,0.7],[100,200,300,400,500,600,700]]
#input_feature_add,input_feature_sub = miv_eva_feature_gen(a,0,flag = "imf")
#print input_feature_add,input_feature_sub
#input_feature_add,input_feature_sub = miv_eva_feature_gen(a,1,flag = "imf")
#print input_feature_add,input_feature_sub
#input_feature_add,input_feature_sub = miv_eva_feature_gen(a,2,flag = "imf")
#print input_feature_add,input_feature_sub
#input_train_feature_1 = input_feature_cut(a,0,)
#print input_train_feature_1


f = open('miv_eva.txt', 'w+')
#MAX_TRAIN_TIMES = 20
MAX_TRAIN_TIMES = 10
nn_input = dat_get_from_log("../../paper_used_log/nn_input.txt") 
nn_input1 = dat_get_from_log("../../paper_used_log/nn_input1.txt") 
nn_input2 = dat_get_from_log("../../paper_used_log/nn_input2.txt")
nn_input_test1 = dat_get_from_log("../../paper_used_log/nn_input_test1.txt") 
nn_input_test2 = dat_get_from_log("../../paper_used_log/nn_input_test2.txt") 
nn_input_test3 = dat_get_from_log("../../paper_used_log/nn_input_test3.txt") 
#nn_input = [[0.1,0.2,0.3,0.4,0.5,0.6,0.7]]
#nn_input1 = [[0.1,0.2,0.9,0.4,0.5,0.6,0.7]]
#nn_input2 = [[0.1,0.2,1.6,0.4,0.5,0.6,0.7]]
#nn_input_test1 = [[0.1,0.2,0.3,0.4,0.5,0.6,0.7]]
#nn_input_test2 = [[0.1,0.2,0.9,0.4,0.5,0.6,0.7]]
#nn_input_test3 = [[0.1,0.2,1.6,0.4,0.5,0.6,0.7]]
for tsk_i in range(0,MAX_TRAIN_TIMES):
   nn_net,nn_train_acc,nn_test_acc = nn_training(nn_input,nn_input1,nn_input2,nn_input_test1,nn_input_test2,nn_input_test3)
   if nn_train_acc >= 0.998 and nn_test_acc >= 0.998: break
if tsk_i > 19 : 
   print "MAX_TRAIN_TIMES reached,but the net is not coveragance"
else:
   round_train_acc = round_train_acc + [nn_train_acc] 
   round_test_acc  = round_test_acc +[nn_test_acc]

f.write("net:" + os.linesep)
f.write(str(nn_net) + os.linesep)
print 'round_train_acc = ',round_train_acc,'round_test_acc = ', round_test_acc
f.write("round_train_acc:" + os.linesep)
f.write(str(round_train_acc) + os.linesep)
f.write("round_test_acc:" + os.linesep)
f.write(str(round_test_acc) + os.linesep)

#Test_feature = nn_input_test1+nn_input_test2+nn_input_test3
input_train_1 = eva_feature_cpy(nn_input)
input_train_2 = eva_feature_cpy(nn_input1)
input_train_3 = eva_feature_cpy(nn_input2)
input_test_1  = eva_feature_cpy(nn_input_test1)
input_test_2  = eva_feature_cpy(nn_input_test2)
input_test_3  = eva_feature_cpy(nn_input_test3)

train_times = 1
while(True):
    
    f.write("Train Times = " + os.linesep)
    f.write(str(train_times) + os.linesep)
    Test_feature = [] 
    Test_feature = input_test_1+input_test_2+input_test_3
    f.write("input data feature length:" + os.linesep)
    f.write(str(len(Test_feature[0])) + os.linesep)
    miv_rlst = [0.0]*((len(Test_feature[0])-1)//2)
    for tsk_i in range(0,((len(Test_feature[0])-1)//2)):
        input_feature_add,input_feature_sub = miv_eva_feature_gen(Test_feature,tsk_i,flag = "imf")
        print input_feature_add,input_feature_sub
        miv_rlst[tsk_i]=miv_eva_rlst(nn_net,input_feature_add,input_feature_sub)
    #    print miv_rlst
    f.write("miv_rlst:" + os.linesep)
    f.write(str(miv_rlst) + os.linesep)
    min_feature_index = np.argmin(miv_rlst)
    f.write("The remove imf is :" + os.linesep)
    f.write(str(min_feature_index) + os.linesep)

    train_times +=1
    input_train_reserve_1 =  [] 
    input_train_reserve_2 =  [] 
    input_train_reserve_3 =  [] 
    input_test_reserve_1  =  [] 
    input_test_reserve_2  =  [] 
    input_test_reserve_3  =  [] 

    input_train_reserve_1 = eva_feature_cpy(input_train_1)
    input_train_reserve_2 = eva_feature_cpy(input_train_2)
    input_train_reserve_3 = eva_feature_cpy(input_train_3)
    input_test_reserve_1  = eva_feature_cpy(input_test_1)
    input_test_reserve_2  = eva_feature_cpy(input_test_2)
    input_test_reserve_3  = eva_feature_cpy(input_test_3)

    input_train_feature_1 = []
    input_train_feature_2 = []
    input_train_feature_3 = []
    input_test_feature_1  = []
    input_test_feature_2  = []
    input_test_feature_3  = []

    input_train_feature_1 = input_feature_cut(input_train_1,min_feature_index,)
    input_train_feature_2 = input_feature_cut(input_train_2,min_feature_index,)
    input_train_feature_3 = input_feature_cut(input_train_3,min_feature_index,)
    input_test_feature_1 = input_feature_cut(input_test_1,min_feature_index,)
    input_test_feature_2 = input_feature_cut(input_test_2,min_feature_index,)
    input_test_feature_3 = input_feature_cut(input_test_3,min_feature_index,)

    input_train_1 =[]
    input_train_2 =[]
    input_train_3 =[]
    input_test_1  =[]
    input_test_2  =[]
    input_test_3  =[]

    input_train_1 = eva_feature_cpy(input_train_feature_1)
    input_train_2 = eva_feature_cpy(input_train_feature_2)
    input_train_3 = eva_feature_cpy(input_train_feature_3)
    input_test_1  = eva_feature_cpy(input_test_feature_1)
    input_test_2  = eva_feature_cpy(input_test_feature_2)
    input_test_3  = eva_feature_cpy(input_test_feature_3)
    for tsk_i in range(0,MAX_TRAIN_TIMES):
           nn_net,nn_train_acc,nn_test_acc = nn_training(input_train_1,input_train_2,input_train_3,input_test_1,input_test_2,input_test_3)
           if nn_train_acc >= 0.998 and nn_test_acc >= 0.998: break
#    if tsk_i >= 19 :
    if tsk_i >= (MAX_TRAIN_TIMES-1) :
           print "MAX_TRAIN_TIMES reached,but the net is not coveragance"
           break
    else:
           round_train_acc = round_train_acc + [nn_train_acc]
           round_test_acc  = round_test_acc +[nn_test_acc]

    nn_net_reserve = nn_net   

    f.write("net:" + os.linesep)
    f.write(str(nn_net) + os.linesep)
    print 'round_train_acc = ',round_train_acc,'round_test_acc = ', round_test_acc
    f.write("round_train_acc:" + os.linesep)
    f.write(str(round_train_acc) + os.linesep)
    f.write("round_test_acc:" + os.linesep)
    f.write(str(round_test_acc) + os.linesep)

#print round
nn_net = nn_net_reserve 
input_train_1 = eva_feature_cpy(input_train_reserve_1)
input_train_2 = eva_feature_cpy(input_train_reserve_2)
input_train_3 = eva_feature_cpy(input_train_reserve_3)
input_test_1  = eva_feature_cpy(input_test_reserve_1)
input_test_2  = eva_feature_cpy(input_test_reserve_2)
input_test_3  = eva_feature_cpy(input_test_reserve_3)

while(True):
    Test_feature = []
    Test_feature = input_test_1+input_test_2+input_test_3
    miv_rlst = [0.0]*len(Test_feature[0])
    for tsk_i in range(0,len(Test_feature[0])):
        input_feature_add,input_feature_sub = miv_eva_feature_gen(Test_feature,tsk_i,flag = "feature")
        miv_rlst[tsk_i]=miv_eva_rlst(nn_net,input_feature_add,input_feature_sub)
                                    #    print miv_rlst
    f.write("miv_rlst:" + os.linesep)
    f.write(str(miv_rlst) + os.linesep)
    min_feature_index = np.argmin(miv_rlst)
    f.write("The remove feature is :" + os.linesep)
    f.write(str(min_feature_index) + os.linesep)
    
    input_train_feature_1 = [] 
    input_train_feature_2 = [] 
    input_train_feature_3 = [] 
    input_test_feature_1  = [] 
    input_test_feature_2  = [] 
    input_test_feature_3  = [] 
    input_train_feature_1 = input_feature_cut(input_train_1,min_feature_index,"feature")
    input_train_feature_2 = input_feature_cut(input_train_2,min_feature_index,"feature")
    input_train_feature_3 = input_feature_cut(input_train_3,min_feature_index,"feature")
    input_test_feature_1 = input_feature_cut(input_test_1,min_feature_index,"feature")
    input_test_feature_2 = input_feature_cut(input_test_2,min_feature_index,"feature")
    input_test_feature_3 = input_feature_cut(input_test_3,min_feature_index,"feature")
    
    input_train_1 =[] 
    input_train_2 =[] 
    input_train_3 =[] 
    input_test_1  =[] 
    input_test_2  =[] 
    input_test_3  =[] 

    input_train_1 = eva_feature_cpy(input_train_feature_1)
    input_train_2 = eva_feature_cpy(input_train_feature_2)
    input_train_3 = eva_feature_cpy(input_train_feature_3)
    input_test_1  = eva_feature_cpy(input_test_feature_1)
    input_test_2  = eva_feature_cpy(input_test_feature_2)
    input_test_3  = eva_feature_cpy(input_test_feature_3)
    for tsk_i in range(0,MAX_TRAIN_TIMES):
        nn_net,nn_train_acc,nn_test_acc = nn_training(input_train_1,input_train_2,input_train_3,input_test_1,input_test_2,input_test_3)
        if nn_train_acc >= 0.998 and nn_test_acc >= 0.998: break
#    if tsk_i >= 19 :
    if tsk_i >= (MAX_TRAIN_TIMES-1) :
       print "MAX_TRAIN_TIMES reached,but the net is not coveragance"
       break
    else:
       round_train_acc = round_train_acc + [nn_train_acc]
       round_test_acc  = round_test_acc +[nn_test_acc]
    nn_net_reverve = nn_net

    f.write("net:" + os.linesep)
    f.write(str(nn_net) + os.linesep)
    print 'round_train_acc = ',round_train_acc,'round_test_acc = ', round_test_acc
    f.write("round_train_acc:" + os.linesep)
    f.write(str(round_train_acc) + os.linesep)
    f.write("round_test_acc:" + os.linesep)
    f.write(str(round_test_acc) + os.linesep)

