import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import preprocessing
from pybrain.tools.shortcuts import buildNetwork
from pybrain.tools.customxml import NetworkWriter
from pybrain.tools.customxml import NetworkReader
from pybrain.structure import *
from pybrain.datasets import ClassificationDataSet
from pybrain.supervised.trainers import BackpropTrainer
from sklearn.metrics import accuracy_score
import xlrd
import math
import random as rand
import matplotlib.pyplot as plt
from emd import EMD 
import numpy as np
import os
from scipy.signal import hilbert
from scipy import angle, unwrap
import re
import warnings
warnings.filterwarnings('ignore')
import time

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
def org_sig_get_excel(path):
    workbook = xlrd.open_workbook(path,"rb")
    print(workbook.sheet_names())
    Data_sheet = workbook.sheets()[0]
    rowNum = Data_sheet.nrows  # sheetrow
    colNum = Data_sheet.ncols  # sheet column
    dust_sig = []
    for i in range(rowNum):
     for j in range(colNum):
         dust_sig.append(Data_sheet.cell_value(i, j))

    return dust_sig
def nn_input_getting(x_data_input,ceemd_max_step,white_noise_var,white_noise_add_num,data_analysis_length):
#  nn_input_getting Algorithm: To get the neural network's input
#     input:
#       x_data_input: The original signal data
#       ceemd_max_step: The max decomposition level of ceemd
#       white_noise_var: The white noise's variance
#       white_noise_num: The noise's add num.
#       data_analysis_length: The length of data for one time analyzing.
#    output:
#       nn_input: Neural Network's input
      nn_input = []
      gal_length = len(x_data_input)//data_analysis_length;

#      for ceemd_num_i in range(1,ceemd_max_step):
#      for xdata_num_i in range(0,gal_length):
      for xdata_num_i in range(1,(gal_length-1)):
          print(xdata_num_i)
          x_data_input_segmet = x_data_input[((data_analysis_length*(xdata_num_i))-500):((data_analysis_length*(xdata_num_i+1))+500)]
          x_data_input_segmet_filtered =noise_filter(x_data_input_segmet)
#          eng = matlab.engine.start_matlab()
          imf_rslt,imf_res_all,imf_res_rlst,cj_out = ceemd(x_data_input_segmet_filtered,white_noise_var,white_noise_add_num,ceemd_max_step)
#          cj_res_out = cj_out+[imf_res_rlst]
          cj_res_out = cj_out
          imfs_rlst = imfs_endpoint_processing(cj_res_out,expected_length = data_analysis_length)
#          print len(imfs_rlst) 

#          feature_p,h = get_feature(cj_res_out)
          feature_p,h = get_feature(imfs_rlst)
          average_instfreq_imfs_rlst = average_instfreq_get(imfs_rlst)

          nn_input_seg = nn_input_setting(feature_p+[h]+average_instfreq_imfs_rlst)
          print nn_input_seg
          nn_input    = nn_input+[nn_input_seg]
      return nn_input
def noise_filter(dust_data):

   dsum = 0;
   print 'd_len = %f'%len(dust_data)
   for nn in range(0,len(dust_data)):
    dsum = dsum+dust_data[nn]

   d_avr = dsum/len(dust_data)
   print 'd_var = %f'%d_avr

   d_dust_temp = 0
   for nn in range(0,len(dust_data)):
       d_dust_temp = dust_data[nn] - d_avr
       dust_data[nn] = d_dust_temp
   dust_data_filtered = dust_data

   return dust_data_filtered

def nn_input_setting(feature_input):
#-----------Get BPNN_INPUT
    p_length = len(feature_input);

    esum = 0 ;

    p_square = 0 ;
    bp_input = [0.0] * p_length

    for ii in range(0,p_length): #= 1:1:P_length
       p_square = feature_input[ii]*feature_input[ii]
       esum = esum + p_square

    for ii in range(0,p_length): #= 1:1:P_length
       bp_input[ii] = feature_input[ii]/(math.sqrt(esum))
 #   BP_INPUT = P/(sqrt(Esum));
    return bp_input


def ceemd(input_sig,nstd,ne,eemd_stop):
#This Program is designed to simulation EEMD work with BP-Neural Network
#This Promgram only process one-dimensional signal
#input_sig:The input signal
#nstd:The noise std
#ne:The noise array number
#eemd_stop:How many times EEMD is stoped

    sig_length = len(input_sig)
    xn=[]
    imf=[]
    imf_res = []
    imf_rlst=[]
#    cj=[]
    xn=make_matrix(ne,sig_length)
    xn_plus=make_matrix(ne,sig_length)
    xn_sub=make_matrix(ne,sig_length)
    imf=make_matrix(ne,sig_length)
    imf_plus=make_matrix(ne,sig_length)
    imf_sub=make_matrix(ne,sig_length)
    imf_res = make_matrix(ne,sig_length)
    imf_res_plus = make_matrix(ne,sig_length)
    imf_res_sub = make_matrix(ne,sig_length)
#    imf_rlst= make_matrix(ne,sig_length)
    cj=make_matrix(eemd_stop,sig_length)
#--------------------Step-1 Signal input with white noise------------------
    for ne_i in range(0,ne):
      for length_k in range(0,sig_length):
        white_noise = (rand.random() - 0.5 ) * math.sqrt( 12 * nstd ) + 0
        input_sig_total_plus = white_noise + input_sig[length_k]
        input_sig_total_sub  = input_sig[length_k]-white_noise
        xn_plus[ne_i][length_k] = input_sig_total_plus
        xn_sub[ne_i][length_k] = input_sig_total_sub
#    plt.figure()
#    plt.plot(xn[0][:])
#    plt.show()
    for stop_count in range(0,eemd_stop):
      print(stop_count)
      for ne_i in range(0,ne):
          for sig_length_k in range(0,sig_length):
            imf[ne_i][sig_length_k]= 0
      for ne_i in range(0,ne):
          if stop_count == 0:
              decomposer_plus = EMD(np.array(xn_plus[ne_i][:]),n_imfs=1)
              imf_rlst_plus = decomposer_plus.decompose()
              decomposer_sub = EMD(np.array(xn_sub[ne_i][:]),n_imfs=1)
              imf_rlst_sub = decomposer_sub.decompose()
#             imf_rlst,ort,nbits = eng.emd_ljm(0,1,xn[ne_i][:])
          else:
#             imf_rlst,ort,nbits  = eng.emd_ljm(0,1,imf_res[ne_i][:])
              decomposer_plus = EMD(np.array(imf_res_plus[ne_i][:]),n_imfs=1)
              imf_rlst_plus = decomposer_plus.decompose()
              decomposer_sub = EMD(np.array(imf_res_sub[ne_i][:]),n_imfs=1)
              imf_rlst_sub = decomposer_sub.decompose()
#          print(len(imf_rlst))
          imf_plus[ne_i][:] = imf_rlst_plus[0][:]
          imf_sub[ne_i][:] = imf_rlst_sub[0][:]
          #rlst_length  = size(IMF_rlst);
          rlst_length = len(imf_rlst_plus)
          #if(rlst_length(1)>1)
          if rlst_length >1:
            imf_res_plus[ne_i][:] = imf_rlst_plus[1][:]
            imf_res_sub[ne_i][:] = imf_rlst_sub[1][:]
          else:
            print 'IMF can not be decomposed to so many times,max decomposed times = %d'%(stop_count-1)
          for tsk_i in range(0,len(imf_plus[ne_i])):
              imf[ne_i][tsk_i] = (imf_plus[ne_i][tsk_i] + imf_sub[ne_i][tsk_i])/2
          for tsk_i in range(0,len(imf_res_plus[ne_i])):
              imf_res[ne_i][tsk_i] = (imf_res_plus[ne_i][tsk_i] + imf_res_sub[ne_i][tsk_i])/2

#------------------Step-3 Get Cj----------------------------------            
      for sig_length_k in range(0,sig_length):
           cj[stop_count][sig_length_k] = 0

      esum = 0;
      for ne_i in range(0,ne):
           for sig_length_k in range(0,sig_length):
              esum =  imf[ne_i][sig_length_k] + cj[stop_count][sig_length_k]
              cj[stop_count][sig_length_k] = esum

    eavr = 0;
    for stop_count in range(0,eemd_stop):
      for sig_length_k in range(0,sig_length):
        eavr = cj[stop_count][sig_length_k]/ne;
        cj[stop_count][sig_length_k] = eavr;

    imf_res_sum = []
    imf_res_rlst = []
    imf_res_sum = [0.0] * sig_length
    imf_res_rlst = [0.0] * sig_length
    for res_i in range(0,ne):
       for sig_length_k in range(0,sig_length):
          esum = imf_res[res_i][sig_length_k] + imf_res_sum[sig_length_k]
          imf_res_sum[sig_length_k] = esum

    for sig_length_k in range(0,sig_length):
        imf_res_rlst[sig_length_k] =  imf_res_sum[sig_length_k]/ne

    imf_rlst = imf
    imf_res_all = imf_res
    cj_out = cj

    return imf_rlst,imf_res_all,imf_res_rlst,cj

def imfs_endpoint_processing(imfs,expected_length):

        if len(imfs[0]) > 1:
           imfs_com_length =  len(imfs[0])
        else :
           imfs_com_length =  len(imfs)

        if len(imfs[0]) > 1 :
           imfs_re_rlst = make_matrix(len(imfs),expected_length)
           for tsk_i in range(0,len(imfs)):
               for tsk_j in range(0,expected_length):
                 imfs_re_rlst[tsk_i][tsk_j] = imfs[tsk_i][(tsk_j+(imfs_com_length-expected_length)//2)]
        else :
           for tsk_j in range(0,expected_length):
               imfs_re_rlst[tsk_j] = imfs[(tsk_j+(imfs_com_length-expected_length)//2)]

        return imfs_re_rlst
def get_feature(imf_input):
#--------------Step - 4 -----------------------
    cj_column = len(imf_input)
#    cj_size   = len(imf_input[0][:])
    cj_row    = len(imf_input[0][:]) #cj_size//cj_column
    print(cj_column,cj_row)

#    for tsk_i in range(1,):tsk_i = 1:1:Cj_size(1)
#      P(1,tsk_i) =  0.0;
#    end
    imf_input_hilbert = abs(hilbert(imf_input))

    P = [0.0] * cj_column
    Pq = [0.0] * cj_column
    for tsk_i in range(0,cj_column):  #= 1:1:Cj_size(1)
      for tsk_k in range(0,cj_row):   #= 1:1:Cj_size(2)
#        cj_square = imf_input[tsk_i][tsk_k]*imf_input[tsk_i][tsk_k] + P[tsk_i]
        cj_square = imf_input_hilbert[tsk_i][tsk_k]*imf_input_hilbert[tsk_i][tsk_k] + P[tsk_i]
        P[tsk_i] =  cj_square


    esum = 0.0
    for tsk_i in range(0,cj_column): #= 1:1:Cj_size(1)
      esum =P[tsk_i]+esum;

    for tsk_i in range(0,cj_column): #= 1:1:Cj_size(1)
       Pq[tsk_i] = 0

    for tsk_i in range(0,cj_column): # = 1:1:Cj_size(1)
       Pq[tsk_i] =P[tsk_i]/esum

    H=0
    for tsk_i in range(0,cj_column): # = 1:1:Cj_size(1)
      H =H+Pq[tsk_i]*math.log(Pq[tsk_i])
    H = -H

    Feature_P = P
    return Feature_P,H
def instfreq_get(imfs):
    imf_num = len(imfs)
    hs = make_matrix(imf_num,len(imfs[0]))
    inst_s = make_matrix(imf_num,len(imfs[0]))

    for tsk_i in range(0,imf_num):
       hs[tsk_i] = hilbert(imfs[tsk_i])
       omega_s = unwrap(angle(hs[tsk_i]))
       inst_s[tsk_i] = np.diff(omega_s)

    return inst_s

def nn_input_getting(x_data_input,ceemd_max_step,white_noise_var,white_noise_add_num,data_analysis_length):
#  nn_input_getting Algorithm: To get the neural network's input
#     input:
#       x_data_input: The original signal data
#       ceemd_max_step: The max decomposition level of ceemd
#       white_noise_var: The white noise's variance
#       white_noise_num: The noise's add num.
#       data_analysis_length: The length of data for one time analyzing.
#    output:
#       nn_input: Neural Network's input
      nn_input = []
      gal_length = len(x_data_input)//data_analysis_length;

#      for ceemd_num_i in range(1,ceemd_max_step):
#      for xdata_num_i in range(0,gal_length):
      for xdata_num_i in range(1,(gal_length-1)):
          print(xdata_num_i)
          x_data_input_segmet = x_data_input[((data_analysis_length*(xdata_num_i))-500):((data_analysis_length*(xdata_num_i+1))+500)]
          x_data_input_segmet_filtered =noise_filter(x_data_input_segmet)
#          eng = matlab.engine.start_matlab()
          imf_rslt,imf_res_all,imf_res_rlst,cj_out = ceemd(x_data_input_segmet_filtered,white_noise_var,white_noise_add_num,ceemd_max_step)
#          cj_res_out = cj_out+[imf_res_rlst]
          cj_res_out = cj_out
          imfs_rlst = imfs_endpoint_processing(cj_res_out,expected_length = data_analysis_length)
#          print len(imfs_rlst) 

#          feature_p,h = get_feature(cj_res_out)
          feature_p,h = get_feature(imfs_rlst)
          average_instfreq_imfs_rlst = average_instfreq_get(imfs_rlst)
          nn_input_seg = nn_input_setting(feature_p+[h]+average_instfreq_imfs_rlst)
          print nn_input_seg
          nn_input    = nn_input+[nn_input_seg]
      return nn_input
 
def average_instfreq_get(imfs):

    instfreq_of_imfs = instfreq_get(imfs)
    imf_num = len(imfs)
    if len(imfs[0])>1 :
       average_instfreq_imfs_rlst = [0.0]*imf_num
#       print average_instfreq_imfs_rlst
       for tsk_i in range(0,imf_num):
          average_instfreq_imfs_rlst[tsk_i] = sum(instfreq_of_imfs[tsk_i])/len(instfreq_of_imfs[tsk_i])
    else:
       average_instfreq_imfs_rlst = sum(instfreq_of_imfs)/imf_num

    return average_instfreq_imfs_rlst
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

                     
#f = open('miv_eva.txt', 'w+')

start_time =time.time()
dust_sig_test1 = org_sig_get_excel('../data/20180330-1/1.5.xlsx')
nn_input_test1 = nn_input_getting(x_data_input = dust_sig_test1[0:200000],ceemd_max_step = 4,white_noise_var=0.3,white_noise_add_num=100,data_analysis_length=6000)
f = open('nn_input_test1.txt', 'a+')
f.write(str(nn_input_test1) + os.linesep)
f.close()
dust_sig_test2 = org_sig_get_excel('../data/20180330-1/2.5.xlsx')
nn_input_test2 = nn_input_getting(x_data_input = dust_sig_test2[0:200000],ceemd_max_step = 4,white_noise_var=0.3,white_noise_add_num=100,data_analysis_length=6000)
f = open('nn_input_test2.txt', 'a+')
f.write(str(nn_input_test2) + os.linesep)
f.close()
dust_sig_test3 = org_sig_get_excel('../data/20180330-1/4.5.xlsx')
nn_input_test3 = nn_input_getting(x_data_input = dust_sig_test3[0:200000],ceemd_max_step = 4,white_noise_var=0.3,white_noise_add_num=100,data_analysis_length=6000)
f = open('nn_input_test3.txt', 'a+')
f.write(str(nn_input_test3) + os.linesep)
f.close()

input_test_feature_1  = []
input_test_feature_2  = []
input_test_feature_3  = []

input_test_feature_1 = input_feature_cut(nn_input_test1,1,)
input_test_feature_2 = input_feature_cut(nn_input_test2,1,)
input_test_feature_3 = input_feature_cut(nn_input_test3,1,)

#nn_input_feature_1  = []
#nn_input_feature_2  = []
#nn_input_feature_3  = []
#
#
#nn_input_feature_1 = input_feature_cut(input_test_feature_1,3,"feature")
#nn_input_feature_2 = input_feature_cut(input_test_feature_2,3,"feature")
#nn_input_feature_3 = input_feature_cut(input_test_feature_3,3,"feature")


nn_input_feature1_1  = []
nn_input_feature1_2  = []
nn_input_feature1_3  = []


nn_input_feature1_1 = input_feature_cut(input_test_feature_1,3,"feature")
nn_input_feature1_2 = input_feature_cut(input_test_feature_2,3,"feature")
nn_input_feature1_3 = input_feature_cut(input_test_feature_3,3,"feature")



net =NetworkReader.readFrom('nn_network.xml')
print net
nn_input_test=nn_input_feature1_1 + nn_input_feature1_2 +nn_input_feature1_3
nn_output_test_expected = [0.0]*len(nn_input_feature1_1)+[1.0]*len(nn_input_feature1_2)+[2.0]*len(nn_input_feature1_3)

X = nn_input_test
y = nn_output_test_expected
print X

dataset = ClassificationDataSet(len(X[0]),1,nb_classes=3)
for i in range(len(y)):
    dataset.addSample(list(X[i]),list([y[i]]))

dataset.calculateStatistics()
dataset._convertToOneOfMany()

dataTrain1,dataTest = dataset.splitWithProportion(proportion=0.0)
x_test,y_test = dataTest['input'],dataTest['target']
predict_test = net.activateOnDataset(dataTest)
actual_test = y_test
predict_test1 =[0.0]*len(predict_test)
actual_test1 =[0.0] * len(actual_test)
for tsk_i in range(0,len(predict_test)):
   predict_test1[tsk_i] = np.argmax(predict_test[tsk_i])
   actual_test1[tsk_i] =  np.argmax(actual_test[tsk_i])

print actual_test1,predict_test1
test_acc = accuracy_score(actual_test1,predict_test1)
for i in range(0,len(predict_test)):
   print "predict,test%d"%i,predict_test[i],actual_test[i]

round_test_acc  = round_test_acc +[test_acc]
end_time =time.time()

print 'round_test_acc = ', round_test_acc
print 'runtime = ', (end_time -start_time) 




