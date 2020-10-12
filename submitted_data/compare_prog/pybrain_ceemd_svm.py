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
from sklearn import svm
from sklearn.model_selection import train_test_split
import time



import warnings
warnings.filterwarnings('ignore')

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
          
def make_matrix(m, n, fill=0.0):
    mat = []
    for i in range(m):
        mat.append([fill] * n)
    return mat

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
#emd
    
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

    
#def nn_input_getting(x_data_input,ceemd_max_step,white_noise_var,white_noise_add_num,data_analysis_length):
##  nn_input_getting Algorithm: To get the neural network's input
##     input:
##       x_data_input: The original signal data
##       ceemd_max_step: The max decomposition level of ceemd
##       white_noise_var: The white noise's variance
##       white_noise_num: The noise's add num.
##       data_analysis_length: The length of data for one time analyzing.
##    output:
##       nn_input: Neural Network's input
#      nn_input = []
#      gal_length = len(x_data_input)//data_analysis_length;
#      
##      for ceemd_num_i in range(1,ceemd_max_step):
#      for xdata_num_i in range(0,gal_length):
#          print(xdata_num_i)
#          x_data_input_segmet = x_data_input[(data_analysis_length*(xdata_num_i)):(data_analysis_length*(xdata_num_i+1))]
#          x_data_input_segmet_filtered =noise_filter(x_data_input_segmet)
##          eng = matlab.engine.start_matlab()
#          imf_rslt,imf_res_all,imf_res_rlst,cj_out = ceemd(x_data_input_segmet_filtered,white_noise_var,white_noise_add_num,ceemd_max_step)
##          cj_res_out = cj_out+[imf_res_rlst]
#          cj_res_out = cj_out
#
#          feature_p,h = get_feature(cj_res_out)
#          nn_input_seg = nn_input_setting(feature_p+[h])
#          nn_input    = nn_input+[nn_input_seg]
#      return nn_input
                  
def Ceemd_Bpnn_Adptive_Algorithm(nn_input_training_data,nn_ouput_train_data,nn_input_test_data,nn_output_test_data):
      print "Pending to write"
      
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

def plot_cimfs(imfs):
    imf_num = len(imfs)
    for tsk_i in range(0,len(imfs)):
         plt.subplot(imf_num,1,(tsk_i+1))
         plt.plot(imfs[tsk_i])

def instfreq_get(imfs):
    imf_num = len(imfs)
    hs = make_matrix(imf_num,len(imfs[0]))
    inst_s = make_matrix(imf_num,len(imfs[0]))

    for tsk_i in range(0,imf_num):
       hs[tsk_i] = hilbert(imfs[tsk_i])
       omega_s = unwrap(angle(hs[tsk_i]))
       inst_s[tsk_i] = np.diff(omega_s)

    return inst_s

def angle_trans_to_hz(inpt_freq,Fs,inpt_dimension = False):
      if inpt_dimension == False :
         for tsk_j in range(0,len(inpt_freq)):
            inpt_freq[tsk_j] = np.abs(inpt_freq[tsk_j])*Fs/2/pi

      else:
         for tsk_i in range(0,len(inpt_freq)):
           for tsk_j in range(0,len(inpt_freq[tsk_i])):
            inpt_freq[tsk_i][tsk_j] = np.abs(inpt_freq[tsk_i][tsk_j])*Fs/2/pi
                                                                 
      return inpt_freq

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
                  

round_train_acc = []
round_test_acc = []

start_time =time.time()
for ceemd_max_step_i in range(4,5):
    dust_sig = org_sig_get_excel('../data/20180330/1.5.xlsx')
    dust_sig1 = org_sig_get_excel('../data/20180330/2.5.xlsx')
    dust_sig2 = org_sig_get_excel('../data/20180330/4.5.xlsx')
    nn_input = nn_input_getting(x_data_input = dust_sig[:],ceemd_max_step = ceemd_max_step_i,white_noise_var=0.3,white_noise_add_num=100,data_analysis_length=6000)
    f = open('nn_input.txt', 'a+')
    f.write(str(nn_input) + os.linesep)
    f.close()
    nn_input1 = nn_input_getting(x_data_input = dust_sig1[:],ceemd_max_step = ceemd_max_step_i,white_noise_var=0.3,white_noise_add_num=100,data_analysis_length=6000)
    f = open('nn_input1.txt', 'a+')
    f.write(str(nn_input1) + os.linesep)
    f.close()
    nn_input2 = nn_input_getting(x_data_input = dust_sig2[:],ceemd_max_step = ceemd_max_step_i,white_noise_var=0.3,white_noise_add_num=100,data_analysis_length=6000)
    f = open('nn_input2.txt', 'a+')
    f.write(str(nn_input2) + os.linesep)
    f.close()
    nn_output_expected = [0.0]*len(nn_input)+[1.0]*len(nn_input1)+[2.0]*len(nn_input2)
    X = np.array(nn_input+nn_input1+nn_input2)     #+nn_input_test
    y = np.array(nn_output_expected)               #+nn_output_test_expected
#    print X
#    print y

#    dataset = ClassificationDataSet(len(X[0]),1,nb_classes=3)
#    for i in range(len(y)):
#        dataset.addSample(list(X[i]),list([y[i]]))
#    dataset.calculateStatistics()
#    dataset._convertToOneOfMany()
#    dataTrain,dataTest1 = dataset.splitWithProportion(proportion=1) 
#    x_train,y_train = dataTrain['input'],dataTrain['target']

    x_train,x_test,y_train,y_test=train_test_split(X,y,random_state=1,train_size=0.6)
    print len(x_train)
    print len(x_test)
    clf=svm.SVC(C=0.8,kernel='rbf',gamma=20,decision_function_shape='ovr')
#    clf.fit(x_train,y_train.ravel)
    clf.fit(x_train,y_train)
    print "SVM Training set accuracy",clf.score(x_train,y_train)
    y_hat =clf.predict(x_train)
#    print "SVM Training set accuracy:",clf.score(x_tr)
 #   show_accuracy(y_hat,y_train,'Training set') 
    print "SVM Testing set accuracy",clf.score(x_test,y_test)
    y_hat =clf.predict(x_test)
 #   show_accuracy(y_hat,y_test,'Testing set') 
#    clf.fit(x_train,y_train.ravel())
#    dust_sig_test1 = org_sig_get_excel('../data/20180330-1/1.5.xlsx')
#    nn_input_test1 = nn_input_getting(x_data_input = dust_sig_test1[:],ceemd_max_step = ceemd_max_step_i,white_noise_var=0.3,white_noise_add_num=100,data_analysis_length=6000)
#    f = open('nn_input_test1.txt', 'a+')
#    f.write(str(nn_input_test1) + os.linesep)
#    f.close()  
#    dust_sig_test2 = org_sig_get_excel('../data/20180330-1/2.5.xlsx')
#    nn_input_test2 = nn_input_getting(x_data_input = dust_sig_test2[:],ceemd_max_step = ceemd_max_step_i,white_noise_var=0.3,white_noise_add_num=100,data_analysis_length=6000)
#    f = open('nn_input_test2.txt', 'a+')
#    f.write(str(nn_input_test2) + os.linesep)
#    f.close() 
#    dust_sig_test3 = org_sig_get_excel('../data/20180330-1/4.5.xlsx')
#    nn_input_test3 = nn_input_getting(x_data_input = dust_sig_test3[:],ceemd_max_step = ceemd_max_step_i,white_noise_var=0.3,white_noise_add_num=100,data_analysis_length=6000)
#    f = open('nn_input_test3.txt', 'a+')
#    f.write(str(nn_input_test3) + os.linesep)
#    f.close() 
#    nn_input_test =   nn_input_test1 + nn_input_test2 +nn_input_test3
#    nn_output_test_expected = [0.0]*len(nn_input_test1)+[1.0]*len(nn_input_test2)+[2.0]*len(nn_input_test3)
#
#    X = nn_input_test
#    y = nn_output_test_expected
#    print X
#    print y
#    dataset = ClassificationDataSet(len(X[0]),1,nb_classes=3)
#    for i in range(len(y)):
#        dataset.addSample(list(X[i]),list([y[i]]))
#
#    dataset.calculateStatistics()
#    dataset._convertToOneOfMany()
#
#    dataTrain1,dataTest = dataset.splitWithProportion(proportion=0.0)
# #   dataTest=dataset 
#    x_test,y_test = dataTest['input'],dataTest['target']
#    print 'Input dim:{} Output dim:{}'.format(dataTrain.indim,dataTrain.outdim)
#    print 'Train: x = {} y = {}'.format(x_train.shape,y_train.shape)
#    print 'Test: x = {} y = {}'.format(x_test.shape,y_test.shape)
#
#    net = buildNetwork(dataTrain.indim,2,5,dataTrain.outdim,bias = True,hiddenclass = TanhLayer,outclass=SoftmaxLayer) #Two hiddenlayer: hidden0=4,hidden1=20
#    model = BackpropTrainer(net,dataTrain,learningrate=0.00001,momentum=0.99,verbose=True)
#    model.trainUntilConvergence(maxEpochs=200000)
#    
#    print net.activateOnDataset(dataTrain)
#    predict_train = net.activateOnDataset(dataTrain)
#    actual_train = y_train
#    predict_train1 =[0.0]*len(predict_train)
#    actual_train1 =[0.0] * len(actual_train)
#    for tsk_i in range(0,len(predict_train)):
#       predict_train1[tsk_i] = np.argmax(predict_train[tsk_i])
#       actual_train1[tsk_i] =  np.argmax(actual_train[tsk_i])
#
#    train_acc = accuracy_score(actual_train1,predict_train1)
#    predict_test = net.activateOnDataset(dataTest)
#    actual_test = y_test
#    predict_test1 =[0.0]*len(predict_test)
#    actual_test1 =[0.0] * len(actual_test)
#    for tsk_i in range(0,len(predict_test)):
#       predict_test1[tsk_i] = np.argmax(predict_test[tsk_i]) 
#       actual_test1[tsk_i] =  np.argmax(actual_test[tsk_i])
#   
#    print actual_test1,predict_test1
#    test_acc = accuracy_score(actual_test1,predict_test1)
#    for i in range(0,len(predict_test)):
#       print "predict,test%d"%i,predict_test[i],actual_test[i]
#
#    round_train_acc = round_train_acc + [train_acc] 
#    round_test_acc  = round_test_acc +[test_acc]
#
#    print 'Train acc = ',round(train_acc,2),' Test acc = ',round(test_acc,2)
#    print 'round_train_acc = ',round_train_acc,'round_test_acc = ', round_test_acc
#
#    if round(test_acc,4) >=0.998 and round(train_acc,2)>=0.998: break
end_time =time.time()

print 'runtime = ', (end_time -start_time)

