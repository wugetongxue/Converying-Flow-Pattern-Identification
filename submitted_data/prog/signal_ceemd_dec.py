from pybrain_ceemd import ceemd
from pybrain_ceemd import org_sig_get_excel
from pybrain_ceemd import noise_filter
from visualization import plot_imfs
import matplotlib.pyplot as plt
from scipy.signal import hilbert
from pybrain_ceemd import make_matrix
import numpy as np
from scipy import angle, unwrap
from math import pi


def plot_cimfs(imfs):
    imf_num = len(imfs)
    for tsk_i in range(0,len(imfs)):
       plt.subplot(imf_num,1,(tsk_i+1))
       plt.plot(imfs[tsk_i])

def instfreq_get(imfs,mix_dismension = True):
    imf_num = len(imfs)
    if mix_dismension == True:
       hs = make_matrix(imf_num,len(imfs[0]))
       inst_s = make_matrix(imf_num,len(imfs[0]))
       
       for tsk_i in range(0,imf_num):
         hs[tsk_i] = hilbert(imfs[tsk_i])
         omega_s = unwrap(angle(hs[tsk_i]))
         inst_s[tsk_i] = np.diff(omega_s)
    else:
       hs = hilbert(imfs)
       omega_s = unwrap(angle(hs))
       inst_s = np.diff(omega_s)

    return inst_s

def angle_trans_to_hz(inpt_freq,Fs,inpt_dimension = False):
  if inpt_dimension == False :
    for tsk_j in range(0,len(inpt_freq)):
       inpt_freq[tsk_j] = np.abs(inpt_freq[tsk_j])*Fs/2/pi

#    return freq
  else:
    for tsk_i in range(0,len(inpt_freq)):
      for tsk_j in range(0,len(inpt_freq[tsk_i])):
       inpt_freq[tsk_i][tsk_j] = np.abs(inpt_freq[tsk_i][tsk_j])*Fs/2/pi
    
  return inpt_freq

def imfs_endpoint_processing(imfs,expected_length,Mix_dismension = False):
  
#    if len(imfs[0]) > 1:
    if Mix_dismension == True:
      imfs_com_length =  len(imfs[0]) 
    else :
      imfs_com_length =  len(imfs) 
    
    if Mix_dismension == True:
      imfs_re_rlst = make_matrix(len(imfs),expected_length)
      for tsk_i in range(0,len(imfs)):
          for tsk_j in range(0,expected_length):
            imfs_re_rlst[tsk_i][tsk_j] = imfs[tsk_i][(tsk_j+(imfs_com_length-expected_length)//2)] 
    else :
      imfs_re_rlst  = [0.0]*expected_length
      for tsk_j in range(0,expected_length):
          imfs_re_rlst[tsk_j] = imfs[(tsk_j+(imfs_com_length-expected_length)//2)]

    return imfs_re_rlst

white_noise_var =0.3
white_noise_add_num=100
ceemd_max_step = 9
acquired_fs = 1000
def emd_dec_instfreq_analysis(inpt_signal,expected_length,signal_name):

    imf_rslt1,imf_res_a1,imf_res_rlst1,cj_out1 = ceemd(inpt_signal,white_noise_var,white_noise_add_num,ceemd_max_step)
    pt_imfs = cj_out1+[imf_res_rlst1]
    pt_imfs_expected =imfs_endpoint_processing(pt_imfs,expected_length,True) 
    inpt_signal_expected =imfs_endpoint_processing(inpt_signal,expected_length) 
    pt_num = len(pt_imfs) + 1
    plt.figure()
    ax=plt.subplot(pt_num,2,1)
#    plt.setp(ax.get_xticklabels(), visible=False)
    plt.title("Signal and Its IMFs/V")
    plt.plot(inpt_signal_expected)
    plt.ylim([(min(inpt_signal_expected)*1.11), (max(inpt_signal_expected)*1.11)])
    plt.xlim([0,expected_length])
    plt.setp(ax.get_xticklabels(), visible=False)
#    plt.axis('off')
#    plt.ylabel(signal_name,'Rotation',90)
#    plt.ylabel(signal_name,rotation='horizontal',fontsize=12,verticalalignment='center',horizontalalignment='left')
    plt.ylabel(signal_name,fontsize=12)
    for tsk_i in range(1,pt_num):
        ax=plt.subplot(pt_num,2,(2*tsk_i+1))
        plt.plot(pt_imfs_expected[(tsk_i-1)])
        imf_name = "imf" + str(tsk_i)
        plt.ylim([(min(pt_imfs_expected[(tsk_i-1)])*1.11), (max(pt_imfs_expected[(tsk_i-1)])*1.11)])
#        plt.ylabel(imf_name,rotation='horizontal',fontsize=12,verticalalignment='center',horizontalalignment='left')
        plt.ylabel(imf_name,fontsize=12)
        plt.xlim([0,expected_length])
        plt.setp(ax.get_xticklabels(), visible=False)
#        plt.axis('off')
    plt.ylabel('RES',fontsize=12)
    plt.setp(ax.get_xticklabels(), visible=True)
    plt.xlim([0,expected_length/acquired_fs])
    plt.xlabel("Time/s") 
    
    signal_inst_s = instfreq_get(inpt_signal,False)
    signal_inst_s_hz = angle_trans_to_hz(signal_inst_s,acquired_fs,inpt_dimension = False)
    imf_inst_s = instfreq_get(pt_imfs)
    imf_inst_s_hz = angle_trans_to_hz(imf_inst_s,acquired_fs,inpt_dimension = True)

    signal_inst_s_hz_expected =imfs_endpoint_processing(signal_inst_s_hz,expected_length) 
    imf_inst_s_hz_expected =imfs_endpoint_processing(imf_inst_s_hz,expected_length,True) 

    ax=plt.subplot(pt_num,2,2)
    
    plt.plot(signal_inst_s_hz_expected,'r')
    plt.ylim([min(signal_inst_s_hz_expected)*1.000, max(signal_inst_s_hz_expected)*1.000])
    plt.xlim([0,expected_length])
    plt.setp(ax.get_xticklabels(), visible=False)
    plt.title("Instantaneous Frequencies/Hz")
#    plt.ylabel("inst freq")
    for tsk_i in range(1,pt_num):
         ax=plt.subplot(pt_num,2,2*(tsk_i+1))
         plt.plot(imf_inst_s_hz_expected[(tsk_i-1)],'r')
         imf_inst_name = "imf" + str(tsk_i)+ "inst freq"
         plt.ylim([min(imf_inst_s_hz_expected[(tsk_i -1)]), max(imf_inst_s_hz_expected[(tsk_i -1)])])
         plt.xlim([0,expected_length])
         plt.setp(ax.get_xticklabels(), visible=False)
#         plt.ylabel(imf_inst_name)
    plt.setp(ax.get_xticklabels(), visible=True)
    plt.xlim([0,expected_length/acquired_fs])
    plt.xlabel("Time/s") 

#    plt.title("Signal and Its IMFs                              instfreq")

#    plot_cimfs(pt_imfs)

dust_sig = org_sig_get_excel('../data/20180330/1.5.xlsx')
dust_sig1 = org_sig_get_excel('../data/20180330/2.5.xlsx')
dust_sig2 = org_sig_get_excel('../data/20180330/4.5.xlsx')
dust_sig_input_segmet_filtered1 =noise_filter(dust_sig[100000:107000])
emd_dec_instfreq_analysis(dust_sig_input_segmet_filtered1,expected_length = 6000,signal_name = "signal")
dust_sig_input_segmet_filtered2 =noise_filter(dust_sig1[100000:107000])
emd_dec_instfreq_analysis(dust_sig_input_segmet_filtered2,expected_length = 6000,signal_name = "signal")
dust_sig_input_segmet_filtered3 =noise_filter(dust_sig2[100000:107000])
emd_dec_instfreq_analysis(dust_sig_input_segmet_filtered3,expected_length = 6000,signal_name = "signal")

plt.show()

