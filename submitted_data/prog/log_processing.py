# -*- coding: utf-8 -*-
import re
import os
import matplotlib.pyplot as plt

class match2Words(object):
      lines=0
      def __init__(self,path,word1,word2,word3,word4):
          self.path = path
          self.word1 = word1
          self.word2 = word2
          self.word3 = word3
          self.word4 = word4
      def key_match(self):
          with open(self.path,'rb') as f:
               buffer = f.read()
              # pattern = re.compile(self.word1+b'(.*?)'+self.word2,re.S)
               pattern = re.compile(self.word1+b'(.*?)'+self.word2,re.S)
               pattern1 = re.compile(self.word3+b'(.*?)'+self.word4,re.S)
               result1 = pattern1.sub(r'',buffer)
               result = pattern.findall(result1)
#               print len(result)
#               print float(result[0])
#               total_error = [0.0]*len(result)
               total_error = []
               train_err = []
               err_i =0;
               for tsk_i in range(0,len(result)):
                   if result[tsk_i].find('(') == -1 :
                      #total_error = total_error+[float(result[tsk_i])] 
                      #total_error.append(float(result[tsk_i])) 
                      if float(result[tsk_i] ) >= 0.12 and tsk_i >=10000:
                        train_err.append(total_error)  
                        err_i+=1
                        print len(total_error) 
                        total_error =[]
                        print result[tsk_i]
                      else:
                        total_error.append(float(result[tsk_i])) 

                   else:
                       train_err.append(total_error) 
                       err_i+=1
                       print len(total_error) 
                       total_error =[]
                       print result[tsk_i] 
               train_err.append(total_error) 
               plt.figure()
               min_length = len(train_err[0]) 
               for tsk_j in range(0,len(train_err)):
                   if min_length > len(train_err[tsk_j]):
                       min_length = len(train_err[tsk_j])
               print "min_length"
               print min_length 
               min_length = 6000
               color_sel =['r','b','g']

               for tsk_j in range(0,len(train_err)):
                   line,=plt.plot(train_err[tsk_j][0:min_length],color_sel[tsk_j])
                   legend_name = "imf order " + str(tsk_j+2)
#                   plt.legend(legend_name)
                   line.set_label(legend_name)
                   plt.ylim([0,max(train_err[tsk_j])*1.1])
#                 plt.plot(train_err[1])
#               plt.show()
#               plt.legend()
               plt.xlim([0,min_length])
#               plt.ylim([0,max(train_err)])
               plt.legend()
               plt.xlabel("Step") 
               plt.ylabel("Convergence error") 

               print tsk_i
               print result[tsk_i] 
               print len(train_err) 
               print len(train_err[0]) 
               print len(train_err[1]) 
               if result != []:
#                  print result
                   f = open('total_error1.txt', 'w+')
                   f.write(str(total_error) + os.linesep)
                   f.close()
               else:
                   print "They are no key word you have imputted."
                           
      def delete_key_word_2(self):
          with open('total_error1.txt','rb') as f:
               buffer = f.read()
               pattern = re.compile(self.word1+b'(.*?)'+self.word2,re.S)
               result = pattern.sub(buffer)
               total_error = [0.0]*len(result) 
               for tsk_i in range(0,len(result)):
                  total_error[tsk_i] = float(result[tsk_i])
               if result != []:
                  f = open('total_error.txt', 'w+')
                  f.write(str(total_error) + os.linesep)
                  f.close()
               else:
                  print "delete file is empty"
#path = input("Please input the log address:")
#word1 = b"begin"
#word2 = b"end"
path = "./log_cemmd.log"
word1 = "Total error:"
word2 = "Total error:"
word3 = "train-errors"
#word3 = "('train-errors:'"
word4 = "Total error:"
matchWords = match2Words(path, word1, word2,word3,word4)
matchWords.key_match()
plt.show()
