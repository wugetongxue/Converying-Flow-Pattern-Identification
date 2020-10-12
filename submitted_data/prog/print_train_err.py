import matplotlib.pyplot as plt
import numpy as np

train_acc =  [0.9922480620155039, 0.8643410852713178, 1.0] 
test_acc =  [0.9949494949494949, 0.8156565656565656, 1.0]

x = np.linspace(2, 4, 3)
line, =plt.plot(x,train_acc,'r-o')
line.set_label("Train Set Test Error")
#plt.plot(x,train_acc,'r')
line1,=plt.plot(x,test_acc,'b-d')
line1.set_label("Test Set Test Error")
plt.xlabel("Order of IMF")
plt.ylabel("Error Rate(%)")
#plt.plot(x,test_acc,'b')
plt.legend()
plt.show()
