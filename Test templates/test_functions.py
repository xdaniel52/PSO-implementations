import numpy as np
import math

def f_1(x): #sphere #(-100,100) #0 #0.01
    result = 0
    for d in range(len(x)):
        result+=np.power(x[d],2)
    return result

def f_2(x): #schwafel's p2.22 #(-10,10) #0 #0.01
    result = 0
    tmp = 1
    for d in range(len(x)):
        result+=abs(x[d])
        tmp*=abs(x[d]) 
    result+=tmp
    return result    
    
def f_3(x): #Quadric #(-100,100) #0 #100
    result = 0
    for d in range(len(x)):
        tmp = 0
        for j in range(d+1):
            tmp+=x[j]
        result+=np.power(tmp,2)
    return result    
   
def f_4(x): #Rosenbrock #(-10,10) #0 #100
    result = 0
    tmp = 0
    for d in range(len(x)-1):
        result+=np.power(x[d+1]-np.power(x[d],2),2)
        tmp+=np.power(x[d]-1,2)
    result*=100       
    return result+tmp 

def f_5(x): #Step #(-100,100) #0 #0
    result = 0
    for d in range(len(x)):
        result+=np.power(np.floor(x[d]+0.5),2)       
    return result

def f_6(x): #Quadric Noise #(-1.28,1.28) #0 #0.01
    result = 0
    for d in range(len(x)):
        result+=d*np.power(x[d],4)+np.random.rand()     
    return result

def f_7(x): #Schwefel #(-500,500) #-12569.5 #-10000
    result = 0
    for d in range(len(x)):
        result-=x[d]*math.sin(math.sqrt(abs(x[d])))    
    return result

def f_8(x): #Rastrigin #(-5.12,5.12) #0 #50
    result = 0
    for d in range(len(x)):
        result+=np.power(x[d],2)-10*math.cos(2*math.pi*x[d])+10    
    return result

def f_9(x): #Noncontinuous Rastrigin #(-5.12,5.12) #0 #50
    result = 0
    for d in range(len(x)):
        if abs(x[d]) <0.5:
            result+=np.power(x[d],2)-10*math.cos(2*math.pi*x[d])+10    
        else:
            result+=np.power(round(2*x[d])/2,2)-10*math.cos(2*math.pi*x[d])+10           
    return result