# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 16:16:28 2021

@author: natma
"""

import numpy as np
from pulp import *
from scipy.optimize import minimize 

def objective(n,sign=-1.0):
    N=25
    L=150#demand curve
    f=np.zeros(4)#freshness
    p=np.array([1,0.75,0.50,0.25])#Price
    cost=max(0.25,0.5-L*0.001)#cost
    profit=0
    fine=0.05
    
    d1=1#gov variable
    d3=1#gov variable
    d2=10#gov variable
    
    #these terms were use but might not be needed if all the terms are summed together
    cum_n=0
    cum_R=0
    
    #The for loop in the non linear term
    n_diff=0
    #terms in fairshare penalty equation
    term_3=0
    term_remaining=0
    
    fareshare_penalty=0
    
    #Supermarket Utility
    Waste=0
    Profit=0
    Benefit_Fairshare=0
    
    #other option is to accumulate all together so supermarket utility will be in the for loop
    #this method is just to check if I was getting the same terms as the excel doc
    for i in range(4):
       # for j in range(4):
       #    n_diff+=(n[j]-N)
        f[i]=(4-i)/4
        term_3+=(n[i]*(1.0-f[i]))
        #term_remaining+=(d1*N)-((d2/(N**2))*(n_diff**2))#this is wrong must check with Alan
        R_pure=L*(f[i]/2.5)
        sigmoid=(L/5)*np.tanh(p[i]-0.5)
        cum_n=n[i]+cum_n
        R=min(R_pure-sigmoid,L-cum_n-cum_R)
        cum_R=R+cum_R
        waste=L-cum_R-cum_n
        profit+=(p[i]*R)
        n_diff=0
    
    #not the recommened way term_remaining should be calculated within the loop
    term_remaining=10*((N-cum_n)**2)/(N**2)
    fareshare_penalty=term_remaining+(d3*term_3)
    #the Total  will be wrong due to the fareshare_penalty term mistake
    #if you want to double check if the answer is right remove 
    #term_remaining variable in this py doc and also in the excel sheet
    #term_remaining is the non linear term in the equation
    Waste=fine*waste
    Profit=profit-(cost*L)
    #Benefit fare share will be wrong due to fareshare_penalty
    Benefit_Fairshare=N-fareshare_penalty
    Total=Profit-(Waste)+Benefit_Fairshare
#     print("Total:{}\nWaste:{}\nProfit:{}\nBenefitFairShare:{}".format(Total,Waste,Profit,Benefit_Fairshare))   
    return sign*(Total)
   
    
def constraint1(x):
    _sum=25
    for i in range(4):
        _sum=_sum-x[i]
    return _sum

def main():
     n0=np.array([1,5,5,1])#proportion of needed served by farshare
     N=25.0
     print(-1*objective(n0))
     b=(0,N)
     bnds=(b,b,b,b)
     con1={'type':'eq','fun':constraint1}
     cons=[con1]
     
     sol=minimize(objective,n0,method='SLSQP',bounds=bnds,constraints=cons)
     print(sol)
    
    
if __name__== "__main__":
    main()




