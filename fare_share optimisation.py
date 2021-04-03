# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 16:16:28 2021
@author: natma
"""
from gurobipy import *
import numpy as np

N=10#Needy people to be served

def Ufairshare(n0,n1,n2,n3,d1,d2,d3):
    
    L=150#demand curve
    f=np.zeros(4)#freshness
    p=np.array([1,0.75,0.50,0.25])#Price
    cost=max(0.25,0.5-L*0.001)#cost
    profit=0
    fine=0.05
    #proportion of needed served by farshare
    n = np.array([n0,n1,n2,n3])

    cum_n=0
    cum_R=0
    #terms in fairshare penalty equation
    term_3=0
    fareshare_penalty=0
    
    #Supermarket Utility
    Waste=0
    Profit=0
    Benefit_Fairshare=0
    

    for i in range(4):
        f[i]=(4-i)/4
        term_3+=(n[i]*(1.0-f[i]))
        R_pure=L*(f[i]/2.5)
        sigmoid=(L/5)*np.tanh(p[i]-0.5)
        cum_n=n[i]+cum_n
#        R=min(R_pure-sigmoid,L-cum_n-cum_R)
        R=L-cum_n-cum_R
        cum_R=R+cum_R
        waste=L-cum_R-cum_n
        profit+=(p[i]*R)
    
    ###########################
    expand_1=N*N
    expand_2=cum_n*cum_n
    ##############################
    expand_3=2*N*cum_n
    #################################
#    expand_5=expand_1+expand_2-expand_4
    expand_5=expand_1+expand_2-expand_3
    expand_6=d2/expand_1
    
    ##############################################
    term_2=expand_5*expand_6
    term_3=d3*term_3
    fareshare_penalty=term_2+term_3
    
    Waste=fine*waste
    Profit=profit-(cost*L)
    
    Benefit_Fairshare= (d1*N)-(fareshare_penalty)
    Total=Profit-(Waste)+Benefit_Fairshare
    return Total


def main():
    
    milp_model = Model("milp")
    
    for d2 in range (1,11):
        d1 = milp_model.addVar(vtype=GRB.INTEGER,lb=1,name="d1")
        d3 = milp_model.addVar(vtype=GRB.INTEGER,lb=1,name="d3")
        n0 = milp_model.addVar(vtype=GRB.INTEGER,lb=0,name="n0")
        n1 = milp_model.addVar(vtype=GRB.INTEGER,lb=0,name="n1")
        n2 = milp_model.addVar(vtype=GRB.INTEGER,lb=0,name="n2")
        n3 = milp_model.addVar(vtype=GRB.INTEGER,lb=0,name="n3")
        obj_fun = Ufairshare(n0,n1,n2,n3,d1,d2,d3)
        milp_model.setObjective(obj_fun,GRB.MAXIMIZE)
    
        c1 = milp_model.addConstr(n0+n1+n2+n3<=N,"c1")
        c2 = milp_model.addConstr(d1+d2+d3<=30,"c2")
        c3 = milp_model.addConstr(d1<=10,"c3")
        c5 = milp_model.addConstr(d3<=10,"c5")

     
#    milp_model.params.NonConvex = 2
        milp_model.optimize()
    
        print('Objective Function Value:%.2f' %milp_model.objVal)
    #get values
        for v in milp_model.getVars():
            print('%s:%g'%(v.varName,v.x))


        
if __name__== "__main__":
    main()
