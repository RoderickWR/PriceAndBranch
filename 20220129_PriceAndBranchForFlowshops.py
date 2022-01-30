# -*- coding: utf-8 -*-
"""
Created on Fri Jan 28 18:36:02 2022

@author: WallrathR
"""



# ### Price and Branch ###

# In[ ]:


from gurobipy import *
from random import randrange
import random
import numpy as np
import matplotlib.pyplot as plt
import math



class Master:
    
    def __init__(self,initPatterns,initProcessingTimes,n,m):      
        self.lamb  = {}
        self.alphaCons = {}
        self.betaCons = {}
        self.gammaCons = {}
        self.omegaCons = {}
        self.s = {}
        self.f = {}
        self.c_max = {}
        self.x = {}
        self.patterns = initPatterns
        self.processing_times = initProcessingTimes
        
        self.bigM = 100
        self.model = None
        self.createMaster(n,m)
        
        
    def createMaster(self,n,m):

        ### 1) CREATE MASTER
        self.model = Model("Bin Packing Price and Branch")
        self.model.params.outputflag = 0
        self.model.modelSense = GRB.MINIMIZE
        
        # Create lambda variables for these patterns.
        for (key, value) in self.patterns.items():
            for i in range(0,m):
                self.lamb[key,i] = self.model.addVar(vtype=GRB.CONTINUOUS, name="lambda(%s,%s)"%(key,i)) #is pattern p used on machine i

        # for p in range(len(patterns)):
        #     for i in range(0,m):
        #         self.lamb[p,i] = self.model.addVar(vtype=GRB.CONTINUOUS, name="lambda(%s,%s)"%(p,i)) #is pattern p used on machine i
        
       
        for i in range(0,m):  
            self.alphaCons["ConvexityOnMachine(%s)"%(i)] = self.model.addConstr( quicksum( self.lamb[key,i] for (key,value) in self.patterns.items()) == 1, "ConvexityOnMachine(%s)"%(i)) # only one pattern per machine
          
         
        for i in range(0,m):
            for j in range(0,n):
                self.s[i,j] = self.model.addVar(vtype=GRB.CONTINUOUS, name="start(%s,%s)"%(i,j))
                self.f[i,j] = self.model.addVar(vtype=GRB.CONTINUOUS, name="finish(%s,%s)"%(i,j))
        
        #Create order matrix

        for i in range(0,m):
            for j in range(0,n):
                for k in range(0,n):
                    self.x[j,k,i] = self.model.addVar(vtype=GRB.CONTINUOUS, name="x(%s,%s,%s)"%(j,k,i)) 
        
        
        for i in range(0,m):   
            for j in range(0,n):
                for k in range(0,n):
                    if k != j: 
                        self.model.addConstr(self.x[j,k,i] + self.x[k,j,i] == 1 , "Assign3(%s)"%(j))
        
        for i in range(0,m):
        
            for k in range(0,n):
                for j in range(0,n):
                    self.omegaCons["JobOrderOnMachine(%s,%s,%s)"%(k,j,i)] = self.model.addConstr(self.f[i,k] <= self.s[i,j] + self.bigM*(1-self.x[k,j,i])) # for each job k the finishing date one machine i has to be smaller than the starting date of the next job j, (1) if j follows k on i, (2) if job k was not the cutoff job (last job) on i 
        
        
             
        self.model.update()
        
        # We need to store the created constraints to be able to columns to them later on.
        # Define master
        

        for i in range(0,m):
            for j in range(0,n):
                self.betaCons["Start(%s,%s)"%(i,j)] = self.model.addConstr(quicksum(self.patterns[key][0][j]*self.lamb[key,i] for (key,value) in self.patterns.items()) == self.s[i,j]) #starting time on machine i for job j is determined by the starting time of job j in the selected pattern p
                self.gammaCons["Finish(%s,%s)"%(i,j)] = self.model.addConstr(quicksum(self.patterns[key][1][j]*self.lamb[key,i] for (key,value) in self.patterns.items()) == self.f[i,j]) #completion time on machine i for job j is determined by the completion time of job j in the selected pattern p
            if i != m-1:
                for j in range(0,n):
                    self.model.addConstr(self.f[i,j] <= self.s[i+1,j], "InterMachine(%s,%s)"%(i,j))
            for j in range(0,n):   
                self.model.addConstr(self.s[i,j] + self.processing_times[j,i] <= self.f[i,j], 
                        "StartFinish(%s,%s)"%(i,j)) 
        
        
        #define makespan
        self.c_max = self.model.addVar(vtype=GRB.CONTINUOUS, name="Makespan", obj=1.0)
        for j in range(0,n):
            self.model.addConstr(self.c_max >= self.f[m-1,j], "MakespanConstr")
        
        
        self.model.update()
        self.model.optimize()
        
        
        
          
    def updateMaster(self):
        
        #clear the global cons then rewrite them given the new dict of patterns and lambdas 
        self.model.remove(self.model.getConstrs())
        self.alphaCons = {}
        self.betaCons = {}
        self.gammaCons = {}   
        self.omegaCons = {} 
       
        for i in range(0,m):  
            # alphaCons["ConvexityOnMachine(%s)"%(i)] = model.addConstr( quicksum( lamb[p,i] for p in range(len(patterns))) == 1, "ConvexityOnMachine(%s)"%(i)) # only one pattern per machine
            self.alphaCons["ConvexityOnMachine(%s)"%(i)] = self.model.addConstr( quicksum( value for key, value in self.lamb.items() if key[1] == i) == 1, "ConvexityOnMachine(%s)"%(i)) # only one pattern per machine
          
            
        self.model.update()
        
        # We need to store the created constraints to be able to columns to them later on.
        # Define master
        

        for i in range(0,m):
            for j in range(0,n):
                #old: betaCons["Start(%s,%s)"%(i,j)] = model.addConstr(quicksum(patterns[p][0][j]*lamb[p,i] for p in range(len(patterns))) == s[i,j]) #starting time on machine i for job j is determined by the starting time of job j in the selected pattern p
              
                self.betaCons["Start(%s,%s)"%(i,j)] = self.model.addConstr(quicksum(self.patterns[key][0][j]*self.lamb[key,i] for (key,value) in self.patterns.items()) == self.s[i,j]) #starting time on machine i for job j is determined by the starting time of job j in the selected pattern p
            
                self.gammaCons["Finish(%s,%s)"%(i,j)] = self.model.addConstr(quicksum(self.patterns[key][1][j]*self.lamb[key,i] for (key,value) in self.patterns.items()) == self.f[i,j]) #completion time on machine i for job j is determined by the completion time of job j in the selected pattern p
            if i != m-1:
                for j in range(0,n):
                    self.model.addConstr(self.f[i,j] <= self.s[i+1,j], "InterMachine(%s,%s)"%(i,j))
            for j in range(0,n):   
                self.model.addConstr(self.s[i,j] + self.processing_times[j,i] <= self.f[i,j], 
                        "StartFinish(%s,%s)"%(i,j)) 
        
        for i in range(0,m):
        
            for k in range(0,n):
                for j in range(0,n):
                    self.omegaCons["JobOrderOnMachine(%s,%s,%s)"%(k,j,i)] = self.model.addConstr(self.f[i,k] <= self.s[i,j] + self.bigM*(1-self.x[k,j,i])) # for each job k the finishing date one machine i has to be smaller than the starting date of the next job j, (1) if j follows k on i, (2) if job k was not the cutoff job (last job) on i 

        
        
        
        #define makespan
        self.c_max = {}
        self.c_max = self.model.addVar(vtype=GRB.CONTINUOUS, name="Makespan", obj=1.0)
        for j in range(0,n):
            self.model.addConstr(self.c_max >= self.f[m-1,j], "MakespanConstr")
        
        
        self.model.update()



class Pricing:
    
    def __init__(self,initProcessingTimes,i):
        self.s = {}
        self.f = {}
        self.x = {}
        self.processing_times = initProcessingTimes
        self.machineIndex = i
        self.bigM = 26
        self.pricing = None
        self.createPricing()
        

        
        
        
    def createPricing(self):
        
        
        ### 2) CREATE PRICING 
        
        self.pricing = Model("Pricing")
        self.pricing.params.outputflag = 0
        self.pricing.modelSense = GRB.MINIMIZE
                           
        
        for j in range(0,n):
            self.s[j] = self.pricing.addVar(vtype=GRB.CONTINUOUS, name="start(%s)"%(j))
            self.pricing.addConstr(self.s[j] <= 100)
            self.pricing.addConstr(self.s[j] >= 0)
            self.f[j] = self.pricing.addVar(vtype=GRB.CONTINUOUS, name="finish(%s)"%(j))
            self.pricing.addConstr(self.f[j] <= 100)
            self.pricing.addConstr(self.f[j] >= 0)
            self.pricing.addConstr(self.s[j] + self.processing_times[j,self.machineIndex] <= self.f[j], "StartFinish(%s)"%(j)) 
         
            
        #Create order matrix
        for j in range(0,n):
            for k in range(0,n):
                self.x[j,k] = self.pricing.addVar(vtype=GRB.BINARY, name="x(%s,%s)"%(j,k)) 



        for j in range(0,n):
            for k in range(0,n):
                if k != j: 
                    self.pricing.addConstr(self.x[j,k] + self.x[k,j] == 1 , "Precedence(%s)"%(j))

        for k in range(0,n):
            for j in range(0,n):
                self.pricing.addConstr(self.f[k] <= self.s[j] + self.bigM*(1-self.x[k,j]), "FinishStart(%s)"%(k)) # for each job k the finishing date one machine i has to be smaller than the starting date of the next job j, (1) if j follows k on i, (2) if job k was not the cutoff job (last job) on i 
     
            
         
        self.pricing.update()
            

            

        
class Optimizer:
    
    def __init__(self,initPatterns,initProcessingTimes,n,m): 
        self.numberJobs = n
        self.numberMachines = m
        self.s = {}
        self.f = {}
        self.x = {}
        self.processing_times = initProcessingTimes
        self.patterns = initPatterns
        
        self.pricingList = self.createPricingList()
        
        self.master = Master(initPatterns,initProcessingTimes,n,m)
        
        
        
    def createPricingList(self):
        pricingList = {}
        for i in range(0,self.numberMachines):
            pricing = Pricing(self.processing_times,i)
            pricingList["pricing(%s)"%i] = pricing
            
        
        return pricingList

    #Draw a Gantt chart with the current master solution
    def Gantt(self):
        # x_array = restructureX(x,m,n) #input x dictionary from solved model, output x numpy array
        fig = plt.figure()
        M = ['red', 'blue', 'yellow', 'orange', 'green', 'moccasin', 'purple', 'pink', 'navajowhite', 'Thistle',
             'Magenta', 'SlateBlue', 'RoyalBlue', 'Aqua', 'floralwhite', 'ghostwhite', 'goldenrod', 'mediumslateblue',
             'navajowhite','navy', 'sandybrown']
        M_num=0
        for i in range(self.numberMachines):
            for j in range(self.numberJobs):
    
                Start_time= self.master.model.getVarByName("start(%s,%s)"%(i,j)).x
                End_time= self.master.model.getVarByName("finish(%s,%s)"%(i,j)).x
                
                # Job=np.nonzero(x_array[j,:,i] == 1 )[0][0] # for each machine and each job position, find the job that takes this position
                Job = j
                plt.barh(i, width=End_time - Start_time, height=0.8, left=Start_time, \
                         color=M[Job], edgecolor='black')
                plt.text(x=Start_time + ((End_time - Start_time) / 2 - 0.25), y=i - 0.2,
                         s=Job+1, size=15, fontproperties='Times New Roman')
                M_num += 1
        plt.yticks(np.arange(M_num + 1), np.arange(1, M_num + 2), size=8, fontproperties='Times New Roman')
        plt.xticks(np.arange(0, self.master.model.objVal +1, 1.0), size=8, fontproperties='Times New Roman')
        plt.ylabel("machine", size=20, fontproperties='SimSun')
        plt.xlabel("time", size=20, fontproperties='SimSun')
        plt.tick_params(labelsize=20)
        plt.tick_params(direction='in')
        plt.show()

    # retrieve a pattern from modelIN
    def retrieveXMatrix(self,modelIN):
        matrix = np.zeros((self.numberJobs,self.numberJobs))
        mat = []
        mat = [[modelIN.getVarByName("start(%s)"%(j)).x for j in range(0,self.numberJobs)],[modelIN.getVarByName("finish(%s)"%(j)).x for j in range(0,self.numberJobs)]]
        
        return mat  


    
##                
    

    def cleanPatterns(self, threshold):  
        #delete the entries in patterns and lamb, that violate threshold
        
        # create a new lamb dict with new keys using the old lamb dict 
        self.master.lamb = {key: value for (index, (key, value)) in enumerate(self.master.lamb.items()) if not max(self.patterns[key[0]][1]) > threshold}  
        
        #checks whether the largest finish time of all jobs in a pattern p is bigger than threshold
        self.patterns = {key: value for (key,value) in self.patterns.items() if not max(value[1]) > threshold}
        
        try:
            self.master.patterns = self.patterns
        except: 
            print("self.master.patterns cannot be updated.")
        
                        

    def loopMasterPricing(self):

        # Pricing loop
        
        while True:
    
          # It looks like the loop runs forever, but we'll later use `break' to leave it.
        
          # First, solve the (restricted) LP relaxation.
          
          self.master.model.update()
          self.master.model.optimize()
          
          print('Solved restricted master LP with', self.master.model.numVars, 'columns. Optimum is', self.master.model.objVal)
        
          # We retrieve the duals for the constraints...
          alpha = {}
          beta = {}
          gamma = {}
          omega = {}
          for i in range(0,m):
              alpha["ConvexityOnMachine(%s)"%(i)] = self.master.alphaCons["ConvexityOnMachine(%s)"%(i)].pi
              print('alpha["ConvexityOnMachine(%s)"%(i)]: ' , alpha["ConvexityOnMachine(%s)"%(i)])
              for j in range(0,n):
                  beta["Start(%s,%s)"%(i,j)] = self.master.betaCons["Start(%s,%s)"%(i,j)].pi
                  print('beta["Start(%s,%s)"%(i,j)]: ', beta["Start(%s,%s)"%(i,j)])                  
                  gamma["Finish(%s,%s)"%(i,j)] = self.master.gammaCons["Finish(%s,%s)"%(i,j)].pi
                  print('gamma["Finish(%s,%s)"%(i,j)]: ', gamma["Finish(%s,%s)"%(i,j)])
                  
                  for k in range(0,n):
                      omega["JobOrderOnMachine(%s,%s,%s)"%(j,k,i)] = self.master.omegaCons["JobOrderOnMachine(%s,%s,%s)"%(j,k,i)].pi
                      print(' omega["JobOrderOnMachine(%s,%s,%s)"%(j,k,i)]: ', omega["JobOrderOnMachine(%s,%s,%s)"%(j,k,i)] )
          # ... and use them as weights completion time problem.
          nbrPricingOpt = 0
          for i in range(0,m):
              pricing = self.pricingList["pricing(%s)"%i]
              for j in range(0,n):
                  pricing.pricing.getVarByName("start(%s)" %(j)).obj = -beta["Start(%s,%s)"%(i,j)]
                  print('pricing.pricing.getVarByName("start(%s)" %(j)).obj: ', pricing.pricing.getVarByName("start(%s)" %(j)).obj)
                  pricing.pricing.getVarByName("finish(%s)" %(j)).obj = -gamma["Finish(%s,%s)"%(i,j)]
                  print('pricing.pricing.getVarByName("finish(%s)" %(j)).obj: ', pricing.pricing.getVarByName("finish(%s)" %(j)).obj)
              
                  for k in range(0,n):
                      pricing.pricing.getVarByName("x(%s,%s)"%(j,k)).obj = omega["JobOrderOnMachine(%s,%s,%s)"%(j,k,i)]
                      print("pricing.getVarByName(x(%s,%s)%(j,k)).obj: ", pricing.pricing.getVarByName("x(%s,%s)"%(j,k)).obj)
                    
                
              # #additional constraints for new patterns during loop
              # for j in range(0,n):
              #     pricing.addConstr(pricing.getVarByName("finish(%s)" %(j)) <= model.getVarByName("finish(%s,%s)" %(i,j)).x)    
                  
                  
              # We solve the pricing problem.
              pricing.pricing.optimize()
              print("Solution of pricing:" , self.retrieveXMatrix(pricing.pricing))
              counter = 0
              if pricing.pricing.status == GRB.OPTIMAL and  pricing.pricing.objVal - alpha["ConvexityOnMachine(%s)"%(i)] < 0:
                  # If the Knapsack solution is good enough, we add the column.
                newPattern = self.retrieveXMatrix(pricing.pricing)
                self.patterns[len(self.patterns)] = newPattern
                print("and ", newPattern, " is added")
                print('Appended pattern: ', counter)
            
                # We now create columns (#m because lambda has dimension [p,m]) to be added to the (restricted) LP relaxation of the main problem.
                    
                for i in range(0,m):
                    col = Column()
                    col.addTerms(1,self.master.alphaCons["ConvexityOnMachine(%s)"%(i)])
                    for j in range(0,n):
                        col.addTerms(newPattern[0][j], self.master.betaCons["Start(%s,%s)"%(i,j)])
                        col.addTerms(newPattern[1][j], self.master.gammaCons["Finish(%s,%s)"%(i,j)])
                    # We create the lambda variable with this column.
                    self.master.lamb[len(self.patterns) - 1,i] = self.master.model.addVar(name="lambda(%s,%s))"%(len(self.patterns) - 1,i), column=col)
        
                counter += 1
        
              if pricing.pricing.objVal - alpha["ConvexityOnMachine(%s)"%(i)] >= 0:
                nbrPricingOpt += 1
                
          print("nbrPricingOpt: ", nbrPricingOpt)  
          if nbrPricingOpt == m:
    
              # BRANCHING 1) restict second machine pricing problem, 2) delete patterns that violate this restiction, 3) delete the respective lambdas in master
              for j in range(0,n):
                  self.pricingList["pricing(%s)"%(1)].pricing.addConstr(self.pricingList["pricing(%s)"%(1)].pricing.getVarByName("finish(%s)" %(j))<= 100 )
               
              self.cleanPatterns(100)
              self.master.updateMaster()
    
              break
        
    
        
    # End of pricing loop.
    
    def solveIPCompact(self):
     
         for key, value in self.master.lamb.items():
             # print("key, value", key , value)
             value.vtype = GRB.BINARY
    
         self.master.model.update()
         self.master.model.params.outputflag = 1
         self.master.model.optimize()
        
         # [print("v.x: ", v.x) for v in model.getVars()]
         #test 
         if self.master.model.status == GRB.OPTIMAL:
           print('Optimal solution:')
           for key, value in self.master.lamb.items():
               if value.x >= 0.5:
                   print('Pattern p', value, 'is used.')
                   
         self.Gantt()

#PARAMS 
n=2 # number of jobs
m=2 # number of machines
processing_times = np.array([[7,1],[1,7]]) #job 1 takes 7 hours on machine 1, and 1 hour on machine 2, job 2 takes 1 hour on machine 1, and 7 hours on machine 2


# We start with only randomly generated patterns.
# pattern 1 is[[0,7],[7,8]]. The structure is [[start time job 1, start time job 2,...],[compl time job 1, compl time job 2,...]]
patterns = {0: [[0,7],[7,8]], 1: [[0,0],[16,60]], 2: [[14,16],[0,0]] }

opt = Optimizer(patterns,processing_times, n, m)

opt.loopMasterPricing()

opt.solveIPCompact()




print()
print('Number of patterns via column generation is', len(patterns))
print()
print('=============================================================================')
print()




# In[ ]:




