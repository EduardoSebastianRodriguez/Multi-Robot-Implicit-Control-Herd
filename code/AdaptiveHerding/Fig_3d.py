# -*- coding: utf-8 -*-
"""
----------------------------------------------------------------------------
This file runs the simulations that yield to Figures 3d, 4b 
of our IEEE T-RO paper (details below)
----------------------------------------------------------------------------
Please, if you find this file useful, consider citing us: 
----------------------------------------------------------------------------
 E. Sebastián, E. Montijano and C. Sagüés, 
 “Adaptive Multirobot Implicit Control of Heterogeneous Herds,” 
 IEEE Transactions on Robotics, vol. 38, no. 6, pp. 3622-3635, 2022. 
----------------------------------------------------------------------------
 [More info at]: https://sites.google.com/unizar.es/poc-team/research/mrherding
 [Video]:        https://www.youtube.com/watch?v=U5KjP-2H1BM
 [Arxiv]:        https://arxiv.org/abs/2206.05888
----------------------------------------------------------------------------
 Eduardo Sebastián -- https://eduardosebastianrodriguez.github.io/

 Ph.D. Candidate
 Departamento de Informática e Ingeniería de Sistemas
 Universidad de Zaragoza
----------------------------------------------------------------------------
 Last modification: December 22, 2022
----------------------------------------------------------------------------
[WARNING]: The simulations may run slow. They are proofs of concept and
the functions are not implemented to be computationally efficient
----------------------------------------------------------------------------
"""
import numpy as np
import matplotlib.pyplot as plt
from functions import buildSystem, calculateF, calculateHStatic, buildTD, buildJparams
from functions import buildJxStatic, buildJuStatic, saturation, applyLimits, implicitMethodStatic 

plt.close("all")
    
""" Definitions """
# Define number of evaders and herders
N_herders = 5
M_evaders = 5

# Define desired position for the evaders 
xD = np.array([1.3, 0.5, -1.5, -0.9, -0.8, 1.1, 1.8, -0.7, 0.4, 0.9])

# Define initial position of the evaders 
x   = np.array([2.1, 0.7, -0.8, -1.4, -1.3, 1.8, 2.1, -1.3, 1.3, 1.5])

# Define initial position of the evaders 
x_prev = np.array([2.1, 0.7, -0.8, -1.4, -1.3, 1.8, 2.1, -1.3, 1.3, 1.5])

# Define initial position of the herders 
u     = np.array([-3.0, 0.0, -1.5, 3.0, 3.0, 0.0, 0.0, -3.0, 1.5, 3.0])

"""

IDs, what type of evader is each one:

    * 0 = Inverse Model
    * 1 = Exponential Model    
    
""" 
ID = np.ones(M_evaders)
      
# Define hiperparameters  of the simulation
T   = 0.01
t   = 20
Sim = int(np.ceil(t/T)) 
timeAxis = np.linspace(0,t,Sim)

# Define limits of the stage 
xMax = 10
xMin = -10
yMax = 10
yMin = -10

# Define maximum velocity of the entities
vMax = 0.4

# Define parameters 
params = {
                "0": 0.5*np.ones(2*M_evaders), # Respresents real parameters
                "1": 0.4*np.ones(2*M_evaders), # Respresents estimates
                "beta": 0.5,
                "tau": 1.0,
                "sigma": 2.0
              }

# Define control gains 
K       = 0.25*np.eye(2*M_evaders)
Kstar   = 50*np.eye(2*M_evaders)
landa   = 0.001
K_params= 200*np.eye(2*M_evaders)

# Numeric parameters
numericParams = {
                    "itMax": 20, # Max iterations
                    "tolAbs": 1e-3, # Bound to stop iterating
                    "lambda": 0.1 # LM parameter
                  }

# Define storing variables 
P   = np.zeros([2*M_evaders,Sim])
PD  = np.zeros([2*M_evaders,Sim])
HP  = np.zeros([2*N_herders,Sim])
CR  = np.zeros([2*M_evaders,Sim])
CE  = np.zeros([2*M_evaders,Sim])

# Run simulation
print("Simulation main parameters:")
print("\n Number of herders: "+str(N_herders))
print("\n Number of evaders: "+str(M_evaders))
name = []
for i in range(len(ID)):
    if ID[i] == 0:
        name.append("Inverse")
    elif ID[i] == 1:
        name.append("Exponential")
    else:
        print("ID must be 0 (Inverse) or 1 (Exponential) !")
print("\n Model of the evaders: "+ str(name))
print("\n Simulation running...")

for k in range(Sim):
    
    # Record variables
    P[:,k]  = np.transpose(x)
    PD[:,k] = np.transpose(xD)
    HP[:,k] = np.transpose(u)
    CR[:,k] = np.transpose(params["0"])
    CE[:,k] = np.transpose(params["1"])
    
    # Calculate u using implicit method for comparison
    uLM,nIter,normError = implicitMethodStatic(np.copy(x),xD,np.copy(u),params,ID,numericParams,K)
    
    # Calculate real f(x,u) 
    A,B  = buildSystem(x,u,params,ID,0)
    xDot = calculateF(x,u,A,B)
    
    # Calculate estimated f(x,u) 
    A,B  = buildSystem(x,u,params,ID,1)
    f    = calculateF(x,u,A,B)
        
    # Calculate h(x,u)
    h = calculateHStatic(x,xD,f,K);
    
    # Calculate h*
    hD = -np.matmul(Kstar,h)
    
    # Calculate \theta*
    tD = buildTD(xDot,f,K_params)

    # Calculate Jacobians
    Jx = buildJxStatic(x,xD,u,params,ID,1,K)
    Ju = buildJuStatic(x,xD,u,params,ID,1,K)
    Jt = buildJparams(x,u,params,ID)
    
    # Calculate udot: udot = pseudoinverse(Ju)*(-Kstar*h - Jx*f)
    uDot = np.matmul(np.transpose(Ju),np.matmul(np.linalg.inv(np.matmul(Ju,np.transpose(Ju))+landa*np.eye(2*N_herders)),(hD - np.matmul(Jx,f))))
    
    # Calculate params_dot: params_dot = pseudoinverse(J)*(-Kstar*h - Jx*f)
    tDot = -np.matmul(np.matmul(np.linalg.inv(np.matmul(np.transpose(Jt),Jt) + landa*np.eye(M_evaders)),np.transpose(Jt)),tD + np.matmul(Jx,xDot) + np.matmul(Ju,uDot))
    
    # Apply saturations
    xDot = saturation(xDot, vMax)
    uDot = saturation((uLM-u)/T, vMax)
    
    # Calculate next u and estimated parameters
    x_prev       = x
    x           += T*xDot
    u           += T*uDot
    params["1"] += T*tDot.repeat(2,0)
        
    # Apply limits
    applyLimits(x,xMax,xMin,yMax,yMin)
    applyLimits(u,xMax,xMin,yMax,yMin)
    
    # Plot iterations
    if np.mod(k,100)==0:
        print("Simulation run is "+str(k))

""" Plots"""
# Set plot parameters
plt.rcParams.update({'font.size': 28})

# Evolution of the game
fig1, ax1 = plt.subplots()
for i in range(M_evaders):
    ax1.plot(xD[2*i],xD[2*i+1],'ro',markersize=22,markerfacecolor='r')
for i in range(N_herders):
    ax1.plot(HP[2*i,0],HP[2*i+1,0],'g',marker=(5, 0, 180),markersize=14,markerfacecolor='g')
    ax1.plot(HP[2*i,-1],HP[2*i+1,-1],'mp',markersize=14,markerfacecolor='m')
    ax1.plot(HP[2*i,:],HP[2*i+1,:],'-.b',linewidth=4) 
for i in range(M_evaders):
    ax1.plot(P[2*i,0],P[2*i+1,0],'gv',markersize=14,markerfacecolor='g')
    ax1.plot(P[2*i,-1],P[2*i+1,-1],'m^',markersize=14,markerfacecolor='m')
    ax1.plot(P[2*i,:],P[2*i+1,:],'k',linewidth=4) 
plt.xlabel('x $[m]$') 
plt.ylabel('y $[m]$')
ax1.set_xlim([-4.4, 4.4])
ax1.set_ylim([-4.4, 4.4])
plt.show() 

# Evolution of the states with time
fig2, ax2 = plt.subplots() 
for i in range(M_evaders):
    ax2.plot(timeAxis,(P[2*i,:]-PD[2*i,:]),'b',linewidth=4)
    ax2.plot(timeAxis,(P[2*i+1,:]-PD[2*i+1,:]),'b',linewidth=4)
plt.xlabel('time $[s]$') 
plt.ylabel('$x-x^*$ $[m]$') 
ax2.set_xlim([0, t])
ax2.set_ylim([-1.2, 1.2])
plt.show() 