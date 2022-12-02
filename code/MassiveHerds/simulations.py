# -*- coding: utf-8 -*-
"""
This file contains the code to run the simulations in:
    
    E. Sebastián, E. Montijano, C. Sagüés, "Multi-robot Implicit Control of Massive Herds"

Current Version: 2022-11-30

Eduardo Sebastián Rodríguez, PhD Student / esebastian@unizar.es / https://eduardosebastianrodriguez.github.io/
Department of Computer Science and Systems Engineering / diis.unizar.es
University of Zaragoza / unizar.es
"""

import numpy as np
import matplotlib.pyplot as plt
from functions import buildSystem, calculateF, calculateH_static, dynamicAssignment
from functions import buildJx_static, buildJu_static, saturation, calculateH_dynamic, buildJx_dynamic, buildJu_dynamic

plt.close("all")
    
""" Definitions """
# Define number of preys and hunters
N_hunters = 3
M_preys   = 20

# Define initial position of the preys (in meters)
x     = np.array([])
x0    = np.array([])
bool1 = np.array([],dtype=bool)
bool2 = np.array([],dtype=bool)
for i in range(M_preys):
    x     = np.concatenate((x,1.0*np.random.normal([1,2])),axis=0)
    x0    = np.concatenate((x0,np.array([2.0,0.0])),axis=0)
    bool1 = np.concatenate((bool1,np.array([True,False],dtype=bool)))
    bool2 = np.concatenate((bool2,np.array([False,True],dtype=bool)))
x      = x + x0

# Define initial position of the hunters (in meters)
u = np.array([])
for i in range(N_hunters):
    u = np.concatenate((u,x[0:2]+np.array([7*np.cos(2*np.pi/N_hunters*i),7*np.sin(2*np.pi/N_hunters*i)])))

# IDs, what type of prey is each one: 0 = Inverse Model, 1 = Exponential Model    
ID = np.ones(M_preys)
      
# Define hiperparameters  of the simulation
T   = 0.04 # In seconds
t   = 400 # In seconds
Sim = int(np.ceil(t/T)) 
timeAxis = np.linspace(0,t,Sim)

# Define maximum velocity of the entities
vMax_evaders = 0.4
vMax_herders = 0.4

# Define parameters 
params = {
            "0": 1.2*np.ones(2*M_preys), # Represents real parameters
            "1": 1.2*np.ones(2*M_preys), # Represents estimates
            "beta": 0.5,
            "tau": 1.0,
            "sigma": 2.0,
            "gamma": 0.0002
          } 

# Define control gains and quantities
landa    = 0.001 # Help stability pseudoinverse
vDx      = -0.02 # Linear speed time-varying reference
wDy      = 0.1   # Angular speed time-varying reference
refAmp   = 0.5   # Amplitude cosine time-varying reference
k_f      = 0.25  # Control constant f*
k_h      = 50    # Control constant h*
time_ref = 4000  # Instant when the time-varying reference begins

# Define storing variables 
P   = np.zeros([2*M_preys,Sim])
PX  = np.zeros([2*N_hunters,Sim])
PD  = np.zeros([2*N_hunters,Sim])
HP  = np.zeros([2*N_hunters,Sim])

# Run simulation
print("Simulation main parameters:")
print("\n Number of herders: "+str(N_hunters))
print("\n Number of preys: "+str(M_preys))
name = []
for i in range(len(ID)):
    if ID[i] == 0:
        name.append("Inverse")
    elif ID[i] == 1:
        name.append("Exponential")
    else:
        print("ID must be 0 (Inverse) or 1 (Exponential) !")
print("\n Model of the preys: "+ str(name))
print("\n Simulation running...")

for k in range(Sim):
    
    # Calculate real f(x,u) 
    A,B  = buildSystem(x,u,params,ID,0)
    xDot = calculateF(x,u,A,B)
    
    # Compute evaders to control
    evaders = dynamicAssignment(x, N_hunters)
    
    # Compute control gains
    paramsS = {
                "0": 1.2*np.ones(evaders.shape[0]), 
                "1": 1.2*np.ones(evaders.shape[0]), 
                "beta": 0.5,
                "tau": 1.0,
                "sigma": 2.0,
                "gamma": 0.0002
              } 
    K       = k_f*np.eye(evaders.shape[0])
    Kstar   = k_h*np.eye(evaders.shape[0])
    
    # Compute reference
    if k < time_ref:
        xD      = np.tile(np.array([-3.5, -3.5]), int(evaders.shape[0]/2))
        xB      = np.copy(xD)
        new_xD  = np.copy(xD)
        xDdot   = 0*np.copy(xD)
    else:
        xD[0]   += T*vDx
        xD[1]    = xB[1] + refAmp*np.sin(wDy*k*T)
        xDdot[0] = vDx
        xDdot[1] = wDy*refAmp*np.cos(wDy*k*T)
        xD       = np.tile(xD[0:2], int(evaders.shape[0]/2))
        xDdot    = np.tile(xDdot[0:2], int(evaders.shape[0]/2))

    # Calculate estimated f(x,u) 
    A,B  = buildSystem(evaders,u,paramsS,ID,1)
    f    = calculateF(evaders,u,A,B)

    # Calculate h(x,u)
    if k < time_ref:
        h = calculateH_static(evaders,xD,f,K)
    else:
        h = calculateH_dynamic(evaders,xD,f,K,xDdot)
        
    # Calculate h*
    hD = -np.matmul(Kstar,h)

    # Calculate Jacobians
    if k < time_ref:
        Jx = buildJx_static(evaders,xD,u,paramsS,ID,1,K)
        Ju = buildJu_static(evaders,xD,u,paramsS,ID,1,K)
    else:
        Jx = buildJx_dynamic(evaders,xD,u,paramsS,ID,1,K,xDdot)
        Ju = buildJu_dynamic(evaders,xD,u,paramsS,ID,1,K,xDdot)
    
    # Calculate udot: udot = pseudoinverse(Ju)*(-Kstar*h - Jx*f)
    uDot = np.matmul(np.transpose(Ju),np.matmul(np.linalg.inv(np.matmul(Ju,np.transpose(Ju)) + landa*np.eye(evaders.shape[0])),(hD - np.matmul(Jx,f))))  

    # Apply saturations
    xDot = saturation(xDot, vMax_evaders)
    uDot = saturation(uDot, vMax_herders)
    
    # Calculate next x and u
    x += T*xDot
    u += T*uDot
    
    # Store values
    if evaders.shape[0] == 2*N_hunters: 
        PX[:,k] = np.transpose(evaders)
        PD[:,k] = np.transpose(xD)
    else:
        number = int((2*N_hunters-evaders.shape[0])/2)
        PX[:,k] = np.transpose(np.concatenate((evaders, np.tile(evaders[0:2], number))))
        PD[:,k] = np.tile(xD[0:2], N_hunters)
    P[:,k]  = np.transpose(x)
    HP[:,k] = np.transpose(u)
    
    # Plot iterations
    if np.mod(k,10)==0:
        print("Simulation run is "+str(k))

""" save data """
np.save('HP',HP)
np.save('P',P)
np.save('PD',PD)
np.save('PX',PX)

""" Plots"""
# Set plot parameters
plt.rcParams.update({'font.size': 28})

# Plot fixed-time reference
instant = 3999
fig1, ax1 = plt.subplots()
for i in range(M_preys):
    if i>0: 
        ax1.plot(P[2*i,:instant]*10,P[2*i+1,:instant]*10,'--',color="lightgreen",linewidth=1) 
        ax1.plot(P[2*i,0]*10,P[2*i+1,0]*10,'go',markersize=12,markerfacecolor='g')
        ax1.plot(P[2*i,instant]*10,P[2*i+1,instant]*10,'ko',markersize=12,markerfacecolor='k')        
for i in range(N_hunters):
    ax1.plot(HP[2*i,:instant]*10,HP[2*i+1,:instant]*10,color='powderblue',linewidth=2) 
    ax1.plot(HP[2*i,0]*10,HP[2*i+1,0]*10,'bD',markersize=14,markerfacecolor='b')
    ax1.plot(HP[2*i,instant]*10,HP[2*i+1,instant]*10,'mD',markersize=14,markerfacecolor='m')
ax1.plot(PD[0,0]*10,PD[1,0]*10,'rX',markersize=20,markerfacecolor='r')
ax1.plot(PD[0,instant]*10,PD[1,instant]*10,'rX',markersize=20,markerfacecolor='r')
ax1.plot(PD[0,:instant]*10,PD[1,:instant]*10,'r',linewidth=2)
plt.xlabel('x $[m]$') 
plt.ylabel('y $[m]$') 
plt.show() 

# Plot time-varying reference
first   = 4001
instant = -1
fig1, ax1 = plt.subplots()
for i in range(M_preys):
    if i>0: 
        ax1.plot(P[2*i,first:instant]*10,P[2*i+1,first:instant]*10,'--',color="lightgreen",linewidth=1) 
        ax1.plot(P[2*i,first]*10,P[2*i+1,first]*10,'go',markersize=12,markerfacecolor='g')
        ax1.plot(P[2*i,instant]*10,P[2*i+1,instant]*10,'ko',markersize=12,markerfacecolor='k')        
for i in range(N_hunters):
    ax1.plot(HP[2*i,first:instant]*10,HP[2*i+1,first:instant]*10,color='powderblue',linewidth=2) 
    ax1.plot(HP[2*i,first]*10,HP[2*i+1,first]*10,'bD',markersize=14,markerfacecolor='b')
    ax1.plot(HP[2*i,instant]*10,HP[2*i+1,instant]*10,'mD',markersize=14,markerfacecolor='m')
ax1.plot(PD[0,first]*10,PD[1,first]*10,'rX',markersize=20,markerfacecolor='r')
ax1.plot(PD[0,instant]*10,PD[1,instant]*10,'rX',markersize=20,markerfacecolor='r')
ax1.plot(PD[0,first:instant]*10,PD[1,first:instant]*10,'r',linewidth=2)
plt.xlabel('x $[m]$') 
plt.ylabel('y $[m]$') 
plt.show() 

# Plot of the whole experiment
first   = 0
instant = -1
fig1, ax1 = plt.subplots()
for i in range(M_preys):
    if i>0: 
        ax1.plot(P[2*i,first:instant]*10,P[2*i+1,first:instant]*10,'--',color="lightgreen",linewidth=1) 
        ax1.plot(P[2*i,first]*10,P[2*i+1,first]*10,'go',markersize=12,markerfacecolor='g')
        ax1.plot(P[2*i,instant]*10,P[2*i+1,instant]*10,'ko',markersize=12,markerfacecolor='k')        
for i in range(N_hunters):
    ax1.plot(HP[2*i,first:instant]*10,HP[2*i+1,first:instant]*10,color='powderblue',linewidth=2) 
    ax1.plot(HP[2*i,first]*10,HP[2*i+1,first]*10,'bD',markersize=14,markerfacecolor='b')
    ax1.plot(HP[2*i,instant]*10,HP[2*i+1,instant]*10,'mD',markersize=14,markerfacecolor='m')
ax1.plot(PD[0,first]*10,PD[1,first]*10,'rX',markersize=20,markerfacecolor='r')
ax1.plot(PD[0,instant]*10,PD[1,instant]*10,'rX',markersize=20,markerfacecolor='r')
ax1.plot(PD[0,first:instant]*10,PD[1,first:instant]*10,'r',linewidth=2)
plt.xlabel('x $[m]$') 
plt.ylabel('y $[m]$') 
plt.show() 