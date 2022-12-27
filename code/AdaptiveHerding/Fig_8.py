# -*- coding: utf-8 -*-
"""
----------------------------------------------------------------------------
This file runs the simulations that yield to Figure 8
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
from functions import buildSystemMassive, calculateF, calculateHStatic
from functions import buildJxStaticMassive, buildJuStaticMassive, saturation, applyLimits

plt.close("all")
    
""" Definitions """
# Define number of evaders and herders
N_herders = 5
M_evaders = 50

# Define desired position for the evaders 
xD0 = np.array([-3.5, -3.5])
xD1 = np.array([-4.5, -6.5])
xD2 = np.array([-7.0, -3.0])

# Define initial position of the evaders 
x     = np.array([])
x0    = np.array([2.0,0.0])
bool1 = np.array([],dtype=bool)
bool2 = np.array([],dtype=bool)
for i in range(M_evaders-1):
    x = np.concatenate((x,1.0*np.random.normal([1,2])),axis=0)
    x0 = np.concatenate((x0,np.array([2.0,0.0])),axis=0)
    bool1 = np.concatenate((bool1,np.array([True,False],dtype=bool)))
    bool2 = np.concatenate((bool2,np.array([False,True],dtype=bool)))
meanx = np.mean(x,where=bool1)
meany = np.mean(x,where=bool2)
x = np.concatenate((np.array([meanx,meany]),x),axis=0)
x = x + x0

# Define initial position of the herders 
u = np.array([])
for i in range(N_herders):
    u = np.concatenate((u,x[0:2]+np.array([7*np.cos(2*np.pi/N_herders*i),7*np.sin(2*np.pi/N_herders*i)])))
integral = np.zeros([2*N_herders])
kp = 0.001
ki = 0.001

"""

IDs, what type of evader is each one:

    * 0 = Inverse Model
    * 1 = Exponential Model    
    
""" 
ID = np.zeros(M_evaders)
      
# Define hiperparameters  of the simulation
T   = 0.01
t   = 200
Sim = int(np.ceil(t/T)) 
timeAxis = np.linspace(0,t,Sim)

# Define limits of the stage 
xMax = 100
xMin = -100
yMax = 100
yMin = -100

# Define maximum velocity of the entities
vMax = 0.4

# Define parameters 
params = {
            "0": 1.2*np.ones(2*M_evaders), # Represents real parameters
            "1": 1.2*np.ones(2*M_evaders), # Represents estimates
            "beta": 0.5,
            "tau": 1.0,
            "sigma": 2.0
          } 
paramsS = {
            "0": 1.2*np.ones(2), # Represents real parameters
            "1": 1.2*np.ones(2), # Represents estimates
            "beta": 0.5,
            "tau": 1.0,
            "sigma": 2.0
          } 

# Define control gains 
K      = 0.25*np.eye(2)
Kstar  = 50*np.eye(2)

# Numeric parameters
numericParams = {
                    "itMax": 20, # Max iterations
                    "tolAbs": 1e-3, # Bound to stop iterating
                    "lambda": 0.1 # LM parameter
                  }

# Define storing variables 
P   = np.zeros([2*M_evaders,Sim])
PX  = np.zeros([4,Sim])
PD  = np.zeros([4,Sim])
HP  = np.zeros([2*N_herders,Sim])

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
        
    # Calculate real f(x,u) 
    A,B = buildSystemMassive(x,u,params,ID,0)
    xDot = calculateF(x,u,A,B)
    
    if k<6000:
        
        # Calculate estimated f(x,u) 
        A,B  = buildSystemMassive(x[0:2],u,paramsS,ID,1)
        f    = calculateF(x[0:2],u,A,B)
        
        # Calculate h(x,u)
        h = calculateHStatic(x[0:2],xD0,f,K)
        
        # Calculate h*
        hD = -np.matmul(Kstar,h)
        
        # Calculate Jacobians
        Jx = buildJxStaticMassive(x[0:2],xD0,u,paramsS,ID,1,K)
        Ju = buildJuStaticMassive(x[0:2],xD0,u,paramsS,ID,1,K)
        
        # Calculate udot: udot = pseudoinverse(Ju)*(-Kstar*h - Jx*f)
        uDot = np.matmul(np.transpose(Ju),np.matmul(np.linalg.inv(np.matmul(Ju,np.transpose(Ju))),(hD - np.matmul(Jx,f))))
        
    else:
        
        if k == 6000:
            # Compute different herds
            closest_to_xd1 = 0
            closest_to_xd2 = 0
            distance_to_xd1 = 10000
            distance_to_xd2 = 10000
            for ii in range(M_evaders):
                dist_xd1 = np.linalg.norm(xD1-x[2*ii:2*ii+2])
                dist_xd2 = np.linalg.norm(xD2-x[2*ii:2*ii+2])
                if distance_to_xd1 > dist_xd1:
                    distance_to_xd1 = dist_xd1
                    closest_to_xd1  = ii
                if distance_to_xd2 > dist_xd2:
                    distance_to_xd2 = dist_xd2
                    closest_to_xd2  = ii
            x_new = np.zeros(2*M_evaders)
            xDot_new = np.zeros(2*M_evaders)
            counter1 = 0
            for ii in range(M_evaders):
                dist_xd1 = np.linalg.norm(x[2*closest_to_xd1:2*closest_to_xd1+2]-x[2*ii:2*ii+2])
                dist_xd2 = np.linalg.norm(x[2*closest_to_xd2:2*closest_to_xd2+2]-x[2*ii:2*ii+2])
                if dist_xd1 < dist_xd2:
                    counter1 = counter1 + 1
            counter2 = 0
            counter3 = 0
            for ii in range(M_evaders):
                dist_xd1 = np.linalg.norm(x[2*closest_to_xd1:2*closest_to_xd1+2]-x[2*ii:2*ii+2])
                dist_xd2 = np.linalg.norm(x[2*closest_to_xd2:2*closest_to_xd2+2]-x[2*ii:2*ii+2])
                if dist_xd1 < dist_xd2:
                    x_new[2*counter2:2*counter2+2] = x[2*ii:2*ii+2]
                    xDot_new[2*counter2:2*counter2+2] = xDot[2*ii:2*ii+2]
                    counter2 = counter2 + 1
                else:
                    x_new[2*(counter1 + counter3):2*(counter1 + counter3)+2] = x[2*ii:2*ii+2]
                    xDot_new[2*(counter1 + counter3):2*(counter1 + counter3)+2] = xDot[2*ii:2*ii+2]
                    counter3 = counter3 + 1
            x = np.copy(x_new)        
            xDot = np.copy(xDot_new)

        # Split and compute centroid
        x1 = x[0:2*counter1]
        x2 = x[2*counter1:2*M_evaders]
        
        meanx1   = np.mean(x1[2:],where=bool1[2:2*counter1])
        meany1   = np.mean(x1[2:],where=bool2[2:2*counter1])
        x1[0]    = meanx1
        x1[1]    = meany1
        
        meanx2   = np.mean(x2[2:],where=bool1[2*counter1:2*M_evaders])
        meany2   = np.mean(x2[2:],where=bool2[2*counter1:2*M_evaders])
        x2[0]    = meanx2
        x2[1]    = meany2
        
        # Calculate estimated f(x,u) 
        A1,B1  = buildSystemMassive(x1[0:2],u,paramsS,ID,1)
        f1     = calculateF(x1[0:2],u,A1,B1)
        A2,B2  = buildSystemMassive(x2[0:2],u,paramsS,ID,1)
        f2     = calculateF(x2[0:2],u,A2,B2)
        
        f = np.concatenate((f1,f2),axis=0)
        
        # Calculate h(x,u)
        h1 = calculateHStatic(x1[0:2],xD1,f1,K)
        h2 = calculateHStatic(x2[0:2],xD2,f2,K)
        
        h = np.concatenate((h1,h2),axis=0)
        
        # Calculate h*
        hD1 = -np.matmul(Kstar,h1)
        hD2 = -np.matmul(Kstar,h2)
        
        hD = np.concatenate((hD1,hD2),axis=0)
        
        # Calculate Jacobians
        Jx1 = buildJxStaticMassive(x1[0:2],xD1,u,paramsS,ID,1,K)
        Ju1 = buildJuStaticMassive(x1[0:2],xD1,u,paramsS,ID,1,K)
        Jx2 = buildJxStaticMassive(x2[0:2],xD2,u,paramsS,ID,1,K)
        Ju2 = buildJuStaticMassive(x2[0:2],xD2,u,paramsS,ID,1,K)
        
        Jx = np.concatenate((np.concatenate((Jx1,np.zeros([2,2])),axis=1),np.concatenate((np.zeros([2,2]),Jx2),axis=1)),axis=0)
        Ju = np.concatenate((Ju1,Ju2),axis=0)
        
        # Calculate udot: udot = pseudoinverse(Ju)*(-Kstar*h - Jx*f)
        uDot = np.matmul(np.transpose(Ju),np.matmul(np.linalg.inv(np.matmul(Ju,np.transpose(Ju)) + 0.001*np.eye(Ju.shape[0])),(hD - np.matmul(Jx,f))))
        
    # Apply saturations
    xDot = saturation(xDot, vMax)
    uDot = saturation(uDot, vMax)
    
    # Calculate next u and estimated parameters
    x           += T*xDot
    u           += T*uDot
        
    # Apply limits
    applyLimits(x,xMax,xMin,yMax,yMin)
    applyLimits(u,xMax,xMin,yMax,yMin)
    
    # Virtual centroid
    if k<=6000:
        PX[:,k] = np.transpose(np.concatenate((x[0:2],x[0:2])))
    else:      
        PX[:,k] = np.transpose(np.concatenate((x1[0:2],x2[0:2])))

    if k<=6000:
        meanx   = np.mean(x[2:],where=bool1)
        meany   = np.mean(x[2:],where=bool2)
        x[0]    = meanx
        x[1]    = meany
    
    P[:,k]  = np.transpose(x)
    if k < 6000:
        PD[:,k] = np.transpose(np.concatenate((xD0,xD0)))        
    else:
        PD[:,k] = np.transpose(np.concatenate((xD1,xD2)))
    HP[:,k] = np.transpose(u)
    
    # Plot iterations
    if np.mod(k,1000)==0:
        print("Simulation run is "+str(k))

""" Plots"""
# Set plot parameters
plt.rcParams.update({'font.size': 28})

# Evolution of the game
instant = -1
fig1, ax1 = plt.subplots()
for i in range(M_evaders):
    if i>0:
        if ID[i]==0:
            ax1.plot(P[2*i,0]*10,P[2*i+1,0]*10,'gv',markersize=6,markerfacecolor='g')
            ax1.plot(P[2*i,instant]*10,P[2*i+1,instant]*10,'m^',markersize=6,markerfacecolor='m')
        else:
            ax1.plot(P[2*i,0]*10,P[2*i+1,0]*10,'gv',markersize=6,markerfacecolor='orange')
            ax1.plot(P[2*i,instant]*10,P[2*i+1,instant]*10,'m^',markersize=6,markerfacecolor='cyan')
            
    if np.mod(i,10)==0:
        ax1.plot(P[2*i,:instant]*10,P[2*i+1,:instant]*10,'k--',linewidth=1) 
for i in range(N_herders):
    ax1.plot(HP[2*i,0]*10,HP[2*i+1,0]*10,'gs',markersize=14,markerfacecolor='g')
    ax1.plot(HP[2*i,instant]*10,HP[2*i+1,instant]*10,'mp',markersize=14,markerfacecolor='m')
    ax1.plot(HP[2*i,:instant]*10,HP[2*i+1,:instant]*10,'-.b',linewidth=4) 

i=0
ax1.plot(P[2*i,0]*10,P[2*i+1,0]*10,'gv',markersize=14,markerfacecolor='g')
ax1.plot(P[2*i,instant]*10,P[2*i+1,instant]*10,'m^',markersize=14,markerfacecolor='m')
ax1.plot(P[2*i,:instant]*10,P[2*i+1,:instant]*10,'k',linewidth=4) 
ax1.plot(xD0[2*i]*10,xD0[2*i+1]*10,'ro',markersize=14,markerfacecolor='r')
plt.xlabel('x $[m]$') 
plt.ylabel('y $[m]$') 
plt.show() 


instant = -1
instant2  = 6000
fig1, ax1 = plt.subplots()
for i in range(N_herders):
    ax1.plot(HP[2*i,instant2]*10,HP[2*i+1,instant2]*10,'gs',markersize=14,markerfacecolor='g')
    ax1.plot(HP[2*i,instant]*10,HP[2*i+1,instant]*10,'mp',markersize=14,markerfacecolor='m')
    ax1.plot(HP[2*i,instant2:instant]*10,HP[2*i+1,instant2:instant]*10,'-.b',linewidth=4) 
for i in range(M_evaders):
    if i!=0 and i!=int(M_evaders/2):
        ax1.plot(P[2*i,0]*10,P[2*i+1,0]*10,'gv',markersize=6,markerfacecolor='g')
        ax1.plot(P[2*i,instant]*10,P[2*i+1,instant]*10,'m^',markersize=6,markerfacecolor='m')
        ax1.plot(P[2*i,:instant]*10,P[2*i+1,:instant]*10,'c--',linewidth=1) 
ax1.plot(P[0,0]*10,P[1,0]*10,'gv',markersize=14,markerfacecolor='g')
ax1.plot(P[0,instant]*10,P[1,instant]*10,'m^',markersize=14,markerfacecolor='m')
ax1.plot(P[0,:instant]*10,P[1,:instant]*10,'k',linewidth=4) 
ax1.plot(P[2*counter1,0]*10,P[2*counter1+1,0]*10,'gv',markersize=14,markerfacecolor='g')
ax1.plot(P[2*counter1,instant]*10,P[2*counter1+1,instant]*10,'m^',markersize=14,markerfacecolor='m')
ax1.plot(P[2*counter1,:instant]*10,P[2*counter1+1,:instant]*10,'k',linewidth=4)
ax1.plot(xD0[0]*10,xD0[1]*10,'ro',markersize=14,markerfacecolor='r')
ax1.plot(xD1[0]*10,xD1[1]*10,'ro',markersize=14,markerfacecolor='orange')
ax1.plot(xD2[0]*10,xD2[1]*10,'ro',markersize=14,markerfacecolor='orange')
plt.xlabel('x $[m]$') 
plt.ylabel('y $[m]$') 
plt.show() 
