# -*- coding: utf-8 -*-
"""
----------------------------------------------------------------------------
This file runs the simulations that yield to Figure 6a
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
from functions import buildSystem, calculateF, calculateHStatic, buildTD
from functions import buildJxStatic, buildJuStatic, saturation, applyLimits 
from functions import buildJparams, measurement_complete, o_dkf_extended_complete
from scipy.optimize import linear_sum_assignment

plt.close("all")
    
""" Definitions """
# Define number of evaders and herders
N_herders = 5
M_evaders = 5

# Define desired position for the evaders 
xD     = np.array([1.3, 0.5, -1.5, -0.9, -0.8, 1.1, 1.8, -0.7, 0.4, 0.9])
new_xD = [np.copy(xD) for i in range(N_herders)]
for i in range(N_herders):
    np.random.shuffle(new_xD[i].reshape([M_evaders,2]))
    new_xD[i] = new_xD[i].reshape(2*M_evaders)

# Define initial position of the evaders 
x   = np.array([2.1, 0.7, -0.8, -1.4, -1.3, 1.8, 2.1, -1.3, 1.3, 1.5])

# Define initial position of the herders 
u     = np.array([-3.0, 0.0, -1.5, 3.0, 3.0, 0.0, 0.0, -3.0, 1.5, 3.0])
u_new = np.array([-3.0, 0.0, -1.5, 3.0, 3.0, 0.0, 0.0, -3.0, 1.5, 3.0])

"""

IDs, what type of evader is each one:

    * 0 = Inverse Model
    * 1 = Exponential Model    
    
""" 
ID = np.zeros(M_evaders)
      
# Define hiperparameters  of the simulation
T   = 0.01
t   = 30.0
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
params = [{
            "0": 1.0*np.ones(2*M_evaders), # Represents real parameters
            "1": 1.2*np.ones(2*M_evaders) # Represents estimates
          } for i in range(N_herders)]

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

# Sensor models, Covariance noise sensor matrices and Measurements 
H = [] ; R = [] ; z = []; measureID = [] ; L = []; estimates = []; estimates_new = []
maxNoise           = 7e-2
minNoise           = 7e-2
Q                  = 2e-2*np.eye(2*(N_herders+M_evaders))
sensorRange        = 6.5
communicationRange = 6.5
for i in range(N_herders):
    H.append(np.eye(2))
    R.append(np.random.uniform(minNoise,maxNoise)*np.eye(2))
    z.append(np.zeros(2*(N_herders+M_evaders)))
    measureID.append(np.zeros(N_herders+M_evaders))
    L.append(np.zeros(N_herders))
    estimates.append(np.random.uniform(-6,6,2*(N_herders+M_evaders)))
    estimates_new.append(np.zeros(2*(N_herders+M_evaders)))
    
# Covariance error matrices
covariance = [] ; covariance_new = []
for i in range(N_herders):
    Psqrt = np.random.uniform(-5,5,[2*(N_herders+M_evaders),2*(N_herders+M_evaders)])
    covariance.append(Psqrt.T @ Psqrt)
    covariance_new.append(Psqrt.T @ Psqrt)

# Define storing variables 
P   = np.zeros([2*M_evaders,Sim])
PD  = np.zeros([2*M_evaders,Sim,N_herders])
HP  = np.zeros([2*N_herders,Sim])
CR  = np.zeros([2*M_evaders,Sim,N_herders])
CE  = np.zeros([2*M_evaders,Sim,N_herders])
SS  = np.zeros([2*(M_evaders+N_herders),Sim,N_herders])
SS2 = np.zeros([2*(M_evaders+N_herders),Sim,N_herders])

# Control mode
control_mode = np.zeros(N_herders)

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
    HP[:,k] = np.transpose(u)
    for i in range(N_herders):
        CR[:,k,i]  = np.transpose(params[i]["0"])
        CE[:,k,i]  = np.transpose(params[i]["1"])
        SS[:,k,i]  = np.transpose(np.concatenate((x,u),0)-estimates[i])**2
        SS2[:,k,i] = np.transpose(estimates[i])
        PD[:,k,i]  = np.transpose(new_xD[i])
        
    # Calculate real f(x,u) 
    A,B  = buildSystem(x,u,params[0],ID,0)
    xDot = calculateF(x,u,A,B)
    
    # For each herder, measure
    for i in range(N_herders):
                
        # Read sensors
        z[i], L[i], measureID[i] = measurement_complete(u[2*i:2*i+2], x, u, H[i], R[i], sensorRange,communicationRange)
    
    # For each herder, compute estimates and control
    for i in range(N_herders):
        
        if np.mod(k,10)==0:
            com = True
        else:
            com = False
            
        # Do distributed estimation
        estimates_new[i], covariance_new[i] = o_dkf_extended_complete(estimates[i], L[i], H, R, 
                                                                      covariance[i], Q, estimates, z, 
                                                                      measureID, N_herders, M_evaders, new_xD[i], 
                                                                      params[i], ID, 1, K, Kstar, u, T, vMax,0*xDot,com)
        # Extract estimates of x and u
        x_estimated = np.zeros(2*M_evaders)
        u_estimated = np.zeros(2*N_herders)
        for j in range(2*M_evaders):
            x_estimated[j] = estimates_new[i][j]
        for j in range(2*N_herders):
            u_estimated[j] = estimates_new[i][j+2*M_evaders]
            
        # Choose control mode according to uncertainty in the estimation
        # if np.abs((np.trace(covariance[i])-np.trace(covariance_new[i]))/np.trace(covariance[i])) < 0.02:
        if k > 2:
            control_mode[i] = 1
        
        # Move or not depending on the control mode
        if control_mode[i] == 0:
            uDot = np.zeros(2*N_herders)
            tDot = np.zeros(M_evaders)
            
            # Do hungarian method
            cost = np.zeros([M_evaders,M_evaders])
            for ii in range(M_evaders):
                for jj in range(M_evaders):
                    cost[ii,jj] = np.linalg.norm(x_estimated[2*ii:2*ii+2]-xD[2*jj:2*jj+2])
            row_index, col_index = linear_sum_assignment(cost)
            new_xD[i] = np.copy(xD)
            for ii in range(M_evaders):
                new_xD[i][2*ii:2*ii+2] = xD[2*col_index[ii]:2*col_index[ii]+2]
        
        elif control_mode[i] == 1:
            # Extract estimates of x and u
            x_estimated = np.zeros(2*M_evaders)
            u_estimated = np.zeros(2*N_herders)
            for j in range(2*M_evaders):
                x_estimated[j] = estimates_new[i][j]
            for j in range(2*N_herders):
                u_estimated[j] = estimates_new[i][j+2*M_evaders]
            
            # Calculate estimated f(x,u) 
            A,B  = buildSystem(x_estimated,u_estimated,params[i],ID,1)
            f    = calculateF(x_estimated,u_estimated,A,B)
            
            # Calculate h(x,u)
            h = calculateHStatic(x_estimated,xD,f,K);
            
            # Calculate h*
            hD = -np.matmul(Kstar,h)
            
            # Calculate \theta*
            tD = buildTD(xDot,f,K_params)
    
            # Calculate Jacobians 
            Jx = buildJxStatic(x_estimated,xD,u_estimated,params[i],ID,1,K)
            Ju = buildJuStatic(x_estimated,xD,u_estimated,params[i],ID,1,K)
            Jt = buildJparams(x_estimated,u_estimated,params[i],ID)
            
            # Calculate udot: udot = pseudoinverse(Ju)*(-Kstar*h - Jx*f)
            uDot = np.matmul(np.transpose(Ju),np.matmul(np.linalg.inv(np.matmul(Ju,np.transpose(Ju))+landa*np.eye(2*N_herders)),(hD - np.matmul(Jx,f))))
            
            # Calculate params_dot: params_dot = pseudoinverse(J)*(-Kstar*h - Jx*f)
            tDot = -np.matmul(np.matmul(np.linalg.inv(np.matmul(np.transpose(Jt),Jt) + landa*np.eye(M_evaders)),np.transpose(Jt)),tD + np.matmul(Jx,xDot) + np.matmul(Ju,uDot))
            
            # Apply saturations
            uDot   = saturation(uDot, vMax)
        
        # Calculate next u and estimated parameters
        u_new[2*i:2*i+2] += T*uDot[2*i:2*i+2]
        params[i]["1"]   += T*tDot.repeat(2,0)
            
    # Apply saturations 
    xDot   = saturation(xDot, vMax)
    
    # Calculate x
    x += T*xDot
    
    # Apply limits
    applyLimits(x,xMax,xMin,yMax,yMin)
    
    # Plot iterations
    if np.mod(k,10)==0:
        print("Simulation run is "+str(k))
        
    # Apply limits
    applyLimits(u_new,xMax,xMin,yMax,yMin)
    
    # Copy new u and estimates
    u          = np.copy(u_new)
    estimates  = np.copy(estimates_new)
    covariance = np.copy(covariance_new)

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

# Evolution of the RMSE between estimated augmented state and real augmented state with time
fig5, ax5 = plt.subplots() 
for j in range(2*M_evaders):
    if j==0:
        ax5.plot(timeAxis,np.sqrt(np.mean(SS[j,:,:],1)), 'b', linewidth=4, label="$states$")
    else:
        ax5.plot(timeAxis,np.sqrt(np.mean(SS[j,:,:],1)), 'b', linewidth=4)
for j in range(2*N_herders):
    if j==0:
        ax5.plot(timeAxis,np.sqrt(np.mean(SS[j+2*M_evaders,:,:],1)), 'r', linewidth=4, label="$inputs$")
    else:
        ax5.plot(timeAxis,np.sqrt(np.mean(SS[j+2*M_evaders,:,:],1)), 'r', linewidth=4)
plt.xlabel('time $[s]$') 
plt.ylabel('Mean $ RMSE $ $[m]$') 
plt.yscale("log")
plt.xscale("log")
ax5.set_xlim([0, t])
ax5.legend(fontsize=20)
plt.show()