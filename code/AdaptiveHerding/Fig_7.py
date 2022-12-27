# -*- coding: utf-8 -*-
"""
----------------------------------------------------------------------------
This file runs the simulations that yield to Figure 7
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
the functions are not implemented to be computationally efficient.
----------------------------------------------------------------------------
"""
import numpy as np
import matplotlib.pyplot as plt
from functions import buildSystem, calculateF, saturation 
from functions import getEllipse, measurement_complete, o_dkf_extended_complete
from functions import functional, calculateHTimeVarying, buildTD, buildJxTimeVarying
from functions import buildJuTimeVarying, buildJparams
from scipy.optimize import newton
import cvxpy as cp

plt.close("all")
    
""" Definitions """
# Define number of evaders and herders
N_herders = 3
M_evaders = 3

# Define desired initial position for the evaders 
xD     = np.array([1.3, 0.5, -1.5, -0.9, -0.8, 1.1])
new_xD = [np.copy(xD) for i in range(N_herders)]
    
# Define vector for time-varying desired position for the evaders 
xB        = np.copy(new_xD)
xDdot     = [np.zeros([2*M_evaders]) for i in range(N_herders)]

# Define initial position of the evaders 
x   = np.array([2.1, 0.7, -0.8, -1.4, -1.3, 1.8])

# Define initial position of the herders 
u     = np.array([-10.5, -10.0, 10.0, 10.0, 10.0, 9.0])
u_new = np.array([-10.5, -10.0, 10.0, 10.0, 10.0, 9.0])
uDot  = np.ones(2*N_herders)


"""

IDs, what type of evader is each one:

    * 0 = Inverse Model
    * 1 = Exponential Model    
    
""" 
ID = np.array([0,0,1])
    
# Define hiperparameters of the simulation
T   = 0.01
t   = 175
Sim = int(np.ceil(t/T)) 

# Time Varying trajectory
tTV = 325
SimTV = int(np.ceil(tTV/T)) 
timeAxis = np.linspace(0,t,Sim+SimTV)

# Define maximum velocity of the entities
vMax   = 0.4
vDx    = 0.02
wDy    = np.array([0.05, 0.1, 0.02])
refAmp = 0.5

# Define parameters 
params = [{
            "0": np.array([1.0,1.0,1.0,1.0,0.5,0.5]), # Respresents real parameters
            "1": np.array([1.2,1.2,1.2,1.2,0.4,0.4]), # Respresents estimates
            "beta": 0.5,
            "tau": 1.0,
            "sigma": 2.0
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
maxNoise           = 8e-2
minNoise           = 6e-2
Q                  = 2e-2*np.eye(2*(N_herders+M_evaders))
sensorRange        = 20.0
communicationRange = 20.0
for i in range(N_herders):
    H.append(np.eye(2))
    R.append(np.random.uniform(minNoise,maxNoise)*np.eye(2))
    z.append(np.zeros(2*(N_herders+M_evaders)))
    measureID.append(np.zeros(N_herders+M_evaders))
    L.append(np.zeros(N_herders))
    estimates.append(np.random.uniform(-5,5,2*(N_herders+M_evaders)))
    estimates_new.append(np.zeros(2*(N_herders+M_evaders)))
    
# Covariance error matrices
covariance = [] ; covariance_new = []
for i in range(N_herders):
    Psqrt = np.random.uniform(-5,5,[2*(N_herders+M_evaders),2*(N_herders+M_evaders)])
    covariance.append(Psqrt.T @ Psqrt)
    covariance_new.append(Psqrt.T @ Psqrt)
    
# Define storing variables 
P   = np.zeros([2*M_evaders,Sim+SimTV])
PD  = np.zeros([2*M_evaders,Sim+SimTV,N_herders])
HP  = np.zeros([2*N_herders,Sim+SimTV])
CR  = np.zeros([2*M_evaders,Sim+SimTV,N_herders])
CE  = np.zeros([2*M_evaders,Sim+SimTV,N_herders])
SS  = np.zeros([2*(M_evaders+N_herders),Sim+SimTV,N_herders])
SS2 = np.zeros([2*(M_evaders+N_herders),Sim+SimTV,N_herders])

# Control mode
control_mode = np.zeros(N_herders)

# Approaching
x_avg = np.zeros(2)
x_std = np.zeros([2,2])
for i in range(M_evaders):
    x_avg += x[2*i:2*i+2]/M_evaders
for i in range(M_evaders):
    x_std[0,0] += (x[2*i]-x_avg[0])*(x[2*i]-x_avg[0])/M_evaders
    x_std[0,1] += (x[2*i]-x_avg[0])*(x[2*i+1]-x_avg[1])/M_evaders
    x_std[1,0] += (x[2*i+1]-x_avg[1])*(x[2*i]-x_avg[0])/M_evaders
    x_std[1,1] += (x[2*i+1]-x_avg[1])*(x[2*i+1]-x_avg[1])/M_evaders
radius = 6.0
goals, agoal, bgoal, Vgoal = getEllipse(x_avg, x_std, N_herders+1, radius)

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

for k in range(Sim+SimTV):
    
    """ Build time varying reference and dynamics of x*(t):
        
             x_dot_j(t) = v_j
             y_dot_j(t) = refAmp * w_j * cos( w_j*t + 2*pi/j )
    
    """
    if (k>Sim and k<(SimTV+Sim-200)):
        for j in range(N_herders):
            for i in range(M_evaders):
                new_xD[j][2*i]  += T*vDx
                new_xD[j][2*i+1] = xB[j][2*i+1] + refAmp*np.sin(wDy[i]*k*T + 2*np.pi/(i+1))
                xDdot[j][2*i]    = vDx
                xDdot[j][2*i+1]  = wDy[i]*refAmp*np.cos(wDy[i]*k*T + 2*np.pi/(i+1))
    elif k>(SimTV+Sim-200):
        for j in range(N_herders):
            for i in range(M_evaders):
                xDdot[j][2*i]   = 0
                xDdot[j][2*i+1] = 0
            
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
    
    for i in range(N_herders):
        
        # Choose control mode according to uncertainty in the estimation and stage of the game
        if np.linalg.norm(uDot) < 1e-4:
            control_mode = np.ones(2*N_herders)
        
        # Move or not depending on the control mode
        uDot = np.zeros(2*N_herders)
        tDot = np.zeros(M_evaders)
        
        if control_mode[i] == 0:
            # Approach near the evaders
            thetas = np.zeros(N_herders)
            for ii in range(N_herders):
                thetas[ii] = np.mod(np.arctan2(u[2*ii+1],u[2*ii]),2*np.pi)
            theta_min = 0
            theta_max = 0
            theta_cen = 0
            order     = thetas.argsort() 
            for ii in range(N_herders):
                if order[ii] == i:
                    theta_cen = thetas[order[ii]]
                    if ii+1 > N_herders-1:
                        theta_max = thetas[order[0]]
                    else:
                        theta_max = thetas[order[ii+1]] 
                    if ii-1 < 0:
                        theta_min = thetas[order[N_herders-1]]
                    else:
                        theta_min = thetas[order[ii-1]]
            d1 = theta_min - theta_max
            d2 = 2*np.pi - np.abs(d1)
            if d1 > 0:
                d2 = -d2
            if np.abs(d1) < np.abs(d2):
                error = np.abs(d1)
                if theta_max > theta_min:
                    sense = 1
                else:
                    sense = -1
            else:
                error = np.abs(d2)
                if theta_max > theta_min:
                    sense = -1
                else:
                    sense = 1
            error = sense*error
            if error < 0:
                error += 2*np.pi
            
            theta_goal = theta_min + error/2
            
            d1 = theta_cen - theta_goal
            d2 = 2*np.pi - np.abs(d1)
            if d1 > 0:
                d2 = -d2
            if np.abs(d1) < np.abs(d2):
                error = np.abs(d1)
                if theta_goal > theta_cen:
                    sense = 1
                else:
                    sense = -1
            else:
                error = np.abs(d2)
                if theta_goal > theta_cen:
                    sense = -1
                else:
                    sense = 1
            theta_dot = sense*error
            theta_cen+= T*theta_dot
            theta_cen = np.mod(theta_cen,2*np.pi)
            state     = Vgoal @ np.array([[agoal*np.cos(theta_cen)],
                                  [bgoal*np.sin(theta_cen)]])
            
            diameterD    = np.linalg.norm(state)
            diameter     = np.linalg.norm(u[2*i:2*i+2])
            diameter_dot = diameterD - diameter
            diameter    += T*diameter_dot
            
            herder_goal = np.array([diameter*np.cos(theta_cen),diameter*np.sin(theta_cen)])
            speed       = (herder_goal-u[2*i:2*i+2])/T
            
            # Apply CBF to surround the group of evaders
            # Instantiate optimization variables 
            uOpt = cp.Variable(2,name="uOpt") 
            
            # Objective function
            objective = cp.Minimize(cp.quad_form(uOpt-speed,np.eye(2)))
              
            # Constraints
            radius         = 2
            goals, a, b, V = getEllipse(x_avg, x_std, 20, radius)
            point          = np.matmul(V,u[2*i:2*i+2])
            try:
                theta      = newton(functional,np.arctan2(a*point[1],b*point[0]),args=(a,b,point[0],point[1],))
            except:
                print('fail')
            point          = np.matmul(V,np.array([a*np.cos(theta),b*np.sin(theta)]))
            
            """ Lfh(x)+Lgh(x)u+gamma*h**3(x)>=0 """
            rel_pos        = point - u[2*i:2*i+2]
            h_cbf          = np.linalg.norm(rel_pos) + T*((rel_pos.reshape([1,2])@speed.reshape([2,1]))[0,0])/np.linalg.norm(rel_pos) - 1
            alpha_function = 50*(h_cbf**3)
            Lf             = -(point[0]-u[2*i])/np.linalg.norm(point - u[2*i:2*i+2])*u[2*i] - \
                              (point[1]-u[2*i+1])/np.linalg.norm(point - u[2*i:2*i+2])*u[2*i+1]
            Lg             = -(point[0]-u[2*i])/np.linalg.norm(point - u[2*i:2*i+2])*T*uOpt[0] - \
                              (point[1]-u[2*i+1])/np.linalg.norm(point - u[2*i:2*i+2])*T*uOpt[1]
            constraints    = [Lf + Lg + alpha_function>=0]
            
            # Create optimization problem
            problem = cp.Problem(objective,constraints)
            
            # Solve it
            problem.solve(solver=cp.MOSEK)
            
            # Get input
            uDot[2*i:2*i+2] = uOpt.value.reshape(2)
            saturation(uDot[2*i:2*i+2], vMax)
            tDot            = np.zeros(M_evaders)
            
        elif control_mode[i] == 1:
            # Drive evaders to assigned positions
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
            
            # Calculate estimated f(x,u) 
            A,B  = buildSystem(x_estimated,u_estimated,params[i],ID,1)
            f    = calculateF(x_estimated,u_estimated,A,B)
            
            # Calculate h(x,u)
            h = calculateHTimeVarying(x_estimated,new_xD[i],f,K,xDdot[i])
            
            # Calculate h*
            hD = -np.matmul(Kstar,h)
            
            # Calculate theta*
            tD = buildTD(xDot,f,K_params)
    
            # Calculate Jacobians 
            Jx = buildJxTimeVarying(x_estimated,new_xD[i],u_estimated,params[i],ID,1,K,xDdot[i])
            Ju = buildJuTimeVarying(x_estimated,new_xD[i],u_estimated,params[i],ID,1,K,xDdot[i])
            Jt = buildJparams(x_estimated,u_estimated,params[i],ID)
            
            # Calculate udot: udot = pseudoinverse(Ju)*(-Kstar*h - Jx*f)
            uDot = np.matmul(np.transpose(Ju),np.matmul(np.linalg.inv(np.matmul(Ju,np.transpose(Ju))+landa*np.eye(2*N_herders)),(hD - np.matmul(Jx,f))))
            
            # Calculate params_dot: params_dot = pseudoinverse(J)*(-Kstar*h - Jx*f)
            tDot = -np.matmul(np.matmul(np.linalg.inv(np.matmul(np.transpose(Jt),Jt) + landa*np.eye(M_evaders)),np.transpose(Jt)),tD + np.matmul(Jx,xDot) + np.matmul(Ju,uDot))
            
            # Apply saturations
            uDot = saturation(uDot, vMax)
        else:
            uDot = np.zeros(2*N_herders)
        
        # Calculate next u and estimated parameters
        u_new[2*i:2*i+2] += T*uDot[2*i:2*i+2]
        params[i]["1"]   += T*tDot.repeat(2,0)
            
    # Apply saturations 
    xDot   = saturation(xDot, vMax)
    
    # Calculate x
    x += T*xDot
               
    # Plot iterations
    if np.mod(k,10)==0:
        print("Simulation run is "+str(k))
        
    # Copy new u and estimates
    u          = np.copy(u_new)
    estimates  = np.copy(estimates_new)
    covariance = np.copy(covariance_new)


""" Plots"""
# Set plot parameters
plt.rcParams.update({'font.size': 28})

# Evolution of the whole game
fig1, ax1 = plt.subplots()
for i in range(M_evaders):
    if ID[i] == 0:
        ax1.plot(P[2*i,0:40000],P[2*i+1,0:40000],'m',linewidth=4) 
        ax1.plot(PD[2*i,0:40000],PD[2*i+1,0:40000],'-.m',linewidth=4)
    else:
        ax1.plot(P[2*i,0:40000],P[2*i+1,0:40000],'r',linewidth=4) 
        ax1.plot(PD[2*i,0:40000],PD[2*i+1,0:40000],'-.r',linewidth=4)
for i in range(N_herders):
    ax1.plot(HP[2*i,0:40000],HP[2*i+1,0:40000],'-.b',linewidth=4) 
for i in range(N_herders):
    ax1.plot(HP[2*i,0],HP[2*i+1,0],'g',marker=(5, 0, 180),markersize=14,markerfacecolor='g')
    ax1.plot(HP[2*i,40000],HP[2*i+1,40000],'mp',markersize=14,markerfacecolor='m')
for i in range(M_evaders):
    ax1.plot(P[2*i,0],P[2*i+1,0],'gv',markersize=14,markerfacecolor='g')
    ax1.plot(P[2*i,40000],P[2*i+1,40000],'m^',markersize=14,markerfacecolor='m')  
plt.xlabel('x $[m]$') 
plt.ylabel('y $[m]$') 
plt.show() 

# Evolution of the 1st part of the game
fig11, ax11 = plt.subplots()
stop = 5500
for i in range(M_evaders):
    if ID[i] == 0:
        ax11.plot(P[2*i,0:stop],P[2*i+1,0:stop],'m',linewidth=4) 
        ax11.plot(PD[2*i,0:stop],PD[2*i+1,0:stop],'-.m',linewidth=4)
    else:
        ax11.plot(P[2*i,0:stop],P[2*i+1,0:stop],'r',linewidth=4) 
        ax11.plot(PD[2*i,0:stop],PD[2*i+1,0:stop],'-.r',linewidth=4)
for i in range(N_herders):
    ax11.plot(HP[2*i,0:stop],HP[2*i+1,0:stop],'-.b',linewidth=4) 
for i in range(N_herders):
    ax11.plot(HP[2*i,0],HP[2*i+1,0],'g',marker=(5, 0, 180),markersize=14,markerfacecolor='g')
    ax11.plot(HP[2*i,stop],HP[2*i+1,stop],'mp',markersize=14,markerfacecolor='m')
for i in range(M_evaders):
    ax11.plot(P[2*i,0],P[2*i+1,0],'gv',markersize=14,markerfacecolor='g')
    ax11.plot(P[2*i,stop],P[2*i+1,stop],'m^',markersize=14,markerfacecolor='m')
plt.xlabel('x $[m]$') 
plt.ylabel('y $[m]$') 
plt.show() 

# Evolution of the 2nd part of the game
fig12, ax12 = plt.subplots()
stop2 = 17500
for i in range(N_herders):
    ax12.plot(HP[2*i,stop:stop2],HP[2*i+1,stop:stop2],'-.b',linewidth=4) 
for i in range(M_evaders):
    if ID[i] == 0:
        ax12.plot(P[2*i,stop:stop2],P[2*i+1,stop:stop2],'m',linewidth=4) 
        ax12.plot(PD[2*i,stop:stop2],PD[2*i+1,stop:stop2],'-.m',linewidth=4)
    else:
        ax12.plot(P[2*i,stop:stop2],P[2*i+1,stop:stop2],'r',linewidth=4) 
        ax12.plot(PD[2*i,stop:stop2],PD[2*i+1,stop:stop2],'-.r',linewidth=4)
for i in range(N_herders):
    ax12.plot(HP[2*i,stop],HP[2*i+1,stop],'g',marker=(5, 0, 180),markersize=14,markerfacecolor='g')
    ax12.plot(HP[2*i,stop2],HP[2*i+1,stop2],'mp',markersize=14,markerfacecolor='m')
for i in range(M_evaders):
    ax12.plot(P[2*i,stop],P[2*i+1,stop],'gv',markersize=10,markerfacecolor='g')
    ax12.plot(P[2*i,stop2],P[2*i+1,stop2],'w^',markersize=10,markerfacecolor='m')
plt.xlabel('x $[m]$') 
plt.ylabel('y $[m]$') 
plt.show() 

# Evolution of the 3rd part of the game
fig13, ax13 = plt.subplots()
stop3 = 40000
for i in range(N_herders):
    ax13.plot(HP[2*i,stop2:stop3],HP[2*i+1,stop2:stop3],'-.b',linewidth=4) 
for i in range(M_evaders):
    if ID[i] == 0:
        ax13.plot(P[2*i,stop2:stop3],P[2*i+1,stop2:stop3],'m',linewidth=4) 
        ax13.plot(PD[2*i,stop2:stop3],PD[2*i+1,stop2:stop3],'-.m',linewidth=4)
    else:
        ax13.plot(P[2*i,stop2:stop3],P[2*i+1,stop2:stop3],'r',linewidth=4) 
        ax13.plot(PD[2*i,stop2:stop3],PD[2*i+1,stop2:stop3],'-.r',linewidth=4)
for i in range(N_herders):
    ax13.plot(HP[2*i,stop2],HP[2*i+1,stop2],'g',marker=(5, 0, 180),markersize=14,markerfacecolor='g')
    ax13.plot(HP[2*i,stop3],HP[2*i+1,stop3],'mp',markersize=14,markerfacecolor='m')
for i in range(M_evaders):
    ax13.plot(P[2*i,stop2],P[2*i+1,stop2],'gv',markersize=14,markerfacecolor='g')
    ax13.plot(P[2*i,stop3],P[2*i+1,stop3],'m^',markersize=14,markerfacecolor='m')
plt.xlabel('x $[m]$') 
plt.ylabel('y $[m]$') 
plt.show() 

# Evolution of the 4th part of the game
fig14, ax14 = plt.subplots()
for i in range(N_herders):
    ax14.plot(HP[2*i,0],HP[2*i+1,0],'gs',markersize=14,markerfacecolor='g')
    ax14.plot(HP[2*i,-1],HP[2*i+1,-1],'mp',markersize=14,markerfacecolor='m')
    ax14.plot(HP[2*i,Sim+int(np.ceil(2*SimTV/3)):-1],HP[2*i+1,Sim+int(np.ceil(2*SimTV/3)):-1],'-.b',linewidth=4) 
for i in range(M_evaders):
    ax14.plot(P[2*i,0],P[2*i+1,0],'gv',markersize=14,markerfacecolor='g')
    ax14.plot(P[2*i,-1],P[2*i+1,-1],'m^',markersize=14,markerfacecolor='m')
    if ID[i] == 0:
        ax14.plot(P[2*i,Sim+int(np.ceil(2*SimTV/3)):-1],P[2*i+1,Sim+int(np.ceil(2*SimTV/3)):-1],'m',linewidth=4) 
        ax14.plot(PD[2*i,Sim+int(np.ceil(2*SimTV/3)):-1],PD[2*i+1,Sim+int(np.ceil(2*SimTV/3)):-1],'-.m',linewidth=4)
    else:
        ax14.plot(P[2*i,Sim+int(np.ceil(2*SimTV/3)):-1],P[2*i+1,Sim+int(np.ceil(2*SimTV/3)):-1],'r',linewidth=4) 
        ax14.plot(PD[2*i,Sim+int(np.ceil(2*SimTV/3)):-1],PD[2*i+1,Sim+int(np.ceil(2*SimTV/3)):-1],'-.r',linewidth=4)
plt.xlabel('x $[m]$') 
plt.ylabel('y $[m]$') 
plt.show() 