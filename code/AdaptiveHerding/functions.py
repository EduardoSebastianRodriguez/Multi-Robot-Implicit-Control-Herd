# -*- coding: utf-8 -*-
"""
----------------------------------------------------------------------------
This file contains all the functions to run the simulations
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
"""

import numpy as np
import matplotlib.pyplot as plt

def buildSystem(
                x: np.array, 
                u: np.array, 
                params: dict, 
                ID: np.array, 
                values: int
                ):
    """Compute the state space matrices of the herding problem A, B:
        
                    f(x,u) = A(x,u)x + B(x,u)u

    Arguments:
        x {np.array}  -- State of the system (position of the evaders)
        u {np.array}  -- Input of the system (position of the herders, robots)
        params {dict} -- Parameters of the evaders' models
        ID {np.array} -- Behavioural model of each evader (0=Inverse, 1=Exponential)
        values {int}  -- Real or estimated parameters (0=Real, 1=Estimated)

    Returns:
        A(x,u) {np.array} 
        B(x,u) {np.array} 
    
    """    
    # Init matrices with proper dimensions
    A  = np.zeros([len(x),len(x)])
    B  = np.zeros([len(x),len(u)])
    
    """ Create A(x,u) """
    # For each evader
    for j in range(int(len(x)/2)):
        # Check model
        if ID[j] == 0:
            for i in range(int(len(u)/2)):
                X              = np.linalg.norm(x[2*j:2*j+2]-u[2*i:2*i+2])**3
                A[2*j,2*j]     = A[2*j,2*j] + params[str(values)][2*j]/X
                A[2*j+1,2*j+1] = A[2*j+1,2*j+1] + params[str(values)][2*j]/X                    
        elif ID[j] == 1:
            var = params["beta"]*params[str(values)][2*j]
            for i in range(int(len(u)/2)):
                if np.linalg.norm(x[2*j:2*j+2]-u[2*i:2*i+2])<params["tau"]:
                    var = params[str(values)][2*j]
                    break
            for i in range(int(len(u)/2)):
                X              = np.linalg.norm(x[2*j:2*j+2]-u[2*i:2*i+2])**2
                A[2*j,2*j]     = A[2*j,2*j] + var*np.exp(-1/(params["sigma"]**2)*X)
                A[2*j+1,2*j+1] = A[2*j+1,2*j+1] + var*np.exp(-1/(params["sigma"]**2)*X)
        else:
            print("ID must be 0 (Inverse) or 1 (Exponential) !")
            return
    
    """ Create B(x,u) """
    # For each evader
    for j in range(int(len(x)/2)):
        # Check model
        if ID[j] == 0:
            for i in range(int(len(u)/2)):
                X              = np.linalg.norm(x[2*j:2*j+2]-u[2*i:2*i+2])**3
                B[2*j,2*i]     = -params[str(values)][2*j]/X
                B[2*j+1,2*i+1] = -params[str(values)][2*j]/X
        elif ID[j] == 1:
            var = params["beta"]*params[str(values)][2*j]
            for i in range(int(len(u)/2)):
                if np.linalg.norm(x[2*j:2*j+2]-u[2*i:2*i+2])<params["tau"]:
                    var = params[str(values)][2*j]
                    break
            for i in range(int(len(u)/2)):
                X              = np.linalg.norm(x[2*j:2*j+2]-u[2*i:2*i+2])**2
                B[2*j,2*i]     = -var*np.exp(-1/(params["sigma"]**2)*X)
                B[2*j+1,2*i+1] = -var*np.exp(-1/(params["sigma"]**2)*X)
        else:
            print("ID must be 0 (Inverse) or 1 (Exponential) !")
            return

    return A, B

def buildSystemMassive(
                x: np.array, 
                u: np.array, 
                params: dict, 
                ID: np.array, 
                values: int
                ):
    """Compute the state space matrices of the herding problem A, B:
        
                    f(x,u) = A(x,u)x + B(x,u)u

    The function includes a slight coalition term to enhance the realism
    of the motion model.
    
    Arguments:
        x {np.array}  -- State of the system (position of the evaders)
        u {np.array}  -- Input of the system (position of the herders, robots)
        params {dict} -- Parameters of the evaders' models
        ID {np.array} -- Behavioural model of each evader (0=Inverse, 1=Exponential)
        values {int}  -- Real or estimated parameters (0=Real, 1=Estimated)

    Returns:
        A(x,u) {np.array} 
        B(x,u) {np.array} 
    
    """    
    # Init matrices with proper dimensions
    A  = np.zeros([len(x),len(x)])
    B  = np.zeros([len(x),len(u)])
    
    """ Create A(x,u) """
    # For each evader
    for j in range(int(len(x)/2)):
        # Check model
        if ID[j] == 0:
            for i in range(int(len(u)/2)):
                X              = np.linalg.norm(x[2*j:2*j+2]-u[2*i:2*i+2])**3
                A[2*j,2*j]     = A[2*j,2*j] + params[str(values)][2*j]/X
                A[2*j+1,2*j+1] = A[2*j+1,2*j+1] + params[str(values)][2*j]/X                    
        elif ID[j] == 1:
            var = params["beta"]*params[str(values)][2*j]
            for i in range(int(len(u)/2)):
                if np.linalg.norm(x[2*j:2*j+2]-u[2*i:2*i+2])<params["tau"]:
                    var = params[str(values)][2*j]
                    break
            for i in range(int(len(u)/2)):
                X              = np.linalg.norm(x[2*j:2*j+2]-u[2*i:2*i+2])**2
                A[2*j,2*j]     = A[2*j,2*j] + var*np.exp(-1/(params["sigma"]**2)*X)
                A[2*j+1,2*j+1] = A[2*j+1,2*j+1] + var*np.exp(-1/(params["sigma"]**2)*X)
        else:
            print("ID must be 0 (Inverse) or 1 (Exponential) !")
            return
        
    for j in range(int(len(x)/2)):
            # Check model
            if ID[j] == 0:
                for i in range(int(len(x)/2)):
                    X              = np.linalg.norm(x[2*j:2*j+2]-x[2*i:2*i+2])**3
                    Y              = np.linalg.norm(x[2*j:2*j+2]-x[2*i:2*i+2])**(2)
                    if X != 0:
                        A[2*j,2*j]     = A[2*j,2*j] + 0.0002*params[str(values)][2*j]/X - 0.0001*params[str(values)][2*j]*Y
                        A[2*j+1,2*j+1] = A[2*j+1,2*j+1] + 0.0002*params[str(values)][2*j]/X - 0.0001*params[str(values)][2*j]*Y 
                        A[2*j,2*i]     = A[2*j,2*i] - 0.0002*params[str(values)][2*j]/X + 0.0001*params[str(values)][2*j]*Y
                        A[2*j+1,2*i+1] = A[2*j+1,2*i+1] - 0.0002*params[str(values)][2*j]/X + 0.0001*params[str(values)][2*j]*Y                    
            elif ID[j] == 1:
                var = params["beta"]*params[str(values)][2*j]
                for i in range(int(len(x)/2)):
                    if np.linalg.norm(x[2*j:2*j+2]-x[2*i:2*i+2])<params["tau"]:
                        var = params[str(values)][2*j]
                        break
                for i in range(int(len(x)/2)):
                    X              = np.linalg.norm(x[2*j:2*j+2]-x[2*i:2*i+2])**3
                    Y              = np.linalg.norm(x[2*j:2*j+2]-x[2*i:2*i+2])**(2)
                    if X != 0:
                        A[2*j,2*j]     = A[2*j,2*j] + 0.0002*params[str(values)][2*j]/X - 0.0001*params[str(values)][2*j]*Y
                        A[2*j+1,2*j+1] = A[2*j+1,2*j+1] + 0.0002*params[str(values)][2*j]/X - 0.0001*params[str(values)][2*j]*Y 
                        A[2*j,2*i]     = A[2*j,2*i] - 0.0002*params[str(values)][2*j]/X + 0.0001*params[str(values)][2*j]*Y
                        A[2*j+1,2*i+1] = A[2*j+1,2*i+1] - 0.0002*params[str(values)][2*j]/X + 0.0001*params[str(values)][2*j]*Y 
            else:
                print("ID must be 0 (Inverse) or 1 (Exponential) !")
                return
    
    """ Create B(x,u) """
    # For each evader
    for j in range(int(len(x)/2)):
        # Check model
        if ID[j] == 0:
            for i in range(int(len(u)/2)):
                X              = np.linalg.norm(x[2*j:2*j+2]-u[2*i:2*i+2])**3
                B[2*j,2*i]     = -params[str(values)][2*j]/X
                B[2*j+1,2*i+1] = -params[str(values)][2*j]/X
        elif ID[j] == 1:
            var = params["beta"]*params[str(values)][2*j]
            for i in range(int(len(u)/2)):
                if np.linalg.norm(x[2*j:2*j+2]-u[2*i:2*i+2])<params["tau"]:
                    var = params[str(values)][2*j]
                    break
            for i in range(int(len(u)/2)):
                X              = np.linalg.norm(x[2*j:2*j+2]-u[2*i:2*i+2])**2
                B[2*j,2*i]     = -var*np.exp(-1/(params["sigma"]**2)*X)
                B[2*j+1,2*i+1] = -var*np.exp(-1/(params["sigma"]**2)*X)
        else:
            print("ID must be 0 (Inverse) or 1 (Exponential) !")
            return

    return A, B

def calculateF(x: np.array, u: np.array, A: np.array, B: np.array):
    """Compute the time derivative of the state:
        
                    xdot = f(x,u) = A(x,u)x + B(x,u)u

    Arguments:
        x {np.array} -- State of the system (position of the evaders)
        u {np.array} -- Input of the system (position of the herders, robots)
        A {np.array} -- State matrix
        B {np.array} -- Input matrix

    Returns:
        f(x,u) {np.array}  
    
    """    
    f = np.matmul(A,x) + np.matmul(B,u)
    
    # This term makes the dynamics to be smooth when applying the saturation
    # in the velocity of the evaders
    f = 0.4/np.sqrt(2)*np.tanh(5*f)
    
    return f

def adaptiveLaw(
                x: np.array,
                xD: np.array,
                u: np.array, 
                params: dict, 
                ID: np.array, 
                S: float,
                T: float
                ):
    """Calculate the estimates with the adaptive law

    Arguments:
        x {np.array}  -- State of the system (position of the evaders)
        xD {np.array} -- Desired final state of the system 
        u {np.array}  -- Input of the system (position of the herders, robots)
        params {dict} -- Parameters of the evaders' models
        ID {np.array} -- Behavioural model of each evader (0=Inverse, 1=Exponential)
        S {float}     -- Adaptation matrix
        T {float}     -- Sample time
    
    """    
    # Create variable to allocate derivative of the estimates
    paramsDot = np.zeros(len(params[str(1)]))
    
    # For each evader, calculate the rate of change of the estimates and then the estimates
    for j in range(int(len(x)/2)):
        # Check model
        if ID[j] == 0:
            for i in range(int(len(u)/2)):
                paramsDot[2*j]   += np.matmul(x[2*j:2*j+2]-xD[2*j:2*j+2],np.reshape(x[2*j:2*j+2]-u[2*i:2*i+2],[2,1]))/(np.linalg.norm(x[2*j:2*j+2]-u[2*i:2*i+2])**3)*S
                paramsDot[2*j+1]  = paramsDot[2*j]                      
        elif ID[j] == 1:
            angry = False
            for i in range(int(len(u)/2)):
                if np.linalg.norm(x[2*j:2*j+2]-u[2*i:2*i+2])<params["tau"]:
                    angry = True
                    break
            for i in range(int(len(u)/2)):
                X              = np.linalg.norm(x[2*j:2*j+2]-u[2*i:2*i+2])**2
                if angry:
                    paramsDot[2*j] += np.matmul(x[2*j:2*j+2]-xD[2*j:2*j+2],np.reshape(x[2*j:2*j+2]-u[2*i:2*i+2],[2,1]))*np.exp(-(1/(params["sigma"]**2))*X)*S
                else:
                    paramsDot[2*j] += params["beta"]*np.matmul(x[2*j:2*j+2]-xD[2*j:2*j+2],np.reshape(x[2*j:2*j+2]-u[2*i:2*i+2],[2,1]))*np.exp(-(1/(params["sigma"]**2))*X)*S
                paramsDot[2*j+1]    = paramsDot[2*j] 
        else:
            print("ID must be 0 (Inverse) or 1 (Exponential) !")
            return  
        # Update estimates
        params[str(1)][2*j] = params[str(1)][2*j] + T*paramsDot[2*j]
        params[str(1)][2*j+1] = params[str(1)][2*j+1] + T*paramsDot[2*j+1]   
    return 


def calculateHStatic(x: np.array, xD: np.array, f: np.array, K: np.array):
    """Compute the time derivative of the state:
        
                    h(x,u) = f(x,u) - f*(x,u) = f(x,u) + K(x-x*)

    Arguments:
        x {np.array}  -- State of the system (position of the evaders)
        xD {np.array} -- Desired final state of the system
        f {np.array}  -- Dynamic of the system
        K {np.array}  -- Control matrix

    Returns:
        h(x,u) {np.array}  
    
    """    
    return f + np.matmul(K,x-xD)

def calculateHTimeVarying(x: np.array, xD: np.array, f: np.array, K: np.array, xDdot: np.array):
    """Compute the time derivative of the state:
        
                    h(x,u) = f(x,u) - f*(x,u) = f(x,u) + K(x-x*) - xDdot

    Arguments:
        x {np.array}     -- State of the system (position of the evaders)
        xD {np.array}    -- Desired state of the system
        f {np.array}     -- Dynamic of the system
        K {np.array}     -- Control matrix
        xDdot {np.array} -- Dynamics of the desired state of the system

    Returns:
        h(x,u) {np.array}  
    
    """    
    return f + np.matmul(K,x-xD) - xDdot

def buildTD(f: np.array, fE: np.array, K: np.array):
    """Compute the desired behaviour of the adaptive expansion

    Arguments:
        f {np.array}  -- Dynamics
        fE {np.array} -- Estimated dynamics 
        K {np.array}  -- Control matrix

    Returns:
        tD {np.array}  
    
    """    
    return -np.matmul(K,f-fE)

def buildJxStatic(x: np.array,
            xD: np.array,
            u: np.array,
            params: dict,
            ID: np.array,
            values: int,
            K: np.array,
            epsilon: float = 1e-6
            ):
    """Compute the Jacobian of h(x,u) with respect to the state
    when the reference is static

    Arguments:
        x {np.array}   -- State of the system (position of the evaders)
        xD {np.array}  -- Desired final state of the system
        u {np.array}   -- Input of the system (position of the herders, robots)
        params {dict}  -- Parameters of the evaders' models
        ID {np.array}  -- Behavioural model of each evader (0=Inverse, 1=Exponential)
        values {int}   -- Real or estimated parameters (0=Real, 1=Estimated)
        K {np.array}   -- Control matrix
        epsilon{float} -- tolerance for the numeric calculation

    Returns:
        Jx {np.array}  
    
    """    
    # Preallocate Jacobian
    Jx = np.zeros([len(x),len(x)])
    
    # Calculate each column of the Jacobian
    for i in range(len(x)):
        v        = np.zeros(len(x))
        v[i]     = epsilon
        A,B      = buildSystem(x+v, u, params, ID, values)
        f        = calculateF(x+v, u, A, B)
        h1       = calculateHStatic(x+v, xD, f, K)
        A,B      = buildSystem(x-v, u, params, ID, values)
        f        = calculateF(x-v, u, A, B)
        h2       = calculateHStatic(x-v, xD, f, K)     
        Jx[:,i]  = np.transpose((h1 - h2)/epsilon/2)
    
    return Jx

def buildJxStaticMassive(x: np.array,
            xD: np.array,
            u: np.array,
            params: dict,
            ID: np.array,
            values: int,
            K: np.array,
            epsilon: float = 1e-6
            ):
    """Compute the Jacobian of h(x,u) with respect to the state
    when the reference is static

    Arguments:
        x {np.array}   -- State of the system (position of the evaders)
        xD {np.array}  -- Desired final state of the system
        u {np.array}   -- Input of the system (position of the herders, robots)
        params {dict}  -- Parameters of the evaders' models
        ID {np.array}  -- Behavioural model of each evader (0=Inverse, 1=Exponential)
        values {int}   -- Real or estimated parameters (0=Real, 1=Estimated)
        K {np.array}   -- Control matrix
        epsilon{float} -- tolerance for the numeric calculation

    Returns:
        Jx {np.array}  
    
    """    
    # Preallocate Jacobian
    Jx = np.zeros([len(x),len(x)])
    
    # Calculate each column of the Jacobian
    for i in range(len(x)):
        v        = np.zeros(len(x))
        v[i]     = epsilon
        A,B      = buildSystemMassive(x+v, u, params, ID, values)
        f        = calculateF(x+v, u, A, B)
        h1       = calculateHStatic(x+v, xD, f, K)
        A,B      = buildSystemMassive(x-v, u, params, ID, values)
        f        = calculateF(x-v, u, A, B)
        h2       = calculateHStatic(x-v, xD, f, K)     
        Jx[:,i]  = np.transpose((h1 - h2)/epsilon/2)
    
    return Jx

def buildJxTimeVarying(x: np.array,
            xD: np.array,
            u: np.array,
            params: dict,
            ID: np.array,
            values: int,
            K: np.array,
            xDdot: np.array,
            epsilon: float = 1e-6
            ):
    """Compute the Jacobian of h(x,u) with respect to the state 
    when the reference is time-varying

    Arguments:
        x {np.array}     -- State of the system (position of the evaders)
        xD {np.array}    -- Desired state of the system
        u {np.array}     -- Input of the system (position of the herders, robots)
        params {dict}    -- Parameters of the evaders' models
        ID {np.array}    -- Behavioural model of each evader (0=Inverse, 1=Exponential)
        values {int}     -- Real or estimated parameters (0=Real, 1=Estimated)
        K {np.array}     -- Control matrix
        xDdot {np.array} -- Dynamics of the desired state of the system
        epsilon{float}   -- tolerance for the numeric calculation
        
    Returns:
        Jx {np.array}  
    
    """    
    # Preallocate Jacobian
    Jx = np.zeros([len(x),len(x)])
    
    # Calculate each column of the Jacobian
    for i in range(len(x)):
        v        = np.zeros(len(x))
        v[i]     = epsilon
        A,B      = buildSystem(x+v, u, params, ID, values)
        f        = calculateF(x+v, u, A, B)
        h1       = calculateHTimeVarying(x+v, xD, f, K, xDdot)
        A,B      = buildSystem(x-v, u, params, ID, values)
        f        = calculateF(x-v, u, A, B)
        h2       = calculateHTimeVarying(x-v, xD, f, K, xDdot)     
        Jx[:,i]  = np.transpose((h1 - h2)/epsilon/2)
    
    return Jx

def buildJuStatic(x: np.array,
            xD: np.array,
            u: np.array,
            params: dict,
            ID: np.array,
            values: int,
            K: np.array,
            epsilon: float = 1e-6
            ):
    """Compute the Jacobian of h(x,u) with respect to the input
    when the reference is static

    Arguments:
        x {np.array}   -- State of the system (position of the evaders)
        xD {np.array}  -- Desired final state of the system
        u {np.array}   -- Input of the system (position of the herders, robots)
        params {dict}  -- Parameters of the evaders' models
        ID {np.array}  -- Behavioural model of each evader (0=Inverse, 1=Exponential)
        values {int}   -- Real or estimated parameters (0=Real, 1=Estimated)
        K {np.array}   -- Control matrix
        epsilon{float} -- tolerance for the numeric calculation

    Returns:
        Ju {np.array}  
    
    """    
    # Preallocate Jacobian
    Ju = np.zeros([len(x),len(u)])
    
    # Calculate each column of the Jacobian
    for i in range(len(u)):
        v        = np.zeros(len(u))
        v[i]     = epsilon
        A,B      = buildSystem(x, u+v, params, ID, values)
        f        = calculateF(x, u+v, A, B)
        h1       = calculateHStatic(x, xD, f, K)
        A,B      = buildSystem(x, u-v, params, ID, values)
        f        = calculateF(x, u-v, A, B)
        h2       = calculateHStatic(x, xD, f, K)     
        Ju[:,i]  = np.transpose((h1 - h2)/epsilon/2)
    
    return Ju

def buildJuStaticMassive(x: np.array,
            xD: np.array,
            u: np.array,
            params: dict,
            ID: np.array,
            values: int,
            K: np.array,
            epsilon: float = 1e-6
            ):
    """Compute the Jacobian of h(x,u) with respect to the input
    when the reference is static

    Arguments:
        x {np.array}   -- State of the system (position of the evaders)
        xD {np.array}  -- Desired final state of the system
        u {np.array}   -- Input of the system (position of the herders, robots)
        params {dict}  -- Parameters of the evaders' models
        ID {np.array}  -- Behavioural model of each evader (0=Inverse, 1=Exponential)
        values {int}   -- Real or estimated parameters (0=Real, 1=Estimated)
        K {np.array}   -- Control matrix
        epsilon{float} -- tolerance for the numeric calculation

    Returns:
        Ju {np.array}  
    
    """    
    # Preallocate Jacobian
    Ju = np.zeros([len(x),len(u)])
    
    # Calculate each column of the Jacobian
    for i in range(len(u)):
        v        = np.zeros(len(u))
        v[i]     = epsilon
        A,B      = buildSystemMassive(x, u+v, params, ID, values)
        f        = calculateF(x, u+v, A, B)
        h1       = calculateHStatic(x, xD, f, K)
        A,B      = buildSystemMassive(x, u-v, params, ID, values)
        f        = calculateF(x, u-v, A, B)
        h2       = calculateHStatic(x, xD, f, K)     
        Ju[:,i]  = np.transpose((h1 - h2)/epsilon/2)
    
    return Ju

def buildJuTimeVarying(x: np.array,
            xD: np.array,
            u: np.array,
            params: dict,
            ID: np.array,
            values: int,
            K: np.array,
            xDdot: np.array,
            epsilon: float = 1e-6
            ):
    """Compute the Jacobian of h(x,u) with respect to the input 
    when the reference is time-varying

    Arguments:
        x {np.array}     -- State of the system (position of the evaders)
        xD {np.array}    -- Desired final state of the system
        u {np.array}     -- Input of the system (position of the herders, robots)
        params {dict}    -- Parameters of the evaders' models
        ID {np.array}    -- Behavioural model of each evader (0=Inverse, 1=Exponential)
        values {int}     -- Real or estimated parameters (0=Real, 1=Estimated)
        K {np.array}     -- Control matrix
        xDdot {np.array} -- Dynamics of the desired state of the system
        epsilon{float}   -- tolerance for the numeric calculation

    Returns:
        Ju {np.array}  
    
    """    
    # Preallocate Jacobian
    Ju = np.zeros([len(x),len(u)])
    
    # Calculate each column of the Jacobian
    for i in range(len(u)):
        v        = np.zeros(len(u))
        v[i]     = epsilon
        A,B      = buildSystem(x, u+v, params, ID, values)
        f        = calculateF(x, u+v, A, B)
        h1       = calculateHTimeVarying(x, xD, f, K, xDdot)
        A,B      = buildSystem(x, u-v, params, ID, values)
        f        = calculateF(x, u-v, A, B)
        h2       = calculateHTimeVarying(x, xD, f, K, xDdot)     
        Ju[:,i]  = np.transpose((h1 - h2)/epsilon/2)
    
    return Ju

def buildJparams(x: np.array,
                 u: np.array, 
                 params: dict,
                 ID: np.array
                 ):
    """Compute the Jacobian of f(x,u) with respect to the parameters

    Arguments:
        x {np.array}   -- State of the system 
        u {np.array}   -- Input of the system 
        params {dict}  -- Parameters of the evaders' models
        ID {np.array}  -- Behavioural model of each evader (0=Inverse, 1=Exponential)
        
    Returns:
        J {np.array}   -- Jacobian
    
    """    
    # Preallocate Jacobian
    J = np.zeros([len(x),int(len(x)/2)])
    
    # For each evader
    for j in range(int(len(x)/2)):
        # Check model
        if ID[j] == 0:
            for i in range(int(len(u)/2)):
                X               = np.linalg.norm(x[2*j:2*j+2]-u[2*i:2*i+2])**3
                J[2*j:2*j+2,j] += (x[2*j:2*j+2]-u[2*i:2*i+2])/X                   
        elif ID[j] == 1:
            var = params["beta"]
            for i in range(int(len(u)/2)):
                if np.linalg.norm(x[2*j:2*j+2]-u[2*i:2*i+2])<params["tau"]:
                    var = 1
                    break
            for i in range(int(len(u)/2)):
                X               = np.linalg.norm(x[2*j:2*j+2]-u[2*i:2*i+2])**2
                J[2*j:2*j+2,j] += var*(x[2*j:2*j+2]-u[2*i:2*i+2])*np.exp(-1/(params["sigma"]**2)*X)
        else:
            print("ID must be 0 (Inverse) or 1 (Exponential) !")
            return
    
    return J

def saturation(xDot: np.array, vMax: float):
    """Apply saturation to the velocity of the entities
    
    Arguments:
        xDot {np.array}  -- Velocity of the entities 
        vMax {float}     -- Maximum velocity 
        
    Returns:
        xDot {np.array} -- Velocity of the entities after saturation
    """
    for i in range(int(len(xDot)/2)):
        if np.linalg.norm(xDot[2*i:2*i+2])>vMax:
            xDot[2*i:2*i+2] = vMax*(xDot[2*i:2*i+2])/np.linalg.norm(xDot[2*i:2*i+2])
      
    return xDot

def applyLimits(x: np.array, xMax: float, xMin: float, yMax: float, yMin: float):
    """Apply space limits to entities (""entities can not go out the fences")

    Arguments:
        x {np.array} -- Position of entities 
        xMax, xMin, yMax, yMin {float} -- Limits of the stage
    """
    for j in range(int(len(x)/2)):
        # X direction
        if x[2*j]>xMax:
            x[2*j] = xMax
        elif x[2*j]<xMin:
            x[2*j] = xMin
        #Y direction
        if x[2*j+1]>yMax:
            x[2*j+1] = yMax
        elif x[2*j+1]<yMin:
            x[2*j+1] = yMin
    
    return

def implicitMethodStatic(x: np.array,
                   xD: np.array,
                   u: np.array,
                   params: dict,
                   ID: np.array,
                   numericParams: dict,
                   K: np.array
                   ):
    """Compute the input by using Levenberg-Marquardt

    Arguments:
        x {np.array}         -- State of the system (position of the evaders)
        xD {np.array}        -- Desired final state of the system (position of the evaders)
        u {np.array}         -- Input of the system (position of the herders, robots)
        params {dict}        -- Parameters of the evaders' models
        ID {np.array}        -- Behavioural model of each evader (0=Inverse, 1=Exponential)
        numericParams {dict} -- Parameters of the numerical method
        K {np.array}         -- Control matrix 
    
    Returns:
        uAux {np.array} -- new input calculated by the numerical method
        
    """ 
    # Preallocate Jacobian of the system and copy input
    J    = np.zeros([len(x),len(u)])
    uAux = np.copy(u)
    
    # Iterations of LM
    for k in range(numericParams["itMax"]):
        # h(x,u)
        h = np.zeros(len(x))
        
        # Calculate Jacobian of the system wrt u
        for j in range(int(len(x)/2)):
            # For each evader, check model
            if ID[j] == 0:
                for i in range(int(len(uAux)/2)):
                    termX = x[2*j]-uAux[2*i]
                    termY = x[2*j+1]-uAux[2*i+1]
                    h[2*j:2*j+2]  += params[str(1)][2*j]*( (x[2*j:2*j+2]-uAux[2*i:2*i+2])/np.linalg.norm(x[2*j:2*j+2]-uAux[2*i:2*i+2])**3 )
                    J[2*j,2*i]     = params[str(1)][2*j]*( ( 3*np.sqrt(termX**2 + termY**2)*termX**2 - np.sqrt(termX**2 + termY**2)**3 ) / (termX**2 + termY**2)**3 )
                    J[2*j,2*i+1]   = params[str(1)][2*j]*( ( 3*np.sqrt(termX**2 + termY**2)*termX*termY ) / (termX**2 + termY**2)**3 )
                    J[2*j+1,2*i]   = J[2*j,2*i+1] 
                    J[2*j+1,2*i+1] = params[str(1)][2*j]*( ( 3*np.sqrt(termX**2 + termY**2)*termY**2 - np.sqrt(termX**2 + termY**2)**3 ) / (termX**2 + termY**2)**3 )
            elif ID[j] == 1:
                var = params["beta"]*params[str(1)][2*j]
                for i in range(int(len(uAux)/2)):
                    if np.linalg.norm(x[2*j:2*j+2]-uAux[2*i:2*i+2])<params["tau"]:
                        var = params[str(1)][2*j]
                        break
                for i in range(int(len(uAux)/2)):
                    termX = x[2*j]-uAux[2*i]
                    termY = x[2*j+1]-uAux[2*i+1]
                    X     = (np.linalg.norm(x[2*j:2*j+2]-uAux[2*i:2*i+2]))**2
                    h[2*j:2*j+2]  += var*(x[2*j:2*j+2]-uAux[2*i:2*i+2])*np.exp(-(1/(params["sigma"]**2))*X)
                    J[2*j,2*i]     = var*( (2/params["sigma"]**2)*np.exp(-(1/(params["sigma"]**2))*X)*termX**2 - np.exp(-(1/(params["sigma"]**2))*X) )
                    J[2*j,2*i+1]   = var*( (2/params["sigma"]**2)*np.exp(-(1/(params["sigma"]**2))*X)*termX*termY )
                    J[2*j+1,2*i]   = J[2*j,2*i+1]
                    J[2*j+1,2*i+1] = var*( (2/params["sigma"]**2)*np.exp(-(1/(params["sigma"]**2))*X)*termY**2 - np.exp(-(1/(params["sigma"]**2))*X) )
            else:
                print("ID must be 0 (Inverse) or 1 (Exponential) !")
                return
            h[2*j:2*j+2] = h[2*j:2*j+2] + np.matmul(K[2*j:2*j+2,2*j:2*j+2],x[2*j:2*j+2]-xD[2*j:2*j+2])

  
        # Calc LM
        incU  = np.matmul(np.linalg.inv(np.matmul(np.transpose(J),J)+numericParams["lambda"]*np.eye(len(u))),np.matmul(np.transpose(J),h))
        uAux  = uAux - incU
        error = np.linalg.norm(incU)
        if error <= numericParams["tolAbs"]:
            break
        
        
    return uAux, k, np.linalg.norm(h)

def implicitMethodTimeVarying(x: np.array,
                   xD: np.array,
                   u: np.array,
                   params: dict,
                   ID: np.array,
                   numericParams: dict,
                   K: np.array,
                   xDdot: np.array
                   ):
    """Compute the input by using Levenberg-Marquardt, but with time-varying 
       reference 

    Arguments:
        x {np.array}         -- State of the system (position of the evaders)
        xD {np.array}        -- Desired final state of the system (position of the evaders)
        u {np.array}         -- Input of the system (position of the herders, robots)
        params {dict}        -- Parameters of the evaders' models
        ID {np.array}        -- Behavioural model of each evader (0=Inverse, 1=Exponential)
        numericParams {dict} -- Parameters of the numerical method
        K {np.array}         -- Control matrix 
        xDdot {np.array}     -- time derivative of the time-varying reference
    
    Returns:
        uAux {np.array} -- new input calculated by the numerical method
        
    """ 
    # Preallocate Jacobian of the system and copy input
    J    = np.zeros([len(x),len(u)])
    uAux = np.copy(u)
    
    # Iterations of LM
    for k in range(numericParams["itMax"]):
        # h(x,u)
        h = np.zeros(len(x))
        
        # Calculate Jacobian of the system wrt u
        for j in range(int(len(x)/2)):
            # For each evader, check model
            if ID[j] == 0:
                for i in range(int(len(uAux)/2)):
                    termX = x[2*j]-uAux[2*i]
                    termY = x[2*j+1]-uAux[2*i+1]
                    h[2*j:2*j+2]  += params[str(1)][2*j]*( (x[2*j:2*j+2]-uAux[2*i:2*i+2])/np.linalg.norm(x[2*j:2*j+2]-uAux[2*i:2*i+2])**3 )
                    J[2*j,2*i]     = params[str(1)][2*j]*( ( 3*np.sqrt(termX**2 + termY**2)*termX**2 - np.sqrt(termX**2 + termY**2)**3 ) / (termX**2 + termY**2)**3 )
                    J[2*j,2*i+1]   = params[str(1)][2*j]*( ( 3*np.sqrt(termX**2 + termY**2)*termX*termY ) / (termX**2 + termY**2)**3 )
                    J[2*j+1,2*i]   = J[2*j,2*i+1] 
                    J[2*j+1,2*i+1] = params[str(1)][2*j]*( ( 3*np.sqrt(termX**2 + termY**2)*termY**2 - np.sqrt(termX**2 + termY**2)**3 ) / (termX**2 + termY**2)**3 )
            elif ID[j] == 1:
                var = params["beta"]*params[str(1)][2*j]
                for i in range(int(len(uAux)/2)):
                    if np.linalg.norm(x[2*j:2*j+2]-uAux[2*i:2*i+2])<params["tau"]:
                        var = params[str(1)][2*j]
                        break
                for i in range(int(len(uAux)/2)):
                    termX = x[2*j]-uAux[2*i]
                    termY = x[2*j+1]-uAux[2*i+1]
                    X     = (np.linalg.norm(x[2*j:2*j+2]-uAux[2*i:2*i+2]))**2
                    h[2*j:2*j+2]  += var*(x[2*j:2*j+2]-uAux[2*i:2*i+2])*np.exp(-(1/(params["sigma"]**2))*X)
                    J[2*j,2*i]     = var*( (2/params["sigma"]**2)*np.exp(-(1/(params["sigma"]**2))*X)*termX**2 - np.exp(-(1/(params["sigma"]**2))*X) )
                    J[2*j,2*i+1]   = var*( (2/params["sigma"]**2)*np.exp(-(1/(params["sigma"]**2))*X)*termX*termY )
                    J[2*j+1,2*i]   = J[2*j,2*i+1]
                    J[2*j+1,2*i+1] = var*( (2/params["sigma"]**2)*np.exp(-(1/(params["sigma"]**2))*X)*termY**2 - np.exp(-(1/(params["sigma"]**2))*X) )
            else:
                print("ID must be 0 (Inverse) or 1 (Exponential) !")
                return
            # h[2*j:2*j+2] = h[2*j:2*j+2] + np.matmul(K[2*j:2*j+2,2*j:2*j+2],np.matmul(x[2*j:2*j+2]-xD[2*j:2*j+2], x[2*j:2*j+2]-xD[2*j:2*j+2])*(x[2*j:2*j+2]-xD[2*j:2*j+2])) - xDdot[2*j:2*j+2]
            h[2*j:2*j+2] = h[2*j:2*j+2] + np.matmul(K[2*j:2*j+2,2*j:2*j+2],x[2*j:2*j+2]-xD[2*j:2*j+2]) - xDdot[2*j:2*j+2]
  
        # Calc LM
        incU  = np.matmul(np.linalg.inv(np.matmul(np.transpose(J),J)+numericParams["lambda"]*np.eye(len(u))),np.matmul(np.transpose(J),h))
        uAux  = uAux - incU
        error = np.linalg.norm(incU)
        if error <= numericParams["tolAbs"]:
            break
        
        
    return uAux, k, np.linalg.norm(h)

def measurement(pos: np.array,
                x: np.array,
                u: np.array,
                H: np.array,
                R: np.array,
                sensorRange: float,
                communicationRange: float
                ):
    """ Simulates de sensing process

    Arguments:
        pos {np.array}     -- Current position
        x {np.array}       -- State of the system (position of the evaders)
        u {np.array}       -- Input of the system (position of the herders, robots)
        H {np.array}       -- Sensor model 
        R {np.array}       -- Covariance of measurement matrix 
        sensorRange{float} -- Range of detection of the sensors
    
    Returns:
        z {np.array}         -- Observation of each herder 
        L {np.array}         -- Communication graph
        measureID {np.array} -- ID saying whether the robot has sensed and entity or not
        
    """                     
    
    # Preallocate measurement vector and build aggregated state
    state           = x
    z               = np.zeros(len(state))
    L               = -np.ones(int(len(u)/2))
    measureID       = np.ones(int(len(state)/2))
    selector        = 0
    
    # For each entity
    for j in range(int(len(z)/2)):
        z[2*j:2*j+2] = state[2*j:2*j+2]*(1-(1/(1+np.exp(5*(-np.linalg.norm(pos-state[2*j:2*j+2])+sensorRange))))) + np.diag(np.random.normal(np.zeros(2),R))
        if 1-(1/(1+np.exp(5*(-np.linalg.norm(pos-state[2*j:2*j+2])+sensorRange)))) < 0.99:
            measureID[j] = 0
            
    for j in range(int(len(u)/2)):
        if (pos == u[2*j:2*j+2]).all():
            L[j]               = 0
            selector           = j 
        if 1-(1/(1+np.exp(5*(-np.linalg.norm(pos-u[2*j:2*j+2])+communicationRange)))) < 0.99:
            L[j]               = 0
    L[selector] = np.sum(-L,0)        
    
    return z, L, measureID

def measurement_complete(   pos: np.array,
                            x: np.array,
                            u: np.array,
                            H: np.array,
                            R: np.array,
                            sensorRange: float,
                            communicationRange: float
                            ):
    """ Simulates de sensing process

    Arguments:
        pos {np.array}     -- Current position
        x {np.array}       -- State of the system (position of the evaders)
        u {np.array}       -- Input of the system (position of the herders, robots)
        H {np.array}       -- Sensor model 
        R {np.array}       -- Covariance of measurement matrix 
        sensorRange{float} -- Range of detection of the sensors
    
    Returns:
        z {np.array}         -- Observation of each herder 
        L {np.array}         -- Communication graph
        measureID {np.array} -- ID saying whether the robot has sensed and entity or not
        
    """                     
    
    # Preallocate measurement vector and build aggregated state
    state     = np.concatenate((x,u),0)
    z         = np.zeros(len(state))
    L         = -np.ones(int(len(u)/2))
    measureID = np.ones(int(len(state)/2))
    selector  = 0
    
    # For each entity
    for j in range(int(len(z)/2)):
        z[2*j:2*j+2] = state[2*j:2*j+2]*(1-(1/(1+np.exp(5*(-np.linalg.norm(pos-state[2*j:2*j+2])+sensorRange))))) + np.diag(np.random.normal(np.zeros(2),R))
        
        # Check if evader or herder
        if j<int(len(x)/2):
            if 1-(1/(1+np.exp(5*(-np.linalg.norm(pos-state[2*j:2*j+2])+sensorRange)))) < 0.99:
                measureID[j] = 0
        else:
            if 1-(1/(1+np.exp(5*(-np.linalg.norm(pos-state[2*j:2*j+2])+sensorRange)))) < 0.99:
                L[j-int(len(x)/2)] = 0
                measureID[j]       = 0
            if (pos == state[2*j:2*j+2]).all():
                L[j-int(len(x)/2)] = 0
                z[2*j:2*j+2]       = state[2*j:2*j+2]
                selector           = j-int(len(x)/2)
    L[selector] = np.sum(-L,0)       
    
    return z, L, measureID

def o_dkf_extended(  mine: np.array,
                     L: np.array,  
                     H: list, 
                     R: list, 
                     P: np.array, 
                     Q: np.array, 
                     x_est: list, 
                     z: list,
                     measureID: list,
                     N_herders: int, 
                     M_evaders: int,
                     xD: np.array,
                     params: dict,
                     ID: np.array,
                     values: int,
                     K: np.array,
                     Kstar: np.array,
                     u: np.array,
                     T: float
                     ):
    """
    This function implements the Olfati-Saber 2007 Distributed Kalman Filter,
    in IEEE CDC 2007, extended version, when the reference is static
    
    Arguments:
        mine {np.array}      -- My estimate
        L {np.array}         -- Topology of the sensor network (Laplacian)
        H {list}             -- Sensor model
        R {list}             -- Covariance of the stochastic gaussian noise affecting the sensor
        P {np.array}         -- Covariance error matrix 
        Q {np.array}         -- Covariance of the stochastic gaussian noise affecting the particle
        x_est {list}         -- Estimate of the state and input
        z {list}             -- Measurements
        measureID {list}     -- ID saying whether the robot has sensed and entity or not
        N_herders {int}      -- Number of herders
        M_evaders {int}      -- Number of evaders
        xD {np.array}        -- Desired final state of the system
        params {dict}        -- Parameters of the evaders' models
        ID {np.array}        -- Behavioural model of each evader (0=Inverse, 1=Exponential)
        values {int}         -- Real or estimated parameters (0=Real, 1=Estimated)
        K {np.array}         -- Control matrix
        Kstar {np.array}     -- Convergence matrix
        T {float}            -- Sample time
        
    Returns:
        x_est {np.array} -- New estimates
        P {np.array}     -- New covariance error matrix
    
    """   
    # Local aggregation
    y, S = information(z,R,H,L,measureID)
        
    # Kalman Gain and consensus
    M     = np.linalg.inv(np.linalg.inv(P) + S)
    mine  = mine + np.matmul(M,(y - np.matmul(S,mine))) + (1/(np.linalg.norm(M,'fro')+1))*M@(-L@(np.array(x_est)).reshape([len(L),len(mine)]))
    
    # Jacobians
    Jx = buildJxStatic(mine,xD,u,params,ID,values,K)
    
    # Update
    A,B   = buildSystem(mine,u,params,ID,values)
    f     = calculateF(mine,u,A,B)
    mine += f*T
    P     = Jx@M@Jx.T + Q
    
    return mine, P

def o_dkf_extended_complete( mine: np.array,
                             L: np.array,  
                             H: list, 
                             R: list, 
                             P: np.array, 
                             Q: np.array, 
                             x_est: list, 
                             z: list,
                             measureID: list,
                             N_herders: int, 
                             M_evaders: int,
                             xD: np.array,
                             params: dict,
                             ID: np.array,
                             values: int,
                             K: np.array,
                             Kstar: np.array,
                             u: np.array,
                             T: float,
                             vMax: float,
                             xDdot: np.array,
                             com: bool
                             ):
    """
    This function implements the Olfati-Saber 2007 Distributed Kalman Filter,
    in IEEE CDC 2007, extended version, when the reference is time-varying
    
    Arguments:
        mine {np.array}      -- My estimate
        L {np.array}         -- Topology of the sensor network (Laplacian)
        H {list}             -- Sensor model
        R {list}             -- Covariance of the stochastic gaussian noise affecting the sensor
        P {np.array}         -- Covariance error matrix 
        Q {np.array}         -- Covariance of the stochastic gaussian noise affecting the particle
        x_est {list}         -- Estimate of the state and input
        z {list}             -- Measurements
        measureID {list}     -- ID saying whether the robot has sensed and entity or not
        N_herders {int}      -- Number of herders
        M_evaders {int}      -- Number of evaders
        xD {np.array}        -- Desired final state of the system
        params {dict}        -- Parameters of the evaders' models
        ID {np.array}        -- Behavioural model of each evader (0=Inverse, 1=Exponential)
        values {int}         -- Real or estimated parameters (0=Real, 1=Estimated)
        K {np.array}         -- Control matrix
        Kstar {np.array}     -- Convergence matrix
        T {float}            -- Sample time
        vMax {float}         -- Maximum speed
        xDdot {np.array}     -- Time varying reference
        com {bool}           -- True if herders communicate, False otherwise
    Returns:
        x_est {np.array} -- New estimates
        P {np.array}     -- New covariance error matrix
    
    """   
    # Local aggregation
    y, S = information(z,R,H,L,measureID,com)
        
    # Kalman Gain and consensus
    M     = np.linalg.inv(np.linalg.inv(P) + S)
    mine  = mine + np.matmul(M,(y - np.matmul(S,mine))) + (1/(np.linalg.norm(M,'fro')+1))*M@(-L@(np.array(x_est)).reshape([len(L),len(mine)]))
    
    # Jacobians
    J = jacobians(mine[0:2*M_evaders],xD,mine[2*M_evaders:2*M_evaders+2*N_herders+1],params,ID,values,K,Kstar,N_herders,M_evaders,vMax,xDdot)
    
    # Extract estimates of x and u
    x_estimated = np.zeros(2*M_evaders)
    u_estimated = np.zeros(2*N_herders)
    for i in range(2*M_evaders):
        x_estimated[i] = mine[i]
    for i in range(2*N_herders):
        u_estimated[i] = mine[i+2*M_evaders]
            
    # Update
    A,B   = buildSystem(x_estimated,u_estimated,params,ID,values)
    f     = calculateF(x_estimated,u_estimated,A,B)
    
    h     = calculateHTimeVarying(x_estimated,xD,f,K,xDdot)
    hD    = -np.matmul(Kstar,h)
    Jx    = buildJxTimeVarying(x_estimated,xD,u_estimated,params,ID,values,K,xDdot)
    Ju    = buildJuTimeVarying(x_estimated,xD,u_estimated,params,ID,values,K,xDdot)
    uDot  = np.matmul(np.transpose(Ju),np.matmul(np.linalg.inv(np.matmul(Ju,np.transpose(Ju))+0.001*np.eye(2*N_herders)),(hD - np.matmul(Jx,f))))
    uDot  = saturation(uDot, vMax)
    
    mine += np.concatenate((f,uDot),0)*T
    P     = J@M@J.T + Q
    
    return mine, P

def information(z: list, R: list, H: list, L: np.array, measureID: list, com: bool):
    """
    This function return the weighted information-form-based sum of the measurements
    
    Arguments:
        z {list}             -- Measurements 
        R {list}             -- Covariance noise 
        H {list}             -- Sensor model
        L {np.array}         -- Neighbours of sensor i
        measureID {list}     -- ID saying whether the robot has sensed and entity or not
        com {bool}           -- True if herders communicate, False otherwise
        
    Returns:
        y_i {np.array}  -- Information fusioned measurement
        S_i {np.array}  -- Information fusioned covariance
    
    """
    # Store dimensions
    n = len(z)
    m = len(z[0])
    
    # Calculate outputs
    y_i = np.zeros(m)
    S_i = np.zeros([m,m])
    
    if com:
    
        for i in range(n):
            if L[i] != 0:
                dim   = int(np.sum(measureID[i])) 
                zReal = np.zeros([2*dim])
                RReal = np.zeros([2*dim,2*dim])
                HReal = np.zeros([2*dim,m])
                invR  = np.linalg.inv(R[i])
                selec = 0
                for j in range(len(measureID[i])):
                    if measureID[i][j] != 0:
                        zReal[2*selec:2*selec+2] = z[i][2*j:2*j+2]
                        RReal[2*selec:2*selec+2,2*selec:2*selec+2] = invR
                        HReal[2*selec:2*selec+2,2*j:2*j+2] = H[i]
                        selec += 1
                y_i += np.matmul(HReal.T,np.matmul(RReal,zReal))
                S_i += HReal.T @ RReal @ HReal
                
    else:
        
        for i in range(n):
            if L[i] >= 0:
                dim   = int(np.sum(measureID[i])) 
                zReal = np.zeros([2*dim])
                RReal = np.zeros([2*dim,2*dim])
                HReal = np.zeros([2*dim,m])
                invR  = np.linalg.inv(R[i])
                selec = 0
                for j in range(len(measureID[i])):
                    if measureID[i][j] != 0:
                        zReal[2*selec:2*selec+2] = z[i][2*j:2*j+2]
                        RReal[2*selec:2*selec+2,2*selec:2*selec+2] = invR
                        HReal[2*selec:2*selec+2,2*j:2*j+2] = H[i]
                        selec += 1
                y_i += np.matmul(HReal.T,np.matmul(RReal,zReal))
                S_i += HReal.T @ RReal @ HReal
        
    
    return y_i, S_i

def jacobians(  x: np.array,
                xD: np.array,
                u: np.array,
                params: dict,
                ID: np.array,
                values: int,
                K: np.array,
                Kstar: np.array,
                N_herders: int,
                M_evaders: int,
                vMax: float,
                xDdot: np.array,
                epsilon: float = 1e-6
                ):
    """Compute the Jacobian of the augmented system wrt to x and u

    Arguments:
        x {np.array}     -- State of the system (position of the evaders)
        xD {np.array}    -- Desired final state of the system
        u {np.array}     -- Input of the system (position of the herders, robots)
        params {dict}    -- Parameters of the evaders' models
        ID {np.array}    -- Behavioural model of each evader (0=Inverse, 1=Exponential)
        values {int}     -- Real or estimated parameters (0=Real, 1=Estimated)
        K {np.array}     -- Control matrix
        Kstar {np.array} -- Convergence matrix
        N_herders {int}  -- Number of herders
        M_evaders {int}    -- Number of evaders
        vMax {float}     -- Maximum speed
        xDdot {np.array} -- Time-varying reference
        epsilon{float}   -- tolerance for the numeric calculation

    Returns:
        Jx {np.array}
        Ju {np.array}  
    
    """    
    # Preallocate Jacobian
    J = np.zeros([2*(N_herders+M_evaders),2*(N_herders+M_evaders)])
    
    # Calculate each column of Jx
    for i in range(2*(N_herders+M_evaders)):
        v        = np.zeros(2*(N_herders+M_evaders))
        v[i]     = epsilon
        
        agregate = np.concatenate((x,u),0) + v
        new_x    = agregate[0:2*M_evaders]
        new_u    = agregate[2*M_evaders:2*(N_herders+M_evaders)]
        Jx       = buildJxTimeVarying(new_x,xD,new_u,params,ID,values,K,xDdot)
        Ju       = buildJuTimeVarying(new_x,xD,new_u,params,ID,values,K,xDdot)
        A,B      = buildSystem(new_x,new_u,params,ID,values)
        f        = calculateF(new_x,new_u,A,B)
        h        = calculateHTimeVarying(new_x,xD,f,K,xDdot)
        hD       = -np.matmul(Kstar,h)
        uDot     = np.matmul(np.transpose(Ju),np.matmul(np.linalg.inv(np.matmul(Ju,np.transpose(Ju))+0.001*np.eye(2*N_herders)),(hD - np.matmul(Jx,f))))
        uDot     = saturation(uDot, vMax)
        state2   = np.concatenate((f,uDot),0)
        
        agregate = np.concatenate((x,u),0) - v
        new_x    = agregate[0:2*M_evaders]
        new_u    = agregate[2*M_evaders:2*(N_herders+M_evaders)]
        Jx       = buildJxTimeVarying(new_x,xD,new_u,params,ID,values,K,xDdot)
        Ju       = buildJuTimeVarying(new_x,xD,new_u,params,ID,values,K,xDdot)
        A,B      = buildSystem(new_x,new_u,params,ID,values)
        f        = calculateF(new_x,new_u,A,B)
        h        = calculateHTimeVarying(new_x,xD,f,K,xDdot)
        hD       = -np.matmul(Kstar,h)
        uDot     = np.matmul(np.transpose(Ju),np.matmul(np.linalg.inv(np.matmul(Ju,np.transpose(Ju))+0.001*np.eye(2*N_herders)),(hD - np.matmul(Jx,f))))
        uDot     = saturation(uDot, vMax)
        state1   = np.concatenate((f,uDot),0)
        
        J[:,i]   = np.transpose((state2-state1)/epsilon)
    
    return J

def getEllipse(x: np.array, 
               P: np.array, 
               n: int, 
               radius:float
               ):
    """
    This function plots an ellipse/polytope given its contour, eigenvalues and eigenvectors
    
    Arguments:
        x {np.array}       -- Center of the polytope
        P {np.array}       -- Covariance matrix
        n {int}            -- Number of desired discrete points
        radius {float}     -- Distance to the original ellipse
        
    Returns:
        ellipse {np.array} -- Points herders must go
        a {float}          -- Mayor axis
        b {float}          -- Minor axis
        V {np.array}       -- Rotation matrix
    
    """
    
    """ Obtain SVD decomposition """
    U, D, V = np.linalg.svd(P)

    """ Get "eigenaxis" """
    a = 1/np.sqrt(D[0]) * radius
    if D[1] == 0:
        b = np.copy(a)
    else:
        b = 1/np.sqrt(D[1]) * radius
    
    """ Grid """
    theta_grid = np.linspace(0,2*np.pi,n)
    
    """ Parametric equation of the ellipse """
    state      = np.zeros([2,n]) 
    state[0,:] = a*np.cos(theta_grid)
    state[1,:] = b*np.sin(theta_grid)
    
    """ Coordinate transform  """
    ellipse = (V @ state).T 
    
    return ellipse, a, b, V

def functional(theta: float,
               a: float,
               b: float,
               x: float,
               y: float):
    """
    This function implements the equation to be solved to obtain the distance between a given point
    (x,y) and an ellipse described by axis a and b
    
    Arguments:
        x {np.array}       -- Center of the polytope
        P {np.array}       -- Covariance matrix
        n {int}            -- Number of desired discrete points
        radius {float}     -- Distance to the original ellipse
        
    Returns:
        ellipse {np.array} -- Points herders must go
        a {float}          -- Mayor axis
        b {float}          -- Minor axis
    
    """
    return (a**2-b**2)*np.cos(theta)*np.sin(theta) - x*a*np.sin(theta) + y*b*np.cos(theta)

def get_cmap(n:int):
    """
    This function generates an equally space array of RGB colours
    """
    return plt.cm.get_cmap('hsv', n+1)