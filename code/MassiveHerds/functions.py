# -*- coding: utf-8 -*-
"""
This file contains all the functions to run the simulations in:
    
    E. Sebastián, E. Montijano, C. Sagüés, "Multi-robot Implicit Control of Massive Herds"

Current Version: 2022-11-30

Eduardo Sebastián Rodríguez, PhD Student / esebastian@unizar.es / https://eduardosebastianrodriguez.github.io/
Department of Computer Science and Systems Engineering / diis.unizar.es
University of Zaragoza / unizar.es
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull, cKDTree
from sklearn.cluster import KMeans

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
        x {np.array}  -- State of the system (position of the preys)
        u {np.array}  -- Input of the system (position of the herders, robots)
        params {dict} -- Parameters of the preys' models
        ID {np.array} -- Behavioural model of each prey (0=Inverse, 1=Exponential)
        values {int}  -- Real or estimated parameters (0=Real, 1=Estimated)

    Returns:
        A(x,u) {np.array} 
        B(x,u) {np.array} 
    
    """    
    # Init matrices with proper dimensions
    A  = np.zeros([len(x),len(x)])
    B  = np.zeros([len(x),len(u)])
    
    """ Create A(x,u) """
    # For each prey
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
                    X = np.linalg.norm(x[2*j:2*j+2]-x[2*i:2*i+2])**2
                    if X != 0:
                        A[2*j,2*j]     = A[2*j,2*j] + params["gamma"]*params[str(values)][2*j]/X 
                        A[2*j+1,2*j+1] = A[2*j+1,2*j+1] + params["gamma"]*params[str(values)][2*j]/X 
                        A[2*j,2*i]     = A[2*j,2*i] - params["gamma"]*params[str(values)][2*j]/X 
                        A[2*j+1,2*i+1] = A[2*j+1,2*i+1] - params["gamma"]*params[str(values)][2*j]/X                   
            elif ID[j] == 1:
                var = params["beta"]*params[str(values)][2*j]
                for i in range(int(len(x)/2)):
                    if np.linalg.norm(x[2*j:2*j+2]-x[2*i:2*i+2])<params["tau"]:
                        var = params[str(values)][2*j]
                        break
                for i in range(int(len(x)/2)):
                    X = np.linalg.norm(x[2*j:2*j+2]-x[2*i:2*i+2])**2
                    if X != 0:
                        A[2*j,2*j]     = A[2*j,2*j] + params["gamma"]*params[str(values)][2*j]/X 
                        A[2*j+1,2*j+1] = A[2*j+1,2*j+1] + params["gamma"]*params[str(values)][2*j]/X 
                        A[2*j,2*i]     = A[2*j,2*i] - params["gamma"]*params[str(values)][2*j]/X 
                        A[2*j+1,2*i+1] = A[2*j+1,2*i+1] - params["gamma"]*params[str(values)][2*j]/X 
            else:
                print("ID must be 0 (Inverse) or 1 (Exponential) !")
                return
    
    """ Create B(x,u) """
    # For each prey
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
        x {np.array} -- State of the system (position of the preys)
        u {np.array} -- Input of the system (position of the herders, robots)
        A {np.array} -- State matrix
        B {np.array} -- Input matrix

    Returns:
        f(x,u) {np.array}  
    
    """     
    return np.matmul(A,x) + np.matmul(B,u) 

def calculateH_static(x: np.array, xD: np.array, f: np.array, K: np.array):
    """Compute the time derivative of h:
        
                    h(x,u) = f(x,u) - f*(x,u) = f(x,u) + K(x-x*)

    Arguments:
        x {np.array}  -- State of the system (position of the preys)
        xD {np.array} -- Desired final state of the system
        f {np.array}  -- Dynamic of the system
        K {np.array}  -- Control matrix

    Returns:
        h(x,u) {np.array}  
    
    """    
    return f + np.matmul(K,x-xD)

def calculateH_dynamic(x: np.array, xD: np.array, f: np.array, K: np.array, xDdot: np.array):
    """Compute the time derivative of the h for time-varying references:
        
                    h(x,u) = f(x,u) - f*(x,u) = f(x,u) + K(x-x*) - dx*dt

    Arguments:
        x {np.array}     -- State of the system (position of the preys)
        xD {np.array}    -- Desired state of the system
        f {np.array}     -- Dynamic of the system
        K {np.array}     -- Control matrix
        xDdot {np.array} -- Dynamics of the desired state of the system

    Returns:
        h(x,u) {np.array}  
    
    """    
    return f + np.matmul(K,x-xD) - xDdot


def buildJx_static(x: np.array,
                   xD: np.array,
                   u: np.array,
                   params: dict,
                   ID: np.array,
                   values: int,
                   K: np.array,
                   epsilon: float = 1e-6
                   ):
    """ Numerically compute the Jacobian of h(x,u) with respect to the state

    Arguments:
        x {np.array}   -- State of the system (position of the preys)
        xD {np.array}  -- Desired final state of the system
        u {np.array}   -- Input of the system (position of the herders, robots)
        params {dict}  -- Parameters of the preys' models
        ID {np.array}  -- Behavioural model of each prey (0=Inverse, 1=Exponential)
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
        h1       = calculateH_static(x+v, xD, f, K)
        A,B      = buildSystem(x-v, u, params, ID, values)
        f        = calculateF(x-v, u, A, B)
        h2       = calculateH_static(x-v, xD, f, K)     
        Jx[:,i]  = np.transpose((h1 - h2)/epsilon/2)
    
    return Jx

def buildJx_dynamic(x: np.array,
                    xD: np.array,
                    u: np.array,
                    params: dict,
                    ID: np.array,
                    values: int,
                    K: np.array,
                    xDdot: np.array,
                    epsilon: float = 1e-6
                    ):
    """ Numerically compute the Jacobian of h(x,u) with respect to the state (but using 
       a time-varying reference )

    Arguments:
        x {np.array}     -- State of the system (position of the preys)
        xD {np.array}    -- Desired state of the system
        u {np.array}     -- Input of the system (position of the herders, robots)
        params {dict}    -- Parameters of the preys' models
        ID {np.array}    -- Behavioural model of each prey (0=Inverse, 1=Exponential)
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
        h1       = calculateH_dynamic(x+v, xD, f, K, xDdot)
        A,B      = buildSystem(x-v, u, params, ID, values)
        f        = calculateF(x-v, u, A, B)
        h2       = calculateH_dynamic(x-v, xD, f, K, xDdot)     
        Jx[:,i]  = np.transpose((h1 - h2)/epsilon/2)
    
    return Jx

def buildJu_static(x: np.array,
                   xD: np.array,
                   u: np.array,
                   params: dict,
                   ID: np.array,
                   values: int,
                   K: np.array,
                   epsilon: float = 1e-6
                   ):
    """Numerically compute the Jacobian of h(x,u) with respect to the input

    Arguments:
        x {np.array}   -- State of the system (position of the preys)
        xD {np.array}  -- Desired final state of the system
        u {np.array}   -- Input of the system (position of the herders, robots)
        params {dict}  -- Parameters of the preys' models
        ID {np.array}  -- Behavioural model of each prey (0=Inverse, 1=Exponential)
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
        h1       = calculateH_static(x, xD, f, K)
        A,B      = buildSystem(x, u-v, params, ID, values)
        f        = calculateF(x, u-v, A, B)
        h2       = calculateH_static(x, xD, f, K)     
        Ju[:,i]  = np.transpose((h1 - h2)/epsilon/2)
    
    return Ju

def buildJu_dynamic(x: np.array,
                    xD: np.array,
                    u: np.array,
                    params: dict,
                    ID: np.array,
                    values: int,
                    K: np.array,
                    xDdot: np.array,
                    epsilon: float = 1e-6
                    ):
    """Numerically compute the Jacobian of h(x,u) with respect to the input (but using 
       the time-varying reference )

    Arguments:
        x {np.array}     -- State of the system (position of the preys)
        xD {np.array}    -- Desired final state of the system
        u {np.array}     -- Input of the system (position of the herders, robots)
        params {dict}    -- Parameters of the preys' models
        ID {np.array}    -- Behavioural model of each prey (0=Inverse, 1=Exponential)
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
        h1       = calculateH_dynamic(x, xD, f, K, xDdot)
        A,B      = buildSystem(x, u-v, params, ID, values)
        f        = calculateF(x, u-v, A, B)
        h2       = calculateH_dynamic(x, xD, f, K, xDdot)     
        Ju[:,i]  = np.transpose((h1 - h2)/epsilon/2)
    
    return Ju


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

def get_cmap(n:int):
    """
    This function generates an equally space array of RGB colours
    """
    return plt.cm.get_cmap('hsv', n+1)

    
def dynamicAssignment(x: np.array, n: int):
    """
    This function returns the evaders that the herders have to control
    
    Arguments:
        x {np.array} -- 2D points
        n {int}      -- Number of desired clusters
        
    Returns:
        output {np.array} -- Evaders to herd
    
    """
    # Reshape vector of evaders
    evaders        = x.reshape(-1, 2)  

    # Compute the convex hull of the evaders              
    convexHull     = ConvexHull(evaders) 
    
    # Compute the clusters of evaders in the convex hull using K-Means
    if convexHull.vertices.shape[0] >= n:
        kmeans         = KMeans(n_clusters=n).fit(evaders[convexHull.vertices])
    else:
        kmeans         = KMeans(n_clusters=convexHull.vertices.shape[0]).fit(evaders[convexHull.vertices])
    
    # Compute the Voronoi regions associated to the clusters and the evaders that belong to them
    voronoi_kdtree = cKDTree(kmeans.cluster_centers_)
    _, regions     = voronoi_kdtree.query(evaders[convexHull.vertices], k=1)
    output         = np.array([])
    
    # Compute the furthest evader in each cluster to its corresponding center
    if convexHull.vertices.shape[0] >= n:
        for i in range(n):
            points = evaders[convexHull.vertices][regions==i]
            if points.shape[0] != 0:
                dist   = np.linalg.norm(points - np.tile(kmeans.cluster_centers_[i], points.shape[0]).reshape(-1,2), ord=2, axis=1)
                output = np.concatenate((output, evaders[convexHull.vertices][regions==i][np.argmax(dist)]))
    else:
        for i in range(convexHull.vertices.shape[0]):
            points = evaders[convexHull.vertices][regions==i]
            if points.shape[0] != 0:
                dist   = np.linalg.norm(points - np.tile(kmeans.cluster_centers_[i], points.shape[0]).reshape(-1,2), ord=2, axis=1)
                output = np.concatenate((output, evaders[convexHull.vertices][regions==i][np.argmax(dist)]))
                
    return output