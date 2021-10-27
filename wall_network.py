# -*- coding: utf-8 -*-
"""
Created on Wed Oct 27 15:19:32 2021

@author: ellio
"""
# Model code
# Thermal network
# ---------------
# Dissembled circuits
# TCd0:  Concrete and insulation wall  (in red)
def wall_network(wall, G_cd, Gw, Capacity, GLW, h, Gv, Kp):

    import numpy as np
    nq = 1 + 2 * (wall['Meshes']['Concrete'] + wall['Meshes']['Insulation'])
    nt = 1 + 2 * (wall['Meshes']['Concrete'] + wall['Meshes']['Insulation'])

    A = np.eye(nq + 1, nt)
    A = -np.diff(A, 1, 0).T
    
    nc = wall['Meshes']['Concrete']
    ni = wall['Meshes']['Insulation']
    Gcm = 2 * nc * [G_cd['Concrete']]
    Gcm = 2 * nc * np.array(Gcm)
    Gim = 2 * ni * [G_cd['Insulation']]
    Gim = 2 * wall['Meshes']['Insulation'] * np.array(Gim)
    G = np.diag(np.hstack([Gw['out'], Gcm, Gim]))
    
    
    b = np.zeros(nq)
    b[0] = 1
    
    Ccm = Capacity['Concrete'] / nc * np.mod(range(0, 2 * nc), 2)
    Cim = Capacity['Insulation'] / ni * np.mod(range(0, 2 * ni), 2)
    C = np.diag(np.hstack([Ccm, Cim, 0]))
    
    f = np.zeros(nt)
    f[0] = f[-1] = 1
    
    y = np.zeros(nt)
    
    TCd0 = {'A': A, 'G': G, 'b': b, 'C': C, 'f': f, 'y': y}
    
    # TCd1: Indoor air (in blue)
    A = np.array([[-1, 1, 0],
                  [-1, 0, 1],
                  [0, -1, 1]])
    G = np.diag(np.hstack([GLW, Gw['in'], h['in'] * wall['Surface']['Glass']]))
    b = np.zeros(3)
    C = np.diag([0, 0, Capacity['Air'] / 2])
    f = np.array([1, 0, 1])
    y = np.array([0, 0, 1])
    TCd1 = {'A': A, 'G': G, 'b': b, 'C': C, 'f': f, 'y': y}
    
    # TCd2: Glass (in green)
    A = np.array([[1, 0],
                  [-1, 1]])
    Ggo = h['out'] * wall['Surface']['Glass']
    Ggs = 1 / (1 / Ggo + 1 / (2 * G_cd['Glass']))
    G = np.diag(np.hstack([Ggs, 2 * G_cd['Glass']]))
    b = np.array([1, 0])
    C = np.diag([Capacity['Glass'], 0])
    f = np.array([1, 0])
    y = np.array([0, 0])
    TCd2 = {'A': A, 'G': G, 'b': b, 'C': C, 'f': f, 'y': y}
    
    # TCd3: air infiltration and controller (in purple)
    A = np.array([[1],
                  [1]])
    G = np.diag(np.hstack([Gv, Kp]))
    b = np.array([1, 1])
    C = np.array([Capacity['Air'] / 2])
    f = 1
    y = 1
    TCd3 = {'A': A, 'G': G, 'b': b, 'C': C, 'f': f, 'y': y}
    
    TCd = {'0': TCd0,
           '1': TCd1,
           '2': TCd2,
           '3': TCd3}
    
    AssX = [[TCd['0'], nt, TCd['1'], 0],
            [TCd['1'], 1, TCd['2'], 1],
            [TCd['1'], 2, TCd['3'], 0]]
    
    AssX = np.array([[0, nt - 1, 1, 0],
                     [1, 1, 2, 1],
                     [1, 2, 3, 0]])
    
    TCa = dm4bem.TCAss(TCd, AssX)
return x
