"""
Created on  Nov 3 2021

@author: l.beber

Input:
    bcp_sur, surface column of bcp dataframe
    h, convection dataframe
    V, Volume of the room (from bcp)
Output: TCd, a dictionary of the all the matrices of the thermal circuit of the inside air
"""

import numpy as np


def indoor_air(bcp_sur, h, V):
    """
    Input:
    bcp_sur, surface column of bcp dataframe
    h, convection dataframe
    V, Volume of the room (from bcp)
    Output: TCd, a dictionary of the all the matrices of the thermal circuit of the inside air
    """
    nt = len(bcp_sur) + 1
    nq = len(bcp_sur)

    nq_ones = np.ones(nq)
    A = np.diag(-nq_ones)
    A = np.c_[nq_ones, A]

    G = np.zeros(nq)
    for i in range(0, len(G)):
        G[i] = h['in'] * bcp_sur[i]
    G = np.diag(G)
    b = np.zeros(nq)
    C = np.zeros(nt)
    C[-1] = (1.2 * 1000 * V) / 2  # Capacity air = Density*specific heat*V
    C = np.diag(C)
    f = np.zeros(nt)
    f[-1] = 1
    y = np.zeros(nt)
    y[-1] = 1
    TCd = {'A': A, 'G': G, 'b': b, 'C': C, 'f': f, 'y': y}

    return TCd
