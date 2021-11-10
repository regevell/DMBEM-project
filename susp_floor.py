"""
Created on We Nov 10 2021

@author: l.beber

Input:
    bcp_r, one row of the bcp dataframe
    h, convection dataframe
    V, Volume of the room from bcp
Output: TCd, a dictionary of the all the matrices of one thermal circuit describing a suspended floor
"""
import numpy as np


def susp_floor(bcp_r, h, V):
    A = np.array([[1, 0, 0, 0],
                  [-1, 1, 0, 0],
                  [0, -1, 1, 0],
                  [0, 0, -1, 1]])
    Gw = h * bcp_r['Surface']
    G_cd = bcp_r['conductivity_1'] / bcp_r['Thickness_1'] * bcp_r['Surface']  # wood
    G = np.diag(np.hstack(
        [Gw['in'], Gw['in'], G_cd, G_cd]))
    b = np.array([1, 0, 0, 0])
    Capacity_w = bcp_r['density_1'] * bcp_r['specific_heat_1'] * bcp_r['Surface'] * bcp_r['Thickness_1']  # wood
    Capacity_a = bcp_r['density_2'] * bcp_r['specific_heat_2'] * V  # air
    C = np.diag([Capacity_a, 0, Capacity_w, 0])
    f = np.array([0, 0, 0, 1])
    y = np.array([0, 0, 0, 0])

    TCd = {'A': A, 'G': G, 'b': b, 'C': C, 'f': f, 'y': y}

    return TCd
