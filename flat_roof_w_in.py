"""
Created on We Nov 10 2021

@author: l.beber

Input:
    bcp_r, one row of the bcp dataframe
    h, convection dataframe
Output: TCd, a dictionary of the all the matrices of one thermal circuit describing a flat roof with insulation
"""
import numpy as np


def flat_roof_w_in(bcp_r, h):
    A = np.array([[-1, 0, 0],
                   [-1, 1, 0],
                   [0, -1, 1]])
    Gw = h * bcp_r['Surface']
    G_cd_in = bcp_r['conductivity_2'] / bcp_r['Thickness_2'] * bcp_r['Surface']  # insulation
    ni = int(bcp_r['Mesh_2'])
    Gim = 2 * ni * G_cd_in
    Gim = 2 * ni * np.array(Gim)
    G = np.diag(np.hstack([Gw['out'], Gim]))
    b = np.array([1, 0, 0])
    Capacity_i = bcp_r['density_2'] * bcp_r['specific_heat_2'] * bcp_r['Surface'] * bcp_r['Thickness_2']  # insulation
    C = np.diag([0, Capacity_i, 0])
    f = np.array([1, 0, 1])
    y = np.array([0, 0, 0])

    TCd = {'A': A, 'G': G, 'b': b, 'C': C, 'f': f, 'y': y}

    return TCd
