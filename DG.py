"""
Created on We Nov 10 2021

@author: l.beber

Input:
    bcp_r, one row of the bcp dataframe
    h, convection dataframe
Output: TCd, a dictionary of the all the matrices of one thermal circuit describing a solid wall with insulation
"""
import numpy as np


def DG(bcp_r, h):
    A = np.array([[1, 0],
                  [-1, 1]])
    Ggo = h['out'] * bcp_r['Surface']
    Ggs = 1 / (1 / Ggo + 1 / (2 * bcp_r['conductivity_1']))
    G = np.diag(np.hstack([Ggs, 2 * bcp_r['conductivity_1']]))
    b = np.array([1, 0])
    C = np.diag([bcp_r['density_1'] * bcp_r['specific_heat_1'] * bcp_r['Surface'] * bcp_r['Thickness_1'], 0])
    f = np.array([1, 0])
    y = np.array([0, 0])
    TCd = {'A': A, 'G': G, 'b': b, 'C': C, 'f': f, 'y': y}

    return TCd
