"""
Created on  Nov 3 2021

@author: l.beber

Input:
    V, Volume of the room (from bcp)
    V_dot
    Kp
Output: TCd, a dictionary of the all the matrices describing the thermal circuit of the ventilation
"""
import numpy as np


def ventilation(V, V_dot, Kp):

    Gv = V_dot * 1.2 * 1000     # Va_dot * air['Density'] * air['Specific heat']
    A = np.array([[1],
                  [1]])
    G = np.diag(np.hstack([Gv, Kp]))
    b = np.array([1, 1])
    C = np.array((1.2 * 1000 * V) / 2)
    f = 1
    y = 1

    TCd = {'A': A, 'G': G, 'b': b, 'C': C, 'f': f, 'y': y}

    return TCd
