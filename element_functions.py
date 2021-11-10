"""
Created on We Nov 10 2021

@author: l.beber

File with all the element functions
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


def ventilation(V, V_dot, Kp):
    """
    Input:
    V, Volume of the room (from bcp)
    V_dot
    Kp
    Output:
    TCd, a dictionary of the all the matrices describing the thermal circuit of the ventilation
    """
    Gv = V_dot * 1.2 * 1000  # Va_dot * air['Density'] * air['Specific heat']
    A = np.array([[1],
                  [1]])
    G = np.diag(np.hstack([Gv, Kp]))
    b = np.array([1, 1])
    C = np.array((1.2 * 1000 * V) / 2)
    f = 1
    y = 1

    TCd = {'A': A, 'G': G, 'b': b, 'C': C, 'f': f, 'y': y}

    return TCd


def solid_wall_w_ins(bcp_r, h):
    """Input:
    bcp_r, one row of the bcp dataframe
    h, convection dataframe
    Output: TCd, a dictionary of the all the matrices of one thermal circuit describing a solid wall with insulation
    """
    # Thermal conductances
    # Conduction
    G_cd_cm = bcp_r['conductivity_1'] / bcp_r['Thickness_1'] * bcp_r['Surface']  # concrete
    G_cd_in = bcp_r['conductivity_2'] / bcp_r['Thickness_2'] * bcp_r['Surface']  # insulation

    # Convection
    Gw = h * bcp_r['Surface']  # wall

    # Thermal capacities
    Capacity_cm = bcp_r['density_1'] * bcp_r['specific_heat_1'] * bcp_r['Surface'] * bcp_r['Thickness_1']
    Capacity_in = bcp_r['density_2'] * bcp_r['specific_heat_2'] * bcp_r['Surface'] * bcp_r['Thickness_2']

    # Thermal network
    # ---------------
    nq = 1 + 2 * (int(bcp_r['Mesh_1']) + int(bcp_r['Mesh_2']))
    nt = 1 + 2 * (int(bcp_r['Mesh_1']) + int(bcp_r['Mesh_2']))

    A = np.eye(nq + 1, nt)
    A = -np.diff(A, 1, 0).T

    nc = int(bcp_r['Mesh_1'])
    ni = int(bcp_r['Mesh_2'])
    Gcm = 2 * nc * G_cd_cm
    Gcm = 2 * nc * np.array(Gcm)
    Gim = 2 * ni * G_cd_in
    Gim = 2 * ni * np.array(Gim)
    G = np.diag(np.hstack([Gw['out'], Gcm, Gim]))

    b = np.zeros(nq)
    b[0] = 1

    Ccm = Capacity_cm / nc * np.mod(range(0, 2 * nc), 2)
    Cim = Capacity_in / ni * np.mod(range(0, 2 * ni), 2)
    C = np.diag(np.hstack([Ccm, Cim, 0]))

    f = np.zeros(nt)
    f[0] = f[-1] = 1

    y = np.zeros(nt)

    TCd = {'A': A, 'G': G, 'b': b, 'C': C, 'f': f, 'y': y}

    return TCd


def window(bcp_r, h):
    """Input:
    bcp_r, one row of the bcp dataframe
    h, convection dataframe
    Output: TCd, a dictionary of the all the matrices of one thermal circuit describing a solid wall with insulation
    """
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


def susp_floor(bcp_r, h, V):
    """Input:
    bcp_r, one row of the bcp dataframe
    h, convection dataframe
    V, Volume of the room from bcp
    Output: TCd, a dictionary of the all the matrices of one thermal circuit describing a suspended floor
    """
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


def flat_roof_w_in(bcp_r, h):
    """Input:
    bcp_r, one row of the bcp dataframe
    h, convection dataframe
    Output: TCd, a dictionary of the all the matrices of one thermal circuit describing a flat roof with insulation
    """
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
