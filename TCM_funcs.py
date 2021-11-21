"""
Created on We Nov 10 2021

@author: L. Beber, C. Gerike-Roberts, E. Regev

File with all the functions
"""

import numpy as np
import pandas as pd
import dm4bem

def building_characteristics():

    """
    This code is designed to read an excel file which contains the characteristics of the building
    and create a data frame from it.
    """

    bc = pd.read_csv(r'Building Characteristics.csv', na_values=["N"], keep_default_na=True)

    return bc


def thphprop(BCdf):
    """
    Parameters
    ----------
    BCdf : data frame of building characteristics
        DESCRIPTION.
        Data Frame of building characteristics. Example:
                BCdf = ['Element Code', 'Element Type', 'Material 1', 'Material 2', 'Material 3', 'Length', 'Width',
                'Height', 'Thickness 1', 'Thickness 2', Thickness 3', 'Surface', 'Volume', 'Slope', 'Azimuth', ]

    Returns
    -------
    Bdf : data frame
        DESCRIPTION.
        data frame of the Building characteristics with associated thermophysical properties
                Bdf = ['Element Code', 'Element Type', 'Material 1', 'Material 2', 'Material 3', 'Length', 'Width',
                'Height', 'Thickness 1', 'Thickness 2', Thickness 3', 'Surface', 'Volume', 'Slope', 'Azimuth',
                'Density 1', 'specific heat 1', 'conductivity 1', 'LW emissivity 1', 'SW transmittance 1',
                'SW absorptivity 1', 'albedo 1', 'Density 2', 'specific heat 2', 'conductivity 2', 'LW emissivity 2',
                'SW transmittance 2', 'SW absorptivity 2', 'albedo 2', 'Density 3', 'specific heat 3', 'conductivity 3',
                'LW emissivity 3', 'SW transmittance 3', 'SW absorptivity 3', 'albedo 3']
    """

    # Thermo-physical and radiative properties - source data frame
    # ----------------------------------------------------------

    """ Incropera et al. (2011) Fundamentals of heat and mass transfer, 7 ed,
        Table A3,
            concrete (stone mix) p. 993
            insulation polystyrene extruded (R-12) p.990
            glass plate p.993
            Clay tile, hollow p.989
            Wood, oak p.989

        EngToolbox Emissivity Coefficient Materials, Glass, pyrex
        EngToolbox Emissivity Coefficient Materials, Clay
        EngToolbox Emissivity Coefficient Materials, Wood Oak, planned
        EngToolbox Absorbed Solar Radiation by Surface Color, white smooth surface
        EngToolbox Optical properties of some typical glazing mat Window glass
        EngToolbox Absorbed Solar Radiation by Material, Tile, clay red
        EngToolbox Absorbed Solar Radiation by Surface Color, Green, red and brown
        """
    thphp = {'Material': ['Concrete', 'Insulation', 'Glass', 'Air', 'Tile', 'Wood'],
             'Density': [2300, 55, 2500, 1.2, None, 720],  # kg/m³
             'Specific_Heat': [880, 1210, 750, 1000, None, 1255],  # J/kg.K
             'Conductivity': [1.4, 0.027, 1.4, None, 0.52, 0.16],  # W/m.K
             'LW_Emissivity': [0.9, 0, 0.9, 0, 0.91, 0.885],
             'SW_Transmittance': [0, 0, 0.83, 1, 0, 0],
             'SW_Absorptivity': [0.25, 0.25, 0.1, 0, 0.64, 0.6],
             'Albedo': [0.75, 0.75, 0.07, 0, 0.36, 0.4]}  # albedo + SW transmission + SW absorptivity = 1

    thphp = pd.DataFrame(thphp)

    # add empty columns for thermo-physical properties
    BCdf = BCdf.reindex(columns=BCdf.columns.to_list() + ['rad_s', 'density_1', 'specific_heat_1', 'conductivity_1',
                                                          'LW_emissivity_1', 'SW_transmittance_1', 'SW_absorptivity_1',
                                                          'albedo_1', 'density_2', 'specific_heat_2', 'conductivity_2',
                                                          'LW_emissivity_2', 'SW_transmittance_2', 'SW_absorptivity_2',
                                                          'albedo_2', 'density_3', 'specific_heat_3', 'conductivity_3',
                                                          'LW_emissivity_3', 'SW_transmittance_3', 'SW_absorptivity_3',
                                                          'albedo_3'])


    # fill columns with properties for the given materials 1-3 of each element
    for i in range(0, len(BCdf)):
        for j in range(0, len(thphp['Material'])):
            if BCdf.loc[i, 'Material_1'] == thphp.Material[j]:
                BCdf.loc[i, 'density_1'] = thphp.Density[j]
                BCdf.loc[i, 'specific_heat_1'] = thphp.Specific_Heat[j]
                BCdf.loc[i, 'conductivity_1'] = thphp.Conductivity[j]
                BCdf.loc[i, 'LW_emissivity_1'] = thphp.LW_Emissivity[j]
                BCdf.loc[i, 'SW_transmittance_1'] = thphp.SW_Transmittance[j]
                BCdf.loc[i, 'SW_absorptivity_1'] = thphp.SW_Absorptivity[j]
                BCdf.loc[i, 'albedo_1'] = thphp.Albedo[j]

        for j in range(0, len(thphp['Material'])):
            if BCdf.loc[i, 'Material_2'] == thphp.Material[j]:
                BCdf.loc[i, 'density_2'] = thphp.Density[j]
                BCdf.loc[i, 'specific_heat_2'] = thphp.Specific_Heat[j]
                BCdf.loc[i, 'conductivity_2'] = thphp.Conductivity[j]
                BCdf.loc[i, 'LW_emissivity_2'] = thphp.LW_Emissivity[j]
                BCdf.loc[i, 'SW_transmittance_2'] = thphp.SW_Transmittance[j]
                BCdf.loc[i, 'SW_absorptivity_2'] = thphp.SW_Absorptivity[j]
                BCdf.loc[i, 'albedo_2'] = thphp.Albedo[j]

        for j in range(0, len(thphp['Material'])):
            if BCdf.loc[i, 'Material_3'] == thphp.Material[j]:
                BCdf.loc[i, 'density_3'] = thphp.Density[j]
                BCdf.loc[i, 'specific_heat_3'] = thphp.Specific_Heat[j]
                BCdf.loc[i, 'conductivity_3'] = thphp.Conductivity[j]
                BCdf.loc[i, 'LW_emissivity_3'] = thphp.LW_Emissivity[j]
                BCdf.loc[i, 'SW_transmittance_3'] = thphp.SW_Transmittance[j]
                BCdf.loc[i, 'SW_absorptivity_3'] = thphp.SW_Absorptivity[j]
                BCdf.loc[i, 'albedo_3'] = thphp.Albedo[j]

    return BCdf

def indoor_air(bcp_sur, h, V, Qa, rad_surf_tot):
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
    Q = np.zeros((rad_surf_tot.shape[0], nt))
    Q[0] = Qa
    Q[:, 1:nt] = 'NaN'
    T = np.zeros((rad_surf_tot.shape[0], nq))
    T[:, 0:nq] = 'NaN'

    TCd = {'A': A, 'G': G, 'b': b, 'C': C, 'f': f, 'y': y, 'Q': Q, 'T': T}

    return TCd


def ventilation(V, V_dot, Kp, T_set, rad_surf_tot):
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
    Q = np.zeros((rad_surf_tot.shape[0], 1))
    Q[:, :] = 'NaN'
    T = np.zeros((rad_surf_tot.shape[0], 2))
    T[:, 0] = rad_surf_tot['To']
    T[:, 1] = T_set['heating']

    TCd = {'A': A, 'G': G, 'b': b, 'C': C, 'f': f, 'y': y, 'Q': Q, 'T': T}

    return TCd


def solid_wall_w_ins(bcp_r, h, rad_surf_tot, uc):
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

    Q = np.zeros((rad_surf_tot.shape[0], nt))
    Q[:, 0] = bcp_r['SW_absorptivity_1'] * bcp_r['Surface'] * rad_surf_tot[str(uc)]
    Q[:, (nt-1)] = -1
    uca = uc+1
    Q[:, 1:(nt - 1)] = 'NaN'

    T = np.zeros((rad_surf_tot.shape[0], nq))
    T[:, 0] = rad_surf_tot['To']
    T[:, 1:nq] = 'NaN'

    TCd = {'A': A, 'G': G, 'b': b, 'C': C, 'f': f, 'y': y, 'Q': Q, 'T': T}

    return TCd, uca


def window(bcp_r, h, rad_surf_tot, uc):
    """Input:
    bcp_r, one row of the bcp dataframe
    h, convection dataframe
    Output: TCd, a dictionary of the all the matrices of one thermal circuit describing a solid wall with insulation
    """
    nq = 2 * (int(bcp_r['Mesh_1']))
    nt = 2 * (int(bcp_r['Mesh_1']))

    A = np.array([[1, 0],
                  [-1, 1]])
    Ggo = h['out'] * bcp_r['Surface']
    Ggs = 1 / (1 / Ggo + 1 / (2 * bcp_r['conductivity_1']))
    G = np.diag(np.hstack([Ggs, 2 * bcp_r['conductivity_1']]))
    b = np.array([1, 0])
    C = np.diag([bcp_r['density_1'] * bcp_r['specific_heat_1'] * bcp_r['Surface'] * bcp_r['Thickness_1'], 0])
    f = np.array([1, 0])
    y = np.array([0, 0])

    Q = np.zeros((rad_surf_tot.shape[0], nt))
    IG_surface = bcp_r['Surface'] * rad_surf_tot[str(uc)]
    IGR = bcp_r['SW_transmittance_1'] * bcp_r['Surface'] * rad_surf_tot[str(uc)]
    Q[:, 0] = bcp_r['SW_absorptivity_1'] * IG_surface
    uca = uc + 1
    Q[:, 1:nt] = 'NaN'

    T = np.zeros((rad_surf_tot.shape[0], nq))
    T[:, 0] = rad_surf_tot['To']
    T[:, 1:nq] = 'NaN'

    TCd = {'A': A, 'G': G, 'b': b, 'C': C, 'f': f, 'y': y, 'Q': Q, 'T': T}

    return TCd, uca, IGR


def susp_floor(bcp_r, h, V, rad_surf_tot, uc):
    """Input:
    bcp_r, one row of the bcp dataframe
    h, convection dataframe
    V, Volume of the room from bcp
    Output: TCd, a dictionary of the all the matrices of one thermal circuit describing a suspended floor
    """
    nq = 2 * (int(bcp_r['Mesh_1']) + int(bcp_r['Mesh_2']))
    nt = 2 * (int(bcp_r['Mesh_1']) + int(bcp_r['Mesh_2']))

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

    Q = np.zeros((rad_surf_tot.shape[0], nt))
    Q[:, 0] = bcp_r['SW_absorptivity_1'] * bcp_r['Surface'] * rad_surf_tot[str(uc)]
    Q[:, (nt - 1)] = -1
    uca = uc + 1
    Q[:, 1:(nt - 1)] = 'NaN'

    T = np.zeros((rad_surf_tot.shape[0], nq))
    T[:, 0] = rad_surf_tot['To']
    T[:, 1:nq] = 'NaN'

    TCd = {'A': A, 'G': G, 'b': b, 'C': C, 'f': f, 'y': y, 'Q': Q, 'T': T}

    return TCd, uca


def flat_roof_w_in(bcp_r, h, rad_surf_tot, uc):
    """Input:
    bcp_r, one row of the bcp dataframe
    h, convection dataframe
    Output: TCd, a dictionary of the all the matrices of one thermal circuit describing a flat roof with insulation
    """
    nq = 2 * (int(bcp_r['Mesh_1']))
    nt = 2 * (int(bcp_r['Mesh_1']))

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

    Q = np.zeros((rad_surf_tot.shape[0], nt))
    Q[:, 0] = bcp_r['SW_absorptivity_1'] * bcp_r['Surface'] * rad_surf_tot[str(uc)]
    Q[:, (nt - 1)] = -1
    uca = uc + 1
    Q[:, 1:(nt - 1)] = 'NaN'

    T = np.zeros((rad_surf_tot.shape[0], nq))
    T[:, 0] = rad_surf_tot['To']
    T[:, 1:nq] = 'NaN'

    TCd = {'A': A, 'G': G, 'b': b, 'C': C, 'f': f, 'y': y, 'Q': Q, 'T': T}

    return TCd, uca


def rad(bcp, albedo_sur, latitude, dt):
    """
    Created on Wed Oct 27 15:19:32 2021

    @author: ellio
    """
    # Simulation with weather data
    # ----------------------------
    filename = 'GBR_ENG_RAF.Lyneham.037400_TMYx.2004-2018.epw'
    start_date = '2000-01-03 12:00:00'
    end_date = '2000-01-04 18:00:00'

    # Read weather data from Energyplus .epw file
    [data, meta] = dm4bem.read_epw(filename, coerce_year=None)
    weather = data[["temp_air", "dir_n_rad", "dif_h_rad"]]
    del data
    weather.index = weather.index.map(lambda t: t.replace(year=2000))
    weather = weather[(weather.index >= start_date) & (
        weather.index < end_date)]
    # Solar radiation on a tilted surface South
    Φt = {}
    for k in range(0, len(bcp)):
        surface_orientationS = {'slope': bcp.loc[k, 'Slope'],
                                'azimuth': bcp.loc[k, 'Azimuth'],
                                'latitude': latitude}
        rad_surf = dm4bem.sol_rad_tilt_surf(weather, surface_orientationS, albedo_sur)
        Φt.update({str(k+2): rad_surf.sum(axis=1)})

    Φt = pd.DataFrame(Φt)
    # Interpolate weather data for time step dt
    data = pd.concat([weather['temp_air'], Φt], axis=1)
    data = data.resample(str(dt) + 'S').interpolate(method='linear')
    data = data.rename(columns={'temp_air': 'To'})

    return data

def indoor_rad(bcp_r, TCd, IG):
    Q = TCd['Q']
    lim = np.shape(Q)[1]
    for i in range(0, lim):
        if Q[0, i] == -1:
            if np.isnan(bcp_r['SW_absorptivity_3']):
                if np.isnan(bcp_r['SW_absorptivity_2']):
                    Q[:, i] = bcp_r['SW_absorptivity_1'] * IG
                else:
                    Q[:, i] = bcp_r['SW_absorptivity_2'] * IG
            else:
                Q[:, i] = bcp_r['SW_absorptivity_3'] * IG
        else:
            print('no change')

    TCd['Q'] = Q               # replace Q in TCd with new Q

    return TCd

def u_assembly(TCd, rad_surf_tot):
    u = np.empty((len(rad_surf_tot), 1))         # create u matrix
    for i in range(0, TCd.shape[1]):
        TCd_i = TCd[str(i)]
        T = TCd_i['T']
        T = T[:, ~np.isnan(T).any(axis=0)]
        if np.shape(T)[1] == 0:
            print('No Temp')
        else:
            u = np.append(u, T, axis=1)

    u = np.delete(u, 0, 1)

    for j in range(0, TCd.shape[1]):
        TCd_j = TCd[str(j)]
        Q = TCd_j['Q']
        Q = Q[:, ~np.isnan(Q).any(axis=0)]
        if np.shape(Q)[1] == 0:
            print('No Heat Flow')
        else:
            u = np.append(u, Q, axis=1)

    return u