# -*- coding: utf-8 -*-
"""
Created on Wed Oct 27 15:19:32 2021

@author: ellio
"""
# Model code
# Thermal network
# ---------------
# Dissembled circuits

import numpy as np
import dm4bem

def indoor_air(GLW, Gw, h, wall, Capacity):
    # TCd3: Roof (in grey)

    A = np.array([[-1, 0, 0],
                  [-1, 1, 0],
                  [0, -1, 0]])
    G = np.diag(np.hstack([GLW, Gw['in'], h['in'] * wall['Surface']['Glass']]))
    b = np.array([1, 0, 0])
    C = np.diag([0, 0, Capacity['Air'] / 2])
    f = np.array([1, 0, 1])
    y = np.array([0, 0, 1])
    TCd1 = {'A': A, 'G': G, 'b': b, 'C': C, 'f': f, 'y': y}
    return TCd1


def weather_data():
    # Simulation with weather data
    # ----------------------------
    filename = 'FRA_Lyon.074810_IWEC.epw'
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
    surface_orientationS = {'slope': 90,
                            'azimuth': 0,
                            'latitude': 45}
    albedo = 0.2
    rad_surfS = dm4bem.sol_rad_tilt_surf(weather, surface_orientationS, albedo)
    rad_surfS['Φt1'] = rad_surfS.sum(axis=1)

    # Solar radiation on a tilted surface North
    surface_orientationN = {'slope': 90,
                            'azimuth': 180,
                            'latitude': 45}
    albedo = 0.2
    rad_surfN = dm4bem.sol_rad_tilt_surf(weather, surface_orientationN, albedo)
    rad_surfN['Φt1'] = rad_surfN.sum(axis=1)

    # Solar radiation on a tilted surface East
    surface_orientationE = {'slope': 90,
                            'azimuth': 0,
                            'latitude': 45}
    albedo = 0.2
    rad_surfE = dm4bem.sol_rad_tilt_surf(weather, surface_orientationE, albedo)
    rad_surfE['Φt1'] = rad_surfE.sum(axis=1)

    # Solar radiation on a tilted surface West
    surface_orientationW = {'slope': 90,
                            'azimuth': 0,
                            'latitude': 45}
    albedo = 0.2
    rad_surfW = dm4bem.sol_rad_tilt_surf(weather, surface_orientationW, albedo)
    rad_surfW['Φt1'] = rad_surfW.sum(axis=1)

    # Solar radiation on a tilted surface roof 1
    surface_orientationR1 = {'slope': 45,
                             'azimuth': 0,
                             'latitude': 45}
    albedo = 0.2
    rad_surfR1 = dm4bem.sol_rad_tilt_surf(weather, surface_orientationR1, albedo)
    rad_surfR1['Φt1'] = rad_surfR1.sum(axis=1)
   
    # Solar radiation on a tilted surface roof 2
    surface_orientationR2 = {'slope': 90,
                             'azimuth': 0,
                             'latitude': 45}
    albedo = 0.2
    rad_surfR2 = dm4bem.sol_rad_tilt_surf(weather, surface_orientationR2, albedo)
    rad_surfR2['Φt1'] = rad_surfR2.sum(axis=1)
    return weather, rad_surfS, rad_surfN, rad_surfE, rad_surfW, rad_surfR1, rad_surfR2


Wdata, radS, radN, radE, radW, radR1, radR2 = weather_data()
