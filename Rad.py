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
import pandas as pd

import dm4bem


def rad(bcp, albedo_sur, latitude):

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

    return Φt
