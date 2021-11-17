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

def rad(bcp):
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
    for k in range bcp...
        surface_orientationS = {'slope': bcp...,
                                'azimuth': bcp...,
                                'latitude': 45}
        albedo = bcp...
        rad_surf = dm4bem.sol_rad_tilt_surf(weather, surface_orientationS, albedo)
        rad_surf['Φt1'] = rad_surf.sum(axis=1)


    return weather, rad_surfS, rad_surfN, rad_surfE, rad_surfW, rad_surfR1, rad_surfR2


Wdata, radS, radN, radE, radW, radR1, radR2 = weather_data()
