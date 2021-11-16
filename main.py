"""
Started on 28 October 2021.

Authors: L.Beber, E.Regev, C.Gerike-Roberts

Code which models the dynamic thermal transfer in a building.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import element_functions
import Weather
import dm4bem
import BuildingCharacteristics
import thermophysicalprop
# import Assembly

# global constants

σ = 5.67e-8     # W/m².K⁴ Stefan-Bolzmann constant

# Define building characteristics

bc = BuildingCharacteristics.building_characteristics()

# Define Inputs
Kp = 1e4  # factor for HVAC
dt = 5  # s - time step for solver
T_set = pd.DataFrame([{'cooling': (26 + 273.15), 'heating': (22 + 273.15)}])  # K - temperature set points
# mesh = 1
Tm = 20 + 273.15  # K - Mean temperature for radiative exchange
ACH = 1  # h*-1 - no. of air changes in volume per hour
h = pd.DataFrame([{'in': 4., 'out': 10}])  # W/m² K - convection coefficients
V = bc.Volume[4]
Vdot = V * ACH / 3600  # m3/s - volume flow rate due to air changes

# Add thermo-physical properties

bcp = thermophysicalprop.thphprop(bc)

# Thermal Circuits
TCd = {}
TCd.update({str(0): element_functions.indoor_air(bcp.Surface, h, V)})  # inside air
TCd.update({str(1): element_functions.ventilation(V, Vdot, Kp)})  # ventilation and heating
for i in range(0, len(bcp)):
    if bcp.Element_Type[i] == 'Solid Wall w/In':
        TCd.update({str(i+2): element_functions.solid_wall_w_ins(bcp.loc[i, :], h)})
    elif bcp.Element_Type[i] == 'SinG':
        TCd.update({str(i+2): element_functions.window(bcp.loc[i, :], h)})
    elif bcp.Element_Type[i] == 'Suspended Floor':
        TCd.update({str(i+2): element_functions.susp_floor(bcp.loc[i, :], h, V)})
    elif bcp.Element_Type[i] == 'Flat Roof 1/In':
        TCd.update({str(i+2): element_functions.flat_roof_w_in(bcp.loc[i, :], h)})

# TCd = pd.DataFrame(TCd)
# Define weather

weather, meta = Weather.weather_input()

