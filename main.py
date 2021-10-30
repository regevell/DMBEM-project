"""
Started on 28 October 2021.

Authors: L.Beber, E.Regev, C.Gerike-Roberts

Code which models the dynamic thermal transfer in a building.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import Weather
import dm4bem
import BuildingCharacteristics
import thermophysicalprop
import solid_wall_w_ins

# global constants
σ = 5.67e-8     # W/m².K⁴ Stefan-Bolzmann constant

# Define building characteristics

bc = BuildingCharacteristics.building_characteristics()

# Define Inputs
Kp = 1e4  # factor for HVAC
dt = 5  # s - time step for solver
T_set = pd.DataFrame([{'cooling': (26 + 273.15), 'heating': (22 + 273.15)}])  # K - temperature set points
mesh = 1
Tm = 20 + 273.15  # K - Mean temperature for radiative exchange
ACH = 1  # h*-1 - no. of air changes in volume per hour
h = pd.DataFrame([{'in': 4., 'out': 10}])  # W/m² K - convection coefficients
Vdot = bc.Volume[4] * ACH / 3600  # m3/s - volume flow rate due to air changes

# Add thermo-physical properties

bcp = thermophysicalprop.thphprop(bc)

# Thermal Circuits
TCd = np.zeros(len(bcp))

for i in range(0, len(bcp)):
    if bcp.Element_Type[i] == 'Solid Wall w/In':
        TCd[i] = solid_wall_w_ins.solid_wall_w_ins(bcp.loc[i, :], h)
    # elif bcp.Element_Type[i] == 'DG':
    #     TCd[i] = DG(bcp.loc[i, :], h)
    # elif bcp.Element_Type[i] == 'Suspended Floor':
    #     TCd[i] = susp_floor(bcp.loc[i, :], h)
    # elif bcp.Element_Type[i] == 'Flat Roof 1/In':
    #     TCd[i] = flat_roof_w_in(bcp.loc[i, :], h)

print(TCd)
# Define weather

weather, meta = Weather.weather_input()
