"""
Started on 28 October 2021.

Authors: L.Beber, E.Regev, C.Gerike-Roberts

Code which models the dynamic thermal transfer in a building.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import TCM_funcs
import dm4bem

# global constants
σ = 5.67e-8     # W/m².K⁴ Stefan-Bolzmann constant

# Define building characteristics
bc = TCM_funcs.building_characteristics()

# Define Inputs
Kp = 1e4                                                                      # factor for HVAC
dt = 5                                                                        # s - time step for solver
T_set = pd.DataFrame([{'cooling': 26, 'heating': 20}])                        # C - temperature set points
Tm = 20 + 273.15                                                              # K - Mean temperature for radiative exchange
ACH = 1                                                                       # h*-1 - no. of air changes in volume per hour
h = pd.DataFrame([{'in': 4., 'out': 10}])                                     # W/m² K - convection coefficients
V = bc.Volume[4]                                                              # m³
Vdot = V * ACH / 3600                                                         # m³/s - volume flow rate due to air changes
albedo_sur = 0.2                                                              # albedo for the surroundings
latitude = 45
Qa = 0                                                                        # auxiliary heat flow

# Add thermo-physical properties
bcp = TCM_funcs.thphprop(bc)

# Determine solar radiation for each element
rad_surf_tot = TCM_funcs.rad(bcp, albedo_sur, latitude, dt)

# Thermal Circuits
TCd = {}
TCd.update({str(0): TCM_funcs.indoor_air(bcp.Surface, h, V, Qa, rad_surf_tot)})  # inside air
TCd.update({str(1): TCM_funcs.ventilation(V, Vdot, Kp, T_set, rad_surf_tot)})  # ventilation and heating
uc = 2                                                          # variable to track how many heat flows have been used
IG = 0                                                          # set the radiation entering through windows to zero
for i in range(0, len(bcp)):
    if bcp.Element_Type[i] == 'Solid Wall w/In':
        TCd_i, uca = TCM_funcs.solid_wall_w_ins(bcp.loc[i, :], h, rad_surf_tot, uc)
        TCd.update({str(i+2): TCd_i})
    elif bcp.Element_Type[i] == 'SinG':
        TCd_i, uca, IGR = TCM_funcs.window(bcp.loc[i, :], h, rad_surf_tot, uc)
        TCd.update({str(i+2): TCd_i})
        IG = IG+IGR                                         # update total radiation coming through windows
    elif bcp.Element_Type[i] == 'Suspended Floor':
        TCd_i, uca = TCM_funcs.susp_floor(bcp.loc[i, :], h, V, rad_surf_tot, uc)
        TCd.update({str(i+2): TCd_i})
    elif bcp.Element_Type[i] == 'Flat Roof w/In':
        TCd_i, uca = TCM_funcs.flat_roof_w_in(bcp.loc[i, :], h, rad_surf_tot, uc)
        TCd.update({str(i+2): TCd_i})
    uc = uca                                                    # update heat flow tracker
TCd = pd.DataFrame(TCd)

for i in range(0, len(bcp)):
    if bcp.Element_Type[i] == 'Solid Wall w/In':
        TCd_i = TCM_funcs.indoor_rad(bcp.loc[i, :], TCd[str(i+2)], IG)
        TCd[str(i+2)] = TCd_i
    elif bcp.Element_Type[i] == 'Suspended Floor':
        TCd_i = TCM_funcs.indoor_rad(bcp.loc[i, :], TCd[str(i+2)], IG)
        TCd[str(i + 2)] = TCd_i
    elif bcp.Element_Type[i] == 'Flat Roof w/In':
        TCd_i = TCM_funcs.indoor_rad(bcp.loc[i, :], TCd[str(i+2)], IG)
        TCd[str(i + 2)] = TCd_i

u = TCM_funcs.u_assembly(TCd, rad_surf_tot)

print(TCd)
