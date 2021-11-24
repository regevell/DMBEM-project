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
import copy

# global constants
σ = 5.67e-8     # W/m².K⁴ Stefan-Bolzmann constant

# Define building characteristics
# xlfile = 'Building Characteristics_Test'
bc = TCM_funcs.building_characteristics()

# Define Inputs
Kpc = 50
Kpf = 1e-3
Kph = 1e4                                                                     # factor for HVAC
dt = 4                                                                        # s - time step for solver
T_set = pd.DataFrame([{'cooling': 26, 'heating': 20}])                        # C - temperature set points
Tm = 20 + 273.15                                                              # K - Mean temperature for radiative exchange
ACH = 1                                                                       # h*-1 - no. of air changes in volume per hour
h = pd.DataFrame([{'in': 4., 'out': 10}])                                     # W/m² K - convection coefficients
V = bc.Volume[0]                                                              # m³
Vdot = V * ACH / 3600                                                         # m³/s - volume flow rate due to air changes
albedo_sur = 0.2                                                              # albedo for the surroundings
latitude = 45
Qa = 100                                                                      # auxiliary heat flow
Tisp = 20
DeltaT = 5
DeltaBlind = 2
WF = 'GBR_ENG_RAF.Lyneham.037400_TMYx.2004-2018.epw'
t_start = '2000-01-03 12:00:00'
t_end = '2000-01-04 18:00:00'
Tg = 10                     # ground temperature

# Add thermo-physical properties
bcp = TCM_funcs.thphprop(bc)

# Determine solar radiation for each element
rad_surf_tot, t = TCM_funcs.rad(bcp, albedo_sur, latitude, dt, WF, t_start, t_end)

# Thermal Circuits
TCd = {}
TCd.update({str(0): TCM_funcs.indoor_air(bcp.Surface, h, V, Qa, rad_surf_tot)})  # inside air
TCd.update({str(1): TCM_funcs.ventilation(V, Vdot, Kpf, T_set, rad_surf_tot)})  # ventilation and heating
uc = 2                                                          # variable to track how many heat flows have been used
IG = np.zeros([rad_surf_tot.shape[0], 1])                                                          # set the radiation entering through windows to zero
for i in range(0, len(bcp)):
    if bcp.Element_Type[i] == 'Solid Wall w/In':
        TCd_i, uca = TCM_funcs.solid_wall_w_ins(bcp.loc[i, :], h, rad_surf_tot, uc)
        TCd.update({str(i+2): TCd_i})
    elif bcp.Element_Type[i] == 'DG':
        TCd_i, uca, IGR = TCM_funcs.window(bcp.loc[i, :], h, rad_surf_tot, uc)
        TCd.update({str(i+2): TCd_i})
        IG = IGR / 7                                      # update total radiation coming through windows
    elif bcp.Element_Type[i] == 'Suspended Floor':
        TCd_i, uca = TCM_funcs.susp_floor(bcp.loc[i, :], h, V, rad_surf_tot, uc, Tg)
        TCd.update({str(i+2): TCd_i})
    elif bcp.Element_Type[i] == 'Flat Roof w/In':
        TCd_i, uca = TCM_funcs.flat_roof_w_in(bcp.loc[i, :], h, rad_surf_tot, uc)
        TCd.update({str(i+2): TCd_i})
    uc = uca                                                    # update heat flow tracker
# TCd = pd.DataFrame(TCd)

TCd_f = copy.deepcopy(TCd)

for i in range(0, len(bcp)):
    if bcp.Element_Type[i] == 'Solid Wall w/In':
        TCd_i = TCM_funcs.indoor_rad(bcp.loc[i, :], TCd_f[str(i+2)], IG)
        TCd_f[str(i+2)] = TCd_i
    elif bcp.Element_Type[i] == 'Suspended Floor':
        TCd_i = TCM_funcs.indoor_rad(bcp.loc[i, :], TCd_f[str(i+2)], IG)
        TCd_f[str(i + 2)] = TCd_i
    elif bcp.Element_Type[i] == 'Flat Roof w/In':
        TCd_i = TCM_funcs.indoor_rad(bcp.loc[i, :], TCd_f[str(i+2)], IG)
        TCd_f[str(i + 2)] = TCd_i

TCd_h = copy.deepcopy(TCd_f)
TCd_c = copy.deepcopy(TCd)

for i in range(0, len(bcp)):
    if bcp.Element_Type[i] == 'Solid Wall w/In':
        TCd_i = TCM_funcs.indoor_rad_c(TCd_c[str(i+2)])
        TCd_c[str(i+2)] = TCd_i
    elif bcp.Element_Type[i] == 'Suspended Floor':
        TCd_i = TCM_funcs.indoor_rad_c(TCd_c[str(i+2)])
        TCd_c[str(i + 2)] = TCd_i
    elif bcp.Element_Type[i] == 'Flat Roof w/In':
        TCd_i = TCM_funcs.indoor_rad_c(TCd_c[str(i+2)])
        TCd_c[str(i + 2)] = TCd_i

TCd_c[str(1)] = TCM_funcs.ventilation(V, Vdot, Kpc, T_set, rad_surf_tot)
TCd_h[str(1)] = TCM_funcs.ventilation(V, Vdot, Kph, T_set, rad_surf_tot)

TCd_f = pd.DataFrame(TCd_f)
TCd_c = pd.DataFrame(TCd_c)
TCd_h = pd.DataFrame(TCd_h)

u, rad_surf_tot = TCM_funcs.u_assembly(TCd_f, rad_surf_tot)
u_c, rad_surf_tot = TCM_funcs.u_assembly_c(TCd_c, rad_surf_tot)
AssX = TCM_funcs.assembly(TCd_f)

TCd_f = TCd_f.drop('Q')
TCd_f = TCd_f.drop('T')
TCd_c = TCd_c.drop('Q')
TCd_c = TCd_c.drop('T')
TCd_h = TCd_h.drop('Q')
TCd_h = TCd_h.drop('T')

TCd_f = pd.DataFrame.to_dict(TCd_f)
TCd_c = pd.DataFrame.to_dict(TCd_c)
TCd_h = pd.DataFrame.to_dict(TCd_h)


TCAf = dm4bem.TCAss(TCd_f, AssX)
TCAc = dm4bem.TCAss(TCd_c, AssX)
TCAh = dm4bem.TCAss(TCd_h, AssX)

TCM_funcs.solver(TCAf, TCAc, TCAh, dt, u, u_c, t, Tisp, DeltaT, DeltaBlind, Kpc, Kph, rad_surf_tot)
