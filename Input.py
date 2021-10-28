"""
Created 28th October 2021

Author: C.Gerike-Roberts

Code for defining the human variables.
"""


def inputs(bc):
    Kp = 1e4    #factor for HVAC
    dt = 5      #s - time step for solver
    T_set = pd.DataFrame([{'cooling': (26+273.15), 'heating': (22+273.15)}])  # K - temperature set points
    mesh = 1
    Tm = 20+273.15     #K - Mean temperature for radiative exchange
    ACH = 1     #h*-1 - no. of air changes in volume per hour
    h = pd.DataFrame([{'in': 4., 'out': 10}])   #W/m² K - convection coefficients
    Vdot = bc.getvalue(1, 'Volume') * ACH / 3600   #m3/s - volume flow rate due to air changes
    return Kp, dt, T_set, mesh, Tm, h, Vdot
