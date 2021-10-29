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

# global constants
σ = 5.67e-8     # W/m².K⁴ Stefan-Bolzmann constant

# Define building characteristics

bc = BuildingCharacteristics.building_characteristics()
print(bc)

# Add thermo-physical properties

bcp = thermophysicalprop.thphprop(bc)
print(bcp)

# Define weather

weather, meta = Weather.weather_input()
