"""
Started on 28 October 2021.

Authors: L.Beber, E.Regev, C.Gerike-Roberts

Code which models the dynamic thermal transfer in a building.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import dm4bem
import BuildingCharacteristics

# Define building characteristics

bc = BuildingCharacteristics.BuildingCharacteristics()
print(bc)

# Define thermal properties
