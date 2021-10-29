"""
Created on Thur Oct 28 2021 09:58:49 2021

@author: c.gerike-roberts

This code is designed to read an excel file which contains the characteristics of the building
and create a data frame from it.
"""

import pandas as pd


# import excel file

def building_characteristics():
    bc = pd.read_csv(r'Building Characteristics.csv')
    return bc
