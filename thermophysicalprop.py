"""
Created on Mi Oct 27  16:03:01 2021

@author: beberlen
"""
import numpy as np
import pandas as pd


def thphprop(BCdf):
    """
    Parameters
    ----------
    BCdf : data frame of building characteristics
        DESCRIPTION.
        Data Frame of building characteristics. Example:
                BCdf = ['Element Code', 'Element Type', 'Material 1', 'Material 2', 'Material 3', 'Length', 'Width',
                'Height', 'Thickness 1', 'Thickness 2', Thickness 3', 'Surface', 'Volume', 'Slope', 'Azimuth', ]

    Returns
    -------
    Bdf : data frame
        DESCRIPTION.
        data frame of the Building characteristics with associated thermophysical properties
                Bdf = ['Element Code', 'Element Type', 'Material 1', 'Material 2', 'Material 3', 'Length', 'Width',
                'Height', 'Thickness 1', 'Thickness 2', Thickness 3', 'Surface', 'Volume', 'Slope', 'Azimuth',
                'Density 1', 'specific heat 1', 'conductivity 1', 'LW emissivity 1', 'SW transmittance 1',
                'SW absorptivity 1', 'albedo 1', 'Density 2', 'specific heat 2', 'conductivity 2', 'LW emissivity 2',
                'SW transmittance 2', 'SW absorptivity 2', 'albedo 2', 'Density 3', 'specific heat 3', 'conductivity 3',
                'LW emissivity 3', 'SW transmittance 3', 'SW absorptivity 3', 'albedo 3']
    """

    # Thermo-physical and radiative properties - source data frame
    # ----------------------------------------------------------

    """ Incropera et al. (2011) Fundamentals of heat and mass transfer, 7 ed,
        Table A3,
            concrete (stone mix) p. 993
            insulation polystyrene extruded (R-12) p.990
            glass plate p.993
            Clay tile, hollow p. 989
            
        EngToolbox Emissivity Coefficient Materials, Glass, pyrex
        EngToolbox Emissivity Coefficient Materials, Clay
        EngToolbox Absorbed Solar Radiation by Surface Color, white smooth surface
        EngToolbox Optical properties of some typical glazing mat Window glass
        EngToolbox Absorbed Solar Radiation by Material, Tile, clay red
        """
    thphp = {'Density': [2300, 55, 2500, 1.2, '-'],          # kg/m³
             'Specific_Heat': [880, 1210, 750, 1000, '-'],    # J/kg.K
             'Conductivity': [1.4, 0.027, 1.4, '-', 0.52],       # W/m.K
             'LW_Emissivity': [0.9, 0, 0.9, 0, 0.91],
             'SW_Transmittance': [0, 0, 0.83, 1, 0],
             'SW_Absorptivity': [0.25, 0.25, 0.1, 0, 0.64],
             'Albedo': [0.75, 0.75, 0.07, 0, 0.36]}              # albedo + SW transmission + SW absorptivity = 1

    thphp = pd.DataFrame(thphp, index=['Concrete', 'Insulation', 'Glass', 'Air', 'Tiles'])

    # add empty columns for thermo-physical properties
    BCdf.assign(columns=['density_1', 'specific_heat_1', 'conductivity_1', 'LW_emissivity_1', 'SW_transmittance_1',
                         'SW_absorptivity_1', 'albedo_1', 'density_2', 'specific_heat_2', 'conductivity_2',
                         'LW_emissivity_2', 'SW_transmittance_2', 'SW_absorptivity_2', 'albedo_2', 'density_3',
                         'specific_heat_3', 'conductivity_3', 'LW_emissivity_3', 'SW_transmittance_3',
                         'SW_absorptivity_3', 'albedo_3'])

    # fill columns with properties for the given materials 1-3 of each element
    for i in BCdf['Material_1']:
        for j in thphp[index]:
            if BCdf.Material_1[i] == thphp.index[j]:
                BCdf.loc[i] = pd.Series({'density_1': thphp.Density[j],
                                         'specific_heat_1': thphp.Specific_Heat[j],
                                         'conductivity_1': thphp.Conductivity[j],
                                         'LW_emissivity_1': thphp.LW_Emissivity[j],
                                         'SW_transmittance_1': thphp.SW_Transmittance[j],
                                         'SW_absorptivity_1': thphp.SW_Absorptivity[j],
                                         'albedo_1': thphp.Albedo[j]})

    for i in BCdf['Material_2']:
        for j in thphp[index]:
            if BCdf.Material_2[i] == thphp.index[j]:
                BCdf.loc[i] = pd.Series({'density_2': thphp.Density[j],
                                         'specific_heat_2': thphp.Specific_Heat[j],
                                         'conductivity_2': thphp.Conductivity[j],
                                         'LW_emissivity_2': thphp.LW_Emissivity[j],
                                         'SW_transmittance_2': thphp.SW_Transmittance[j],
                                         'SW_absorptivity_2': thphp.SW_Absorptivity[j],
                                         'albedo_2': thphp.Albedo[j]})

    for i in BCdf['Material_3']:
        for j in thphp[index]:
            if BCdf.Material_3[i] == thphp.index[j]:
                BCdf.loc[i] = pd.Series({'density_3': thphp.Density[j],
                                         'specific_heat_3': thphp.Specific_Heat[j],
                                         'conductivity_3': thphp.Conductivity[j],
                                         'LW_emissivity_3': thphp.LW_Emissivity[j],
                                         'SW_transmittance_3': thphp.SW_Transmittance[j],
                                         'SW_absorptivity_3': thphp.SW_Absorptivity[j],
                                         'albedo_3': thphp.Albedo[j]})

    return BCdf