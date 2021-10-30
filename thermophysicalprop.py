"""
Created on Mi Oct 27  16:03:01 2021

@author: beberlen
"""
import numpy as np
import pandas as pd
import BuildingCharacteristics
bc = BuildingCharacteristics.building_characteristics()

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
            Clay tile, hollow p.989
            Wood, oak p.989
            
        EngToolbox Emissivity Coefficient Materials, Glass, pyrex
        EngToolbox Emissivity Coefficient Materials, Clay
        EngToolbox Emissivity Coefficient Materials, Wood Oak, planned
        EngToolbox Absorbed Solar Radiation by Surface Color, white smooth surface
        EngToolbox Optical properties of some typical glazing mat Window glass
        EngToolbox Absorbed Solar Radiation by Material, Tile, clay red
        EngToolbox Absorbed Solar Radiation by Surface Color, Green, red and brown
        """
    thphp = {'Material': ['Concrete', 'Insulation', 'Glass', 'Air', 'Tile', 'Wood'],
             'Density': [2300, 55, 2500, 1.2, None, 720],            # kg/m³
             'Specific_Heat': [880, 1210, 750, 1000, None, 1255],    # J/kg.K
             'Conductivity': [1.4, 0.027, 1.4, None, 0.52, 0.16],    # W/m.K
             'LW_Emissivity': [0.9, 0, 0.9, 0, 0.91, 0.885],
             'SW_Transmittance': [0, 0, 0.83, 1, 0, 0],
             'SW_Absorptivity': [0.25, 0.25, 0.1, 0, 0.64, 0.6],
             'Albedo': [0.75, 0.75, 0.07, 0, 0.36, 0.4],}            # albedo + SW transmission + SW absorptivity = 1

    thphp = pd.DataFrame(thphp)

    # add empty columns for thermo-physical properties
    BCdf = BCdf.reindex(columns=BCdf.columns.to_list() + ['density_1', 'specific_heat_1', 'conductivity_1',
                                                          'LW_emissivity_1', 'SW_transmittance_1', 'SW_absorptivity_1',
                                                          'albedo_1', 'mesh_1', 'density_2', 'specific_heat_2', 'conductivity_2',
                                                          'LW_emissivity_2', 'SW_transmittance_2', 'SW_absorptivity_2',
                                                          'albedo_2', 'mesh_2', 'density_3', 'specific_heat_3', 'conductivity_3',
                                                          'LW_emissivity_3', 'SW_transmittance_3', 'SW_absorptivity_3',
                                                          'albedo_3'])
    print(BCdf)
    # fill columns with properties for the given materials 1-3 of each element
    for i in range(0, len(BCdf['Material_1'])):
        for j in range(0, len(thphp['Material'])):
            if BCdf.Material_1[i] == thphp.Material[j]:
                BCdf.density_1[i] = thphp.Density[j]
                BCdf.specific_heat_1[i] = thphp.Specific_Heat[j]
                BCdf.conductivity_1[i] = thphp.Conductivity[j]
                BCdf.LW_emissivity_1[i] = thphp.LW_Emissivity[j]
                BCdf.SW_transmittance_1[i] = thphp.SW_Transmittance[j]
                BCdf.SW_absorptivity_1[i] = thphp.SW_Absorptivity[j]
                BCdf.albedo_1[i] = thphp.Albedo[j]

    for i in range(0, len(BCdf['Material_2'])):
        for j in range(0, len(thphp['Material'])):
            if BCdf.Material_2[i] == thphp.Material[j]:
                BCdf.density_2[i] = thphp.Density[j]
                BCdf.specific_heat_2[i] = thphp.Specific_Heat[j]
                BCdf.conductivity_2[i] = thphp.Conductivity[j]
                BCdf.LW_emissivity_2[i] = thphp.LW_Emissivity[j]
                BCdf.SW_transmittance_2[i] = thphp.SW_Transmittance[j]
                BCdf.SW_absorptivity_2[i] = thphp.SW_Absorptivity[j]
                BCdf.albedo_2[i] = thphp.Albedo[j]

    for i in range(0, len(BCdf['Material_3'])):
        for j in range(0, len(thphp['Material'])):
            if BCdf.Material_3[i] == thphp.Material[j]:
                BCdf.density_3[i] = thphp.Density[j]
                BCdf.specific_heat_3[i] = thphp.Specific_Heat[j]
                BCdf.conductivity_3[i] = thphp.Conductivity[j]
                BCdf.LW_emissivity_3[i] = thphp.LW_Emissivity[j]
                BCdf.SW_transmittance_3[i] = thphp.SW_Transmittance[j]
                BCdf.SW_absorptivity_3[i] = thphp.SW_Absorptivity[j]
                BCdf.albedo_3[i] = thphp.Albedo[j]

    return BCdf