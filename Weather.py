"""
Created 28 October 2021

Author: C.Gerike-Roberts

Extract data from .epw file to give inputs for solver.
"""
import dm4bem

# Simulation with weather data
# ----------------------------
filename = 'GBR_ENG_RAF.Lyneham.037400_TMYx.2004-2018.epw'
start_date = '2018-01-26 09:00:00'
end_date = '2018-01-27 09:00:00'


def weather_input():
    # Read weather data from climateonebuilding.org .epw file
    [data, meta] = dm4bem.read_epw(filename, coerce_year=None)
    month_year = data['month'].astype(str) + '-' + data['year'].astype(str)
    print(f"Months - years in the dataset: {sorted(set(month_year))}")
    weather = data[["temp_air", "dir_n_rad", "dif_h_rad"]]
    del data
    weather.index = weather.index.map(lambda t: t.replace(year=2018))
    weather = weather[(weather.index >= start_date) & (
        weather.index < end_date)]
    return weather, meta
