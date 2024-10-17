# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 14:53:09 2024
Author: Frank Meijering (Delft University of Technology)

Materials.py reads the Materials.csv database and converts it to Python objects and dictionaries. This file does not
cross the general user's path, since this file is embedded into the ThermalBudget.py classes.
"""


import os
import pandas as pd
import numpy as np


def get_file(file):
    return os.path.join(os.path.dirname(__file__), file)


def get_folder_file(folder, file):
    return get_file(os.path.join(folder, file))


class Material:
    """
    Material class with all its properties that are relevant for thermal analysis.
    """
    def __init__(self, name, density, c_cap, k_through):
        """
        Initialise a new Material object. Ensure that the correct units are used!

        :param name: Name of the material.
        :param density: Density [kg/m^3]
        :param c_cap: Specific heat capacity [J/(kg.K)]
        :param k_through: Thermal conductivity [W/(m.K)].
        """
        self.name = name
        self.density = density
        self.c_cap = c_cap
        self.k_through = k_through


class Coating:
    """
    Coating to be applied on another material, assigning the optical properties.
    """
    def __init__(self, name, alpha, epsilon):
        """
        Initialise a new Coating object.

        :param name: Name of the coating.
        :param alpha: Absorptivity in the solar spectrum.
        :param epsilon: Emissivity in the IR spectrum.
        """
        self.name = name
        self.alpha = alpha
        self.epsilon = epsilon


class Contact:
    """
    Approximation of contact conductance for two materials pressed together. It highly depends on the applied pressure.
    """
    def __init__(self, name, h_contact):
        """
        Initialise a new Contact object. Ensure that the correct units are used!

        :param name: Name of contact connection.
        :param h_contact: Heat transfer coefficient [W/(m^2.K)].
        """
        self.name = name
        self.h_contact = h_contact


with open(get_file('Materials.csv'), 'rb') as f:
    df = pd.read_csv(f, delimiter=',', header=None)
    df = df.dropna(how='all')  # drop empty rows

    idx1 = np.argwhere(df[0] == 'MATERIALS-BEGIN')[0, 0]
    idx2 = np.argwhere(df[0] == 'OPTICAL-BEGIN')[0, 0]
    idx3 = np.argwhere(df[0] == 'CONTACT-BEGIN')[0, 0]
    idx1_end = np.argwhere(df[0] == 'MATERIALS-END')[0, 0]
    idx2_end = np.argwhere(df[0] == 'OPTICAL-END')[0, 0]
    idx3_end = np.argwhere(df[0] == 'CONTACT-END')[0, 0]

    mat_names = list(df[0][idx1+1:idx1_end])
    coat_names = list(df[0][idx2+1:idx2_end])
    con_names = list(df[0][idx3+1:idx3_end])

    densities = list(pd.to_numeric(df[1][idx1+1:idx1_end]))
    specific_heat_capacities = list(pd.to_numeric(df[3][idx1+1:idx1_end]))
    thermal_conductivities = list(pd.to_numeric(df[9][idx1+1:idx1_end]))

    absorptivities = list(pd.to_numeric(df[5][idx2+1:idx2_end]))
    emissivities = list(pd.to_numeric(df[7][idx2+1:idx2_end]))

    heat_transfer_coefficients = list(pd.to_numeric(df[11][idx3+1:idx3_end]))

mat_lst = {}
coat_lst = {}
con_lst = {}
for i, name in enumerate(mat_names):
    mat_lst[name] = Material(name, densities[i], specific_heat_capacities[i], thermal_conductivities[i])
for i, name in enumerate(coat_names):
    coat_lst[name] = Coating(name, absorptivities[i], emissivities[i])
for i, name in enumerate(con_names):
    con_lst[name] = Contact(name, heat_transfer_coefficients[i])
