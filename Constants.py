# -*- coding: utf-8 -*-
"""
Created on Mon Apr 8 15:28:19 2024
Author: Frank Meijering (Delft University of Technology)

Constants.py is a collection of physical constants used for all the computations. Some values are not actual constants,
such as the Solar 'constant'. Please refer to the websites mentioned below. This file does not cross the general user's
path, since the constants are embedded into the computations in EnvironmentRadiation.py and ThermalBudget.py.
"""


import numpy as np

# (1) https://nssdc.gsfc.nasa.gov/planetary/factsheet/earthfact.html
# (2) https://www.iau.org/static/resolutions/IAU2012_English.pdf
# (3) http://astropixels.com/ephemeris/perap2001.html
# (4) https://physics.nist.gov/cgi-bin/cuu/Value?bg
# (5) https://physics.nist.gov/cgi-bin/cuu/Value?sigma
# (6) http://www.astropixels.com/ephemeris/soleq2001.html
# (7) https://lambda.gsfc.nasa.gov/product/cobe/
# (8) https://ttu-ir.tdl.org/items/abf38b2c-8142-432f-bc86-3ed991766524

AU = 1.495978707e11  # [m] Astronomical unit (2)
mu_e = 3.986e14  # [m^3/s^2] Gravitational parameter of Earth (1)
Re = 6371e3   # [m] Earth volumetric mean radius (1)
solar_const = 1361.  # [W/m^2] Solar constant (= solar flux at 1 AU) (1)
e = 0.01671  # [-] Eccentricity of Earth orbit (1)
a = 1.*AU  # [m] Semi-major axis of Earth orbit (1)
T = 365.25  # [days] Earth orbital period (1)
perigee_day = 3  # [days] Day of perigee of Earth orbit is approximately 3rd of January (3)
albedo = 0.294  # [-] Average Earth bond albedo (1)
stefan = 5.670374419e-8  # [W/(m^2*K^4)] Stefan-Boltzmann constant (5)
obliquity = 23.44*np.pi/180.  # [rad] Obliquity of the ecliptic (1)
march_equinox = 79.25  # [days] Average day in the year of the March equinox (20 March) (6)
kelvin = 273.15  # Add this to degrees Celsius to obtain Kelvin
T_space = 3.  # [K] Temperature of space due to cosmic background radiation (7)
IR_e = 239.  # [W/m^2] Average IR emission by the Earth's surface (8)

const_lst = {'AU': AU, 'mu_e': mu_e, 'Re': Re, 'solar_const': solar_const, 'e': e, 'a': a, 'T': T,
             'perigee_day': perigee_day, 'albedo': albedo, 'stefan': stefan, 'obliquity': obliquity,
             'march_equinox': march_equinox, 'kelvin': kelvin, 'T_space': T_space, 'IR_e': IR_e}
