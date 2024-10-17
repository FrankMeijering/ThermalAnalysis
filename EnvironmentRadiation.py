# -*- coding: utf-8 -*-
"""
Created on Mon Apr 8 15:27:42 2024
Author: Frank Meijering (Delft University of Technology)

EnvironmentRadiation.py contains functions to ultimately compute all external heat fluxes (solar, albedo, and Earth IR)
onto an orbiting plate or multiple plates. The heat_received function is the one that computes the final outputs. This
file or any of the functions in this file do not cross the general user's path, since these functions are embedded into
the ThermalBudget.py classes.
"""


import numpy as np
from matplotlib import pyplot as plt
from Constants import const_lst
from scipy.spatial.transform import Rotation


def earth_orbit(day):
    """
    Compute Earth-Sun distance, based on the day from January  1.
    Perigee is on January 3.

    :param day: Number of days from January 1 (January 1 itself is day 1).
    :return: Sun-Earth distance in metres at the given day.
    """
    return const_lst['a']*(1-const_lst['e']**2)/(1+const_lst['e']*np.cos(((day-const_lst['perigee_day'])/const_lst['T'])*2*np.pi))


def solar_flux(day):
    """
    Compute solar flux on Earth based on the Earth-Sun distance on the given day. Solar minima/maxima are currently
    not included.

    :param day: Number of days from January 1 (January 1 itself is day 1).
    :return: Solar flux in W/m^2 at the given day.
    """
    distance = earth_orbit(day)

    return const_lst['solar_const']*(const_lst['AU']/distance)**2


def albedo_flux(day, beta, theta):
    """
    Compute albedo flux at the Earth's surface, in the direction of a given orbital position.
    For the actual power in Watt, this value must be multiplied by (Re/(Re+h))^2, the area, spacecraft absorptivity,
    and the view factor.

    :param day: Number of days from January 1 (January 1 itself is day 1).
    :param beta: Angle [rad] between the Sun vector and orbital plane.
    :param theta: True anomaly [rad] of the spacecraft's orbit around the Earth, where theta = 0 at the orthogonal
    projection of the solar vector.
    :return: Albedo flux in W/m^2 at the Earth's surface in the direction of the given orbital position.
    """
    return np.maximum(const_lst['albedo'] * solar_flux(day) * np.cos(beta) * np.cos(theta), 0.)


def planet_flux():
    """
    Average Earth infrared flux at the Earth's surface.
    For the actual power in Watt, this value must be multiplied by the Earth's surface area and emissivity (assume 1),
    spacecraft absorptivity, and the view factor.

    :return: Earth total infrared flux in W/m^2 at the Earth's surface.
    """
    return const_lst['IR_e']


def view_factor(h, eta):
    """
    Returns view factor from a flat surface to a sphere (ECSS-E-HB-31-01 Part 1A 4.2.2 Planar to spherical).
    Valid for 0 < eta < 180 deg. Note that lambda = 180 - eta.

    :param h: Altitude [m] of the orbit above Earth.
    :param eta: Angle [rad] between the outward normal of the surface and the zenith direction with respect to Earth.
    :return: View factor between the Earth and a flat plate; is zero when the plate faces away from Earth.
    """
    aL = np.arcsin(1/(1+h/const_lst['Re']))

    B0 = (2/7/np.pi)*(577/105-7*np.cos(aL)+4/3*np.cos(aL)**3-2/5*np.cos(aL)**5+4/7*np.cos(aL)**7)
    B1 = 1/2*np.sin(aL)**2
    B2 = 8/(7*np.pi)*(np.cos(aL)-2*np.cos(aL)**3+4*np.cos(aL)**5-3*np.cos(aL)**7)
    B3 = 4/7/np.pi*(-np.cos(aL)+40/3*np.cos(aL)**3-91/3*np.cos(aL)**5+18*np.cos(aL)**7)
    B4 = 8/35/np.pi*(5*np.cos(aL)-35*np.cos(aL)**3+63*np.cos(aL)**5-33*np.cos(aL)**7)

    lam = np.pi-eta
    F = B0+B1*np.cos(lam)+B2*np.cos(lam)**2+B3*np.cos(lam)**4+B4*np.cos(lam)**6  # View factor

    return np.maximum(F, 0.)


def eclipse_time(beta, h):
    """
    Calculates eclipse parameters based on a simplified cylindrical shadow behind Earth.
    Penumbra effect is neglected.

    :param beta: Beta angle [rad] of Sun with respect to the spacecraft's orbital plane.
    :param h: Orbital altitude [m].
    :return: T_ecl, ratio_ecl, theta_ecl: time of eclipse [s], orbital ratio of eclipse,
    true anomaly of eclipse onset [rad].
    """

    Re = const_lst['Re']  # [m] Earth radius
    mu_e = const_lst['mu_e']  # [m^3/s^2] Gravitational parameter of Earth

    # Orbital period of the satellite
    T = 2*np.pi*np.sqrt((Re+h)**3/mu_e)  # [s]

    # Maximum eclipse angle, equivalent to maximum beta angle for which there is still an eclipse
    beta_max = np.arcsin(Re/(Re+h))  # [rad]

    if np.abs(beta) < beta_max:
        psi_ecl_half = np.arcsin(np.sqrt(((Re/(Re+h))**2-np.sin(beta)**2)/((np.cos(beta))**2)))  # [rad] eclipse half-angle
        theta_ecl = np.pi-psi_ecl_half  # [rad] true anomaly onset of eclipse
        ratio_ecl = psi_ecl_half/np.pi  # [-]
        T_ecl = ratio_ecl*T  # [s]
    else:
        theta_ecl = np.nan  # [rad] de-activates eclipse
        ratio_ecl = 0.  # [-]
        T_ecl = 0.  # [s]

    return T_ecl, ratio_ecl, theta_ecl


def heat_received(day, beta, theta, h, tau, phi):
    """
    Compute the solar, albedo, and Earth IR flux received by an oriented flat face of a satellite.
    This does not yet include the surface area nor absorptivity.

    :param day: Number of days from January 1 (January 1 itself is day 1).
    :param beta: Beta angle [rad] of Sun with respect to the spacecraft's orbital plane.
    :param theta: Array of true anomalies [rad] of the spacecraft's orbit around the Earth, where theta = 0 at the
    orthogonal projection of the solar vector.
    :param h: Orbital altitude [m].
    :param tau: Polar angle [rad] in spherical coordinates (z-axis parallel to velocity direction).
    :param phi: Azimuth angle [rad] in spherical coordinates.
    :return: q_pla, q_alb, q_s: Earth IR heat flux [W/m^2], albedo heat flux [W/m^2], solar heat flux [W/m^2].
    """
    eta = np.arccos(np.cos(phi)*np.sin(tau))  # [rad] angle between surface normal and zenith direction from Earth
    q_pla = np.ones(np.shape(theta)) * planet_flux() * view_factor(h, eta)  # [W/m^2] Earth IR radiation
    q_alb = albedo_flux(day, beta, theta) * view_factor(h, eta)  # [W/m^2] albedo radiation

    # Solar radiation
    T_ecl, ratio_ecl, theta_ecl = eclipse_time(beta, h)
    cosgamma = np.cos(beta)*np.cos(theta)*np.sin(tau)*np.cos(phi)-np.sin(beta)*np.sin(tau)*np.sin(phi)-\
               np.cos(beta)*np.sin(theta)*np.cos(tau)  # cosine of the angle between the solar vector and surface normal
    q_s = solar_flux(day) * np.maximum(cosgamma, 0.)
    idx_ecl = np.argwhere(np.logical_and(theta_ecl <= theta%(2*np.pi), theta%(2*np.pi) <= 2*np.pi-theta_ecl))  # satellite in shadow
    q_s[idx_ecl] = 0.

    return q_pla, q_alb, q_s


def t_to_theta(h, t):
    """
    Converts orbital time [s] to true anomaly [rad].

    :param h: Orbital altitude [m].
    :param t: Time [s] after passing the Sun-Earth vertical plane.
    :return: True anomaly [rad].
    """
    T = 2*np.pi*np.sqrt((const_lst['Re']+h)**3/const_lst['mu_e'])  # [s] Orbital period of the satellite
    convert = 2*np.pi/T  # Conversion from time [s] to true anomaly [rad]
    return convert*t


def theta_to_t(h, theta):
    """
    Converts true anomaly [rad] to orbital time [s].

    :param h: Orbital altitude [m].
    :param theta: True anomaly [rad].
    :return: Time [s] after passing the Sun-Earth vertical plane.
    """
    T = 2*np.pi*np.sqrt((const_lst['Re']+h)**3/const_lst['mu_e'])  # [s] Orbital period of the satellite
    convert = T/(2*np.pi)  # Conversion from true anomaly [rad] to time [s]
    return convert*theta


def beta_angle(day, RAAN, i):
    """
    Computes the solar declination angle beta for the given day, for a given orbit.

    :param day: Number of days from January 1 (January 1 itself is day 1).
    :param RAAN: Right ascension of the ascending node [rad] of the satellite's orbit.
    :param i: Inclination [rad] of the satellite's orbit with respect to the Earth's equator.
    :return: Solar declination (beta) angle [rad].
    """
    Gamma = ((day-const_lst['march_equinox'])*2*np.pi/const_lst['T'])%(2*np.pi)
    return np.arcsin(np.cos(Gamma)*np.sin(RAAN)*np.sin(i)-np.sin(Gamma)*np.cos(const_lst['obliquity'])*np.cos(RAAN)*
                     np.sin(i)+np.sin(Gamma)*np.sin(const_lst['obliquity'])*np.cos(i))


def tau_phi(direction):
    """
    Converts a direction ('x+', 'y-', etc) into spherical coordinates.

    :param direction: String indicating the direction of a face (one of 'x+', 'y+', 'z+', 'x-', 'y-', 'z-').
    :return: Two arguments tau (polar angle [rad]) and phi (azimuth angle [rad]) in spherical coordinates
    (z-axis parallel to velocity direction).
    """
    direction = direction.lower()
    if direction == 'x+':
        tau = np.pi/2.
        phi = 0.
    elif direction == 'y+':
        tau = np.pi/2.
        phi = np.pi/2.
    elif direction == 'z+':
        tau = 0.
        phi = 0.
    elif direction == 'x-':
        tau = np.pi/2.
        phi = np.pi
    elif direction == 'y-':
        tau = np.pi/2.
        phi = 3*np.pi/2.
    elif direction == 'z-':
        tau = np.pi
        phi = 0.
    else:
        tau = 0.
        phi = 0.
        print(f"----------------ERROR----------------\n"
              f"Incorrect argument specification ({direction}) for the tau_phi function.\n"
              f"Default direction ('z+') was selected.\n"
              f"Specify the argument as either one of: ['x+', 'x-', 'y+', 'y-', 'z+', 'z-'].\n"
              f"-------------------------------------\n")
    return tau, phi


def spherical_to_cartesian(tauphi):
    """
    Takes spherical coordinates (tau angle from the z-axis and phi angle from the x-axis) and converts to a cartesian
    unit vector.

    :param tauphi: Tuple of tau (polar angle [rad]) and phi (azimuth angle [rad]) in spherical coordinates,
                   (z-axis parallel to velocity direction)
    :return: Cartesian unit vector.
    """
    tau = tauphi[0]
    phi = tauphi[1]
    surface = np.array([0., 0., 1.])
    # Following rotation is extrinsic (small 'xyz'). Capital 'XYZ' would have meant intrinsic.
    euler_to_sphere = Rotation.from_euler('xyz', [0, tau, phi])
    return euler_to_sphere.apply(surface)


def cartesian_to_spherical(cartesian):
    """
    Takes a cartesian unit vector and converts to spherical coordinates (tau angle from the z-axis and phi angle from
    the x-axis).

    :param cartesian: Cartesian coordinates of a point.
    :return: Tuple of tau (polar angle [rad]) and phi (azimuth angle [rad]) in spherical coordinates,
             (z-axis parallel to velocity direction)
    """
    x = cartesian[0]
    y = cartesian[1]
    z = cartesian[2]
    tau = np.arccos(max(-1, min(1, z)))  # use max and min in case the float value slightly exceeds magnitude 1
    phi = 0.  # remains zero when phi=0 and tau=90 deg (final else statement not needed).
    if (abs(tau) > 1e-10 or abs(tau-np.pi) > 1e-10) and abs(y) > 1e-10:  # if tau = 0 or 180, phi is set to zero.
        phi = np.arctan2(y, x)
        if phi < 0:
            phi = 2*np.pi+phi  # 2pi minus the magnitude of the angle; -- is +
    elif abs(y) < 1e-10 < abs(x) and x < 0:  # if y is indeed zero and x is negative (and nonzero)
        phi = np.pi
    return tau, phi

