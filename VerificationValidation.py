# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 12:25:13 2024
Author: Frank Meijering (Delft University of Technology)

VerificationValidation.py computes numerous cases and compares them to ESATAN data and FUNcube flight data. Mainly used
for the thesis report; not intended to be used by the general user.
"""


import numpy as np
from matplotlib import pyplot as plt
import os
from EnvironmentRadiation import heat_received, view_factor, eclipse_time, beta_angle
from ThermalBudget import NodalModel, Node, theta_to_t, OrbitalModel, view_factor_par, view_factor_perp
from Constants import const_lst
import pandas as pd
from FUNcube import FUNcubeModel, FUNcubeOrbit, P_tot
from CommonNodalModels import assign_q_ext_to_pcb


def get_file(file):
    """
    Returns the path of a file in the same directory as this Python file; works on most operating systems.

    :param file: Name (str) of the file to be retrieved.
    :return: Path of the file.
    """
    return os.path.join(os.path.dirname(__file__), file)


def get_folder_file(folder, file):
    """
    Returns the path of a file in a folder within the same directory as this Python file;
    works on most operating systems.

    :param folder: Name (str) of the folder which the file is in.
    :param file: Name (str) of the file to be retrieved.
    :return: Path of the file.
    """
    return get_file(os.path.join(folder, file))


def calc_err(y_ref, y_python, t_ref, t_python):
    """
    Calculates the error between two differently shaped arrays. Interpolates where needed.

    :param y_ref: Y-values of the reference results array.
    :param y_python: Y-values of the Python results array.
    :param t_ref: X-values of the reference results array.
    :param t_python: X-values of the Python results array.
    :return: Error between y_ref and y_python.
    """
    y_python_interp = np.interp(t_ref, t_python, y_python)
    return y_python_interp-y_ref, y_python_interp


def calc_simple_rmse(y_diff, outlier_sens=10):
    """
    Calculates the Root Mean Square Error (RMSE) of an array that contains the difference (error) between two arrays.
    Also returns the index in the array of any outliers present

    :param y_diff: Array with the difference (error) between two arrays.
    :param outlier_sens: Sensitivity of detecting outliers. The larger this variable, the larger the outlier must be
                         before it gets detected.
    :return: RMSE, outliers: Single value representing the RMSE of the array, index of outliers.
    """
    idx = np.argwhere(np.abs(y_diff) > outlier_sens*np.mean(np.abs(y_diff)))
    y_diff_no_outliers = np.array(y_diff)
    y_diff_no_outliers = np.delete(y_diff_no_outliers, idx)
    return np.sqrt(np.sum(y_diff_no_outliers**2)/y_diff_no_outliers.shape[0]), idx


def verify_viewfactors():
    # Verify view factor computations (parallel)
    c = 1
    X = np.array([0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 2.0, 4.0, 5.0, 10., 9999.])
    Y = np.arange(0.1, 10.01, 0.01)
    a = X * c
    b = Y * c

    i = 0
    plt.figure()
    for a in a:
        vf = view_factor_par(a, b, c)
        plt.semilogx(Y, vf, label=f'X={X[i]}')
        i += 1
    plt.title('View Factor for Parallel Plates')
    plt.ylabel(r'F$_{12}$')
    plt.xlabel('Y')
    plt.grid(which='both')
    plt.legend()

    # Verify view factor computations (perpendicular)
    b = 1
    L = np.array([0.1, 0.2, 0.4, 0.6, 1.0, 2.0, 4.0, 6.0, 10., 20.])
    N = np.arange(0.1, 10.01, 0.01)
    a = N * b
    c = L * b

    i = 0
    plt.figure()
    for c in c:
        vf = view_factor_perp(a, b, c)
        plt.semilogx(N, vf, label=f'L={L[i]}')
        i += 1
    plt.title('View Factor for Perpendicular Plates')
    plt.ylabel(r'F$_{12}$')
    plt.xlabel('N')
    plt.grid(which='both')
    plt.legend()

    # View factor for flat plate and sphere (used in environmental analysis)
    H = np.array([0.001, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 1., 2., 5.])
    lamb = np.arange(0., np.pi+0.01, 0.01)

    i = 0
    plt.figure()
    for H in H:
        vf = view_factor(H*const_lst['Re'], np.pi-lamb)
        plt.plot(lamb*180/np.pi, vf, label=f'H={H}')
        i += 1
    plt.title('View Factor from Plate to Sphere')
    plt.ylabel(r'F$_{12}$')
    plt.xlabel(r'$\lambda$ [deg]')
    plt.xticks([0, 20, 60, 100, 140, 180])
    plt.xticks([0, 20, 40, 60, 80, 100, 120, 140, 160, 180], minor=True)
    plt.grid(which='both')
    plt.legend()


def verify_eclipse():
    beta = np.arange(0, np.pi/2, 0.005)
    time_ecl = np.zeros(beta.shape[0])
    ratio_ecl = np.zeros(beta.shape[0])
    th_ecl = np.zeros(beta.shape[0])
    alt = 408e3
    for i in range(beta.shape[0]):
        time_ecl[i], ratio_ecl[i], th_ecl[i] = eclipse_time(beta[i], alt)
    plt.figure()
    plt.title(f'Eclipse Fraction at {alt/1e3:.0f} km Altitude')
    plt.plot(beta*180/np.pi, 1-ratio_ecl, label='Fraction Spent in Sunlight')
    plt.plot(beta*180/np.pi, ratio_ecl, label='Fraction Spent in Eclipse')

    plt.xlabel('Beta angle [deg]')
    plt.ylabel('Fraction of Orbit')
    plt.xticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90])
    plt.yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
    plt.grid()
    plt.legend()


def verify_beta_from_parameters():
    # first plot
    day = 79.25  # RAAN = 0 = facing the Sun
    raan = np.array([-90., -60., -30., 0., 30., 60., 90.])*np.pi/180.
    incl = np.arange(0., np.pi+0.01, 0.01)
    plt.figure()
    for raan_val in raan:
        beta = beta_angle(day, raan_val, incl)
        plt.plot(incl*180/np.pi, beta*180/np.pi, label=f'RAAN={raan_val*180/np.pi:.0f}'+r'$\degree$')
    plt.title(f'Beta Angle for Varying Orbital Parameters\nDay = {day:.0f} (March)')
    plt.xlabel('Inclination [deg]')
    plt.ylabel('Beta Angle [deg]')
    plt.legend(loc='lower right')
    plt.grid()
    plt.tight_layout()

    # second plot
    colours = ['tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
    day = 79.25+365.25*3/4  # RAAN = 0 = facing the Sun
    # raan = np.array([-180., -150., -120., -90., -60., -30., 0., 30., 60., 90., 120., 150., 180.])*np.pi/180.
    raan = np.array([0., 30., 60., 90., 120., 150., 180.])*np.pi/180.
    incl = np.arange(0., np.pi+0.01, 0.01)
    plt.figure()
    for x, raan_val in enumerate(raan):
        beta = beta_angle(day, raan_val, incl)
        plt.plot(incl*180/np.pi, beta*180/np.pi, label=f'RAAN={raan_val*180/np.pi:.0f}'+r'$\degree$', c=colours[x])
    plt.title(f'Beta Angle for Varying Orbital Parameters\nDay = {day:.0f} (December)')
    plt.xlabel('Inclination [deg]')
    plt.ylabel('Beta Angle [deg]')
    plt.legend(loc='lower right')
    plt.grid()
    plt.tight_layout()


def verify_EnvironmentRadiation():
    # Verification of EnvironmentRadiation with ESATAN
    cases = np.array([[0., 0., 0., 300e3],
                      [0., 0., 0., 408e3],
                      [0., 0., 0., 1000e3],
                      [0., 90.*np.pi/180., 180.*np.pi/180., 408e3],
                      [45.*np.pi/180., 0., 0., 408e3],
                      [80.*np.pi/180., 0., 0., 408e3]])  # [[beta, tau, phi, h], [case 2], [case 3], ...]
    esatan_names = ['BetaZero_Z_300km.csv', 'BetaZero_Z_408km.csv', 'BetaZero_Z_1000km.csv', 'BetaZero_Xm_408km.csv',
                    'Beta45_Z_408km.csv', 'Beta80_Z_408km.csv']

    for n in range(cases.shape[0]):
        figA = plt.figure()
        axA = figA.add_subplot()
        beta = cases[n][0]
        tau = cases[n][1]
        phi = cases[n][2]
        h = cases[n][3]
        axA.set_title('Heat Flux on an Oriented Plate\n'+r'$\beta$'+f': {beta*180/np.pi:.0f} deg, '+r'$\tau$'+
                  f': {tau*180/np.pi:.0f} deg, '+r'$\phi$'+f': {phi*180/np.pi:.0f} deg, '+r'$h$'+f': {h:.0f} m')
        theta_lst = np.arange(0., np.pi*2, np.pi/1000)
        pla_lst, alb_lst, s_lst = heat_received(1., beta, theta_lst, h, tau, phi)

        # ESATAN data [[time, albedo, Earth IR, solar], [...]]
        esatan = np.array(pd.read_csv(get_folder_file('ESATAN', esatan_names[n]))).T[[0, 1, 3, 5]].T[5:].astype(float)

        Re = const_lst['Re']  # [m] Earth radius
        mu_e = const_lst['mu_e']  # [m^3/s^2] Gravitational parameter of Earth
        T = 2*np.pi*np.sqrt((Re+h)**3/mu_e)  # [s] Orbital period of the satellite
        convert = 2*np.pi/T  # Conversion from time [s] to true anomaly [rad]
        esatan_xval = esatan[:, 0]*convert

        # compute RMSE
        q_pla_err, q_pla_new = calc_err(esatan[:, 2], pla_lst, esatan_xval, theta_lst)
        q_alb_err, q_alb_new = calc_err(esatan[:, 1], alb_lst, esatan_xval, theta_lst)
        q_s_err, q_s_new = calc_err(esatan[:, 3], s_lst, esatan_xval, theta_lst)
        q_pla_rmse, q_pla_outliers = calc_simple_rmse(q_pla_err)
        q_alb_rmse, q_alb_outliers = calc_simple_rmse(q_alb_err)
        q_s_rmse, q_s_outliers = calc_simple_rmse(q_s_err)

        figERR = plt.figure()
        axERR = figERR.add_subplot()
        axERR.set_title(f'Error in Heat Fluxes\n'+r'$\beta$'+f': {beta*180/np.pi:.0f} deg, '+r'$\tau$'+
                  f': {tau*180/np.pi:.0f} deg, '+r'$\phi$'+f': {phi*180/np.pi:.0f} deg, '+r'$h$'+f': {h:.0f} m')
        axERR.plot(esatan_xval*180./np.pi, q_pla_err, label=f'Earth IR, RMSE={q_pla_rmse:.2f}'+r' W/m$^2\approx$'+f'{q_pla_rmse/np.mean(pla_lst)*100:.1f}%', c='tab:blue')
        axERR.scatter(esatan_xval[q_pla_outliers]*180./np.pi, q_pla_err[q_pla_outliers], c='r', marker='X', zorder=100, s=50)

        axERR.plot(esatan_xval*180./np.pi, q_alb_err, label=f'Albedo, RMSE={q_alb_rmse:.2f}'+r' W/m$^2\approx$'+f'{q_alb_rmse/np.mean(alb_lst)*100:.1f}%', linestyle='dashed', c='tab:orange')
        axERR.scatter(esatan_xval[q_alb_outliers]*180./np.pi, q_alb_err[q_alb_outliers], c='r', marker='X', zorder=100, s=50)

        axERR.plot(esatan_xval*180./np.pi, q_s_err, label=f'Solar, RMSE={q_s_rmse:.2f}'+r' W/m$^2\approx$'+f'{q_s_rmse/np.mean(s_lst)*100:.1f}%', linestyle='dashdot', c='tab:green')
        axERR.scatter(esatan_xval[q_s_outliers]*180./np.pi, q_s_err[q_s_outliers], c='r', marker='X', zorder=100, s=50, label='Outliers')

        axERR.set_xlabel('True anomaly [deg]')
        axERR.set_ylabel(r"Error in Heat flux [W/m$^2$]")
        axERR.grid()
        axERR.legend(loc='lower left')
        figERR.tight_layout()

        axA.plot(theta_lst*180./np.pi, pla_lst, label='Earth IR, sim.', linewidth=2, zorder=10)
        axA.plot(theta_lst*180./np.pi, alb_lst, label='Albedo, sim.', linewidth=2, zorder=11)
        axA.plot(theta_lst*180./np.pi, s_lst, label='Direct solar, sim.', linewidth=2, zorder=12)

        # scatter plot instead of line plot because the lines overlap quite closely, making it difficult to visualise.
        axA.scatter(esatan[::2, 0]*convert*180./np.pi, esatan[::2, 2], label='Earth IR, ESATAN', marker='*', color='k')
        axA.scatter(esatan[::2, 0]*convert*180./np.pi, esatan[::2, 1], label='Albedo, ESATAN', marker='+', color='k')
        axA.scatter(esatan[::2, 0]*convert*180./np.pi, esatan[::2, 3], label='Solar, ESATAN', marker='x', color='k')

        axA.legend(loc='upper left')
        axA.set_xlabel('True anomaly [deg]')
        axA.set_ylabel(r"Heat flux [W/m$^2$]")
        axA.grid()
        figA.tight_layout()


def verify_TransientAnalysis():
    dt = 0.01  # [s]
    t_end = 10.  # [s]
    t = np.arange(0., t_end+dt, dt)

    # Simulation
    TestModel1 = NodalModel(t, celsius=True)
    TestModel1.add_node(Node(name='0', C_cap=1, T0=20, P_int=5*np.ones(t.shape[0])))
    TestModel1.add_node(Node(name='1', C_cap=2, T0=30))
    TestModel1.add_node(Node(name='2', C_cap=3, T0=40))
    TestModel1.add_node(Node(name='3', C_cap=4, T0=50))
    TestModel1.add_node(Node(name='4', C_cap=1000, T0=0))
    TestModel1.connect(node1=1, nodes2=[0, 2, 3], C_con=[10, 1, 5])
    TestModel1.connect(node1=4, nodes2=3, C_con=2)
    TestModel1.solve()

    # ESATAN data [[time, temperature1, temperature2, temperature3, temperature4, temperature5]
    esatan = np.array(pd.read_csv(get_folder_file('ESATAN', 'SimpleTransient.csv'))).T[[0, 1, 3, 5, 7, 9]].T[5:].astype(float)

    markers = ['*', 'x', '+', 'o', '^']
    plt.figure()
    for i in range(TestModel1.n):
        plt.plot(TestModel1.t, TestModel1.T[:, i], label=f'Node {i}, sim.', linewidth=2, zorder=10+i)
    for i in range(TestModel1.n):  # separate for-loop for legend ordering in the figure
        # ::30 makes esatan data more sparse to see more easily
        # scatter plot instead of line plot because the lines almost perfectly overlap, making it invisible.
        plt.scatter(esatan[::30, 0], esatan[::30, i+1], label=f'Node {i}, ESATAN', marker=markers[i], color='k')
    plt.title('Temperature Response of Five Conductive Nodes')
    plt.legend().set_zorder(999)
    plt.grid()
    plt.xlabel('Time [s]')
    plt.ylabel(r'Temperature [$\degree$C]')

    # compute RMSE
    err1, temp_new = calc_err(esatan[:, 1], TestModel1.T[:, 0], esatan[:, 0], t)
    err2, temp_new = calc_err(esatan[:, 2], TestModel1.T[:, 1], esatan[:, 0], t)
    err3, temp_new = calc_err(esatan[:, 3], TestModel1.T[:, 2], esatan[:, 0], t)
    err4, temp_new = calc_err(esatan[:, 4], TestModel1.T[:, 3], esatan[:, 0], t)
    err5, temp_new = calc_err(esatan[:, 5], TestModel1.T[:, 4], esatan[:, 0], t)
    rmse1, outliers1 = calc_simple_rmse(err1)
    rmse2, outliers2 = calc_simple_rmse(err2)
    rmse3, outliers3 = calc_simple_rmse(err3)
    rmse4, outliers4 = calc_simple_rmse(err4)
    rmse5, outliers5 = calc_simple_rmse(err5)

    figERR = plt.figure()
    axERR = figERR.add_subplot()
    axERR.set_title('Error in Temperature')
    axERR.plot(esatan[:, 0], err1, label=f'Node 0, RMSE={rmse1:.4f}' + r'$\degree$C', linestyle='solid')
    axERR.scatter(esatan[outliers1, 0], err1[outliers1], c='r', marker='X', zorder=100, s=50)
    axERR.plot(esatan[:, 0], err2, label=f'Node 1, RMSE={rmse2:.4f}' + r'$\degree$C', linestyle='dotted')
    axERR.scatter(esatan[outliers2, 0], err2[outliers2], c='r', marker='X', zorder=100, s=50)
    axERR.plot(esatan[:, 0], err3, label=f'Node 2, RMSE={rmse3:.4f}' + r'$\degree$C', linestyle='dashed')
    axERR.scatter(esatan[outliers3, 0], err3[outliers3], c='r', marker='X', zorder=100, s=50)
    axERR.plot(esatan[:, 0], err4, label=f'Node 3, RMSE={rmse4:.4f}' + r'$\degree$C', linestyle='dashdot')
    axERR.scatter(esatan[outliers4, 0], err4[outliers4], c='r', marker='X', zorder=100, s=50)
    axERR.plot(esatan[:, 0], err5, label=f'Node 4, RMSE={rmse5:.4f}' + r'$\degree$C', linestyle=(0, (3, 5, 1, 5, 1, 5)))
    axERR.scatter(esatan[outliers5, 0], err5[outliers5], c='r', marker='X', zorder=100, s=50, label='Outliers')
    axERR.set_xlabel('Time [s]')
    axERR.set_ylabel(r"Error in Temperature [$\degree$C]")
    axERR.grid()
    axERR.legend()
    figERR.tight_layout()


def verify_TwoPlatesNoIntRad():
    beta = 0. * np.pi / 180.
    tau1 = 0. * np.pi / 180.
    phi1 = 0. * np.pi / 180.
    tau2 = 90. * np.pi / 180.
    phi2 = 0. * np.pi / 180.

    h = 408e3
    orbits = 2
    dtheta = 0.01
    theta_lst = np.arange(0., orbits * 2 * np.pi + dtheta, dtheta)
    dt = theta_to_t(h, dtheta)
    t = theta_to_t(h, theta_lst)

    pla_lst1, alb_lst1, s_lst1 = heat_received(1., beta, theta_lst, h, tau1, phi1)
    pla_lst2, alb_lst2, s_lst2 = heat_received(1., beta, theta_lst, h, tau2, phi2)

    # still must multiply with optical properties
    alpha = 1.0
    epsilon = 1.0
    area = 1.0  # [m^2] (the angles etc. have already been included so all is multiplied here with the entire area)

    # Weird thing where ESATAN requires 2 mm thickness to achieve 1000 J/K of thermal capacitance.
    OrbitingPlate = NodalModel(t, celsius=True)
    OrbitingPlate.add_node(Node(C_cap=1000, T0=20, q_ext=(pla_lst1, alb_lst1, s_lst1), outer=True,
                           alpha=alpha, epsilon=epsilon, area=area, geom=(1., 1., 0.), origin=(0., 0., 0.5), name='z+'))  # z_plus
    OrbitingPlate.add_node(Node(C_cap=1000, T0=25, q_ext=(pla_lst2, alb_lst2, s_lst2), outer=True,
                           alpha=alpha, epsilon=epsilon, area=area, geom=(0., 1., 1.), origin=(0.5, 0., 0.), name='x+'))  # x_plus
    OrbitingPlate.connect(node1='z+', nodes2='x+', C_con=1)
    OrbitingPlate.solve()

    # OrbitingPlateRad = NodalModel(t, celsius=True)
    # OrbitingPlateRad.add_node(Node(C_cap=1000, T0=20, q_ext=(pla_lst1, alb_lst1, s_lst1), outer=True,
    #                        alpha=alpha, epsilon=epsilon, area=area, geom=(1., 1., 0.), origin=(0., 0., 0.5), name='z+'))  # z_plus
    # OrbitingPlateRad.add_node(Node(C_cap=1000, T0=25, q_ext=(pla_lst2, alb_lst2, s_lst2), outer=True,
    #                        alpha=alpha, epsilon=epsilon, area=area, geom=(0., 1., 1.), origin=(0.5, 0., 0.), name='x+'))  # x_plus
    # OrbitingPlateRad.connect(node1='z+', nodes2='x+', C_con=1, rad=True)
    # OrbitingPlateRad.solve()


    # ESATAN data [[time, temperature1, temperature2, temperature3, temperature4, temperature5]
    esatan = np.array(pd.read_csv(get_folder_file('ESATAN', 'BetaZero_Z_408km_platesZX_norad_temps.csv'))).T[[0, 1, 3]].T[5:].astype(float)

    markers = ['o', 'x', '+', '*', '^']
    plt.figure()
    for i in range(OrbitingPlate.n):
        plt.plot(OrbitingPlate.t, OrbitingPlate.T[:, i], label=f'Node {i}, sim.', linewidth=2, zorder=10+i)
        # plt.plot(OrbitingPlateRad.t, OrbitingPlateRad.T[:, i], label=f'Node {i}, sim.+rad.', linewidth=2, zorder=10+i, linestyle='dashed')
    for i in range(OrbitingPlate.n):  # separate for-loop for legend ordering in the figure
        # ::10 makes esatan data more sparse to see more easily
        # scatter plot instead of line plot because the lines almost perfectly overlap, making it invisible.
        plt.scatter(esatan[::10, 0], esatan[::10, i+1], label=f'Node {i}, ESATAN', marker=markers[i], color='k')
    plt.title('Temperature of Two Orbiting Perpendicular Plates\n No Radiation Between Nodes\n'+r'$\beta$'+
              f': {beta*180/np.pi:.0f} deg, '+r'$h$'+f': {h:.0f} m')
    plt.legend(loc='lower left').set_zorder(999)
    plt.grid()
    plt.xlabel('Time [s]')
    plt.ylabel(r'Temperature [$\degree$C]')
    plt.tight_layout()


def verify_BoxNoIntRad():
    BoxOrbit = OrbitalModel(h=408e3, surfaces=['x+', 'y+', 'z+', 'x-', 'y-', 'z-'], beta=0., n_orbits=2)
    BoxOrbit.compute()
    t = BoxOrbit.t

    # still must multiply with optical properties
    alpha = 1.0
    epsilon = 1.0
    area = 1.0  # [m^2] (the angles etc. have already been included so all is multiplied here with the entire area)

    OrbitingPlate = NodalModel(t)
    OrbitingPlate.add_node(Node(C_cap=1000, T0=20, q_ext=BoxOrbit.get_heat('x+'), outer=True,
                           alpha=alpha, epsilon=epsilon, area=area, name='x+'))
    OrbitingPlate.add_node(Node(C_cap=1000, T0=20, q_ext=BoxOrbit.get_heat('y+'), outer=True,
                           alpha=alpha, epsilon=epsilon, area=area, name='y+'))
    OrbitingPlate.add_node(Node(C_cap=1000, T0=20, q_ext=BoxOrbit.get_heat('z+'), outer=True,
                           alpha=alpha, epsilon=epsilon, area=area, name='z+'))
    OrbitingPlate.add_node(Node(C_cap=1000, T0=20, q_ext=BoxOrbit.get_heat('x-'), outer=True,
                           alpha=alpha, epsilon=epsilon, area=area, name='x-'))
    OrbitingPlate.add_node(Node(C_cap=1000, T0=20, q_ext=BoxOrbit.get_heat('y-'), outer=True,
                           alpha=alpha, epsilon=epsilon, area=area, name='y-'))
    OrbitingPlate.add_node(Node(C_cap=1000, T0=20, q_ext=BoxOrbit.get_heat('z-'), outer=True,
                           alpha=alpha, epsilon=epsilon, area=area, name='z-'))
    OrbitingPlate.connect('x+', ['y+', 'z+', 'y-', 'z-'], C_con=1)
    OrbitingPlate.connect('y+', ['z+', 'x-', 'z-'], C_con=[1, 1, 1])
    OrbitingPlate.connect('z+', ['x-', 'y-'], C_con=1)
    OrbitingPlate.connect('x-', ['y-', 'z-'], C_con=1)
    OrbitingPlate.connect('y-', 'z-', C_con=1)
    OrbitingPlate.solve()

    # ESATAN data [[time, temperature1, temperature2, temperature3, temperature4, temperature5, temperature6]]
    esatan = np.array(pd.read_csv(get_folder_file('ESATAN', 'BetaZero_Z_408km_box_norad_temps.csv'))).T[[0, 5, 7, 1, 9, 3, 11]].T[5:].astype(float)

    markers = ['o', 'x', '+', '*', '^', 'v']
    plt.figure()
    for i in range(OrbitingPlate.n):
        if i == 0 or i == 3:
            plt.plot(OrbitingPlate.t, OrbitingPlate.T[:, i], label=f'Node {i}, {OrbitingPlate.name[i]}, sim.', linewidth=2, zorder=10+i)
    for i in range(OrbitingPlate.n):  # separate for-loop for legend ordering in the figure
        # ::10 makes esatan data more sparse to see more easily
        # scatter plot instead of line plot because the lines almost perfectly overlap, making it invisible.
        if i == 0 or i == 3:
            plt.scatter(esatan[::10, 0], esatan[::10, i+1], label=f'Node {i}, ESATAN', marker=markers[i], color='k')
    plt.title('Temperatures of Orbiting Box\nWithout Internal Radiation')
    plt.legend(loc='lower left').set_zorder(999)
    plt.grid()
    plt.xlabel('Time [s]')
    plt.ylabel(r'Temperature [$\degree$C]')
    plt.tight_layout()

    # compute RMSE
    xp_err, temp_new = calc_err(esatan[:, 1], OrbitingPlate.T[:, 0], esatan[:, 0], OrbitingPlate.t)
    yp_err, temp_new = calc_err(esatan[:, 2], OrbitingPlate.T[:, 1], esatan[:, 0], OrbitingPlate.t)
    zp_err, temp_new = calc_err(esatan[:, 3], OrbitingPlate.T[:, 2], esatan[:, 0], OrbitingPlate.t)
    xm_err, temp_new = calc_err(esatan[:, 4], OrbitingPlate.T[:, 3], esatan[:, 0], OrbitingPlate.t)
    ym_err, temp_new = calc_err(esatan[:, 5], OrbitingPlate.T[:, 4], esatan[:, 0], OrbitingPlate.t)
    zm_err, temp_new = calc_err(esatan[:, 6], OrbitingPlate.T[:, 5], esatan[:, 0], OrbitingPlate.t)
    xp_rmse, xp_outliers = calc_simple_rmse(xp_err, outlier_sens=10)
    yp_rmse, yp_outliers = calc_simple_rmse(yp_err, outlier_sens=10)
    zp_rmse, zp_outliers = calc_simple_rmse(zp_err, outlier_sens=10)
    xm_rmse, xm_outliers = calc_simple_rmse(xm_err, outlier_sens=10)
    ym_rmse, ym_outliers = calc_simple_rmse(ym_err, outlier_sens=10)
    zm_rmse, zm_outliers = calc_simple_rmse(zm_err, outlier_sens=10)

    figERR = plt.figure()
    axERR = figERR.add_subplot()
    axERR.set_title(f'Error in Temperatures\nWithout Internal Radiation')
    axERR.plot(esatan[:, 0], xp_err, label=f'Node 0, x+, RMSE={xp_rmse:.2f}' + r' $\degree$C')
    axERR.scatter(esatan[:, 0][xp_outliers], xp_err[xp_outliers], c='r', marker='X', zorder=100, s=50)
    # axERR.plot(esatan[:, 0], yp_err, label=f'Node 1, y+, RMSE={yp_rmse:.2f}' + r' $\degree$C')
    # axERR.scatter(esatan[:, 0][yp_outliers], yp_err[yp_outliers], c='r', marker='X', zorder=100, s=50)
    # axERR.plot(esatan[:, 0], zp_err, label=f'Node 2, z+, RMSE={zp_rmse:.2f}' + r' $\degree$C')
    # axERR.scatter(esatan[:, 0][zp_outliers], zp_err[zp_outliers], c='r', marker='X', zorder=100, s=50)
    axERR.plot(esatan[:, 0], xm_err, label=f'Node 3, x-, RMSE={xm_rmse:.2f}' + r' $\degree$C', linestyle='dashed')
    axERR.scatter(esatan[:, 0][xm_outliers], xm_err[xm_outliers], c='r', marker='X', zorder=100, s=50, label='Outliers')
    # axERR.plot(esatan[:, 0], ym_err, label=f'Node 4, y-, RMSE={ym_rmse:.2f}' + r' $\degree$C')
    # axERR.scatter(esatan[:, 0][ym_outliers], ym_err[ym_outliers], c='r', marker='X', zorder=100, s=50)
    # axERR.plot(esatan[:, 0], zm_err, label=f'Node 5, z-, RMSE={zm_rmse:.2f}' + r' $\degree$C')
    # axERR.scatter(esatan[:, 0][zm_outliers], zm_err[zm_outliers], c='r', marker='X', zorder=100, s=50)
    axERR.set_xlabel('Time [s]')
    axERR.set_ylabel(r"Error in Temperature [$\degree$C]")
    axERR.grid()
    axERR.legend()
    figERR.tight_layout()


def verify_BoxIntRad():
    BoxOrbit = OrbitalModel(h=408e3, surfaces=['x+', 'y+', 'z+', 'x-', 'y-', 'z-'], beta=0., n_orbits=2)
    BoxOrbit.compute()
    t = BoxOrbit.t

    # still must multiply with optical properties
    alpha = 1.0
    epsilon = 1.0
    area = 1.0  # [m^2] (the angles etc. have already been included so all is multiplied here with the entire area)
    OrbitingPlate = NodalModel(t)
    OrbitingPlate.add_node(Node(C_cap=1000, T0=20, q_ext=BoxOrbit.get_heat('x+'), outer=True,
                           alpha=alpha, epsilon=epsilon, area=area, name='x+', origin=(0.5, 0, 0), geom=(0, 1, 1)))
    OrbitingPlate.add_node(Node(C_cap=1000, T0=20, q_ext=BoxOrbit.get_heat('y+'), outer=True,
                           alpha=alpha, epsilon=epsilon, area=area, name='y+', origin=(0, 0.5, 0), geom=(1, 0, 1)))
    OrbitingPlate.add_node(Node(C_cap=1000, T0=20, q_ext=BoxOrbit.get_heat('z+'), outer=True,
                           alpha=alpha, epsilon=epsilon, area=area, name='z+', origin=(0, 0, 0.5), geom=(1, 1, 0)))
    OrbitingPlate.add_node(Node(C_cap=1000, T0=20, q_ext=BoxOrbit.get_heat('x-'), outer=True,
                           alpha=alpha, epsilon=epsilon, area=area, name='x-', origin=(-0.5, 0, 0), geom=(0, 1, 1)))
    OrbitingPlate.add_node(Node(C_cap=1000, T0=20, q_ext=BoxOrbit.get_heat('y-'), outer=True,
                           alpha=alpha, epsilon=epsilon, area=area, name='y-', origin=(0, -0.5, 0), geom=(1, 0, 1)))
    OrbitingPlate.add_node(Node(C_cap=1000, T0=20, q_ext=BoxOrbit.get_heat('z-'), outer=True,
                           alpha=alpha, epsilon=epsilon, area=area, name='z-', origin=(0, 0, -0.5), geom=(1, 1, 0)))
    OrbitingPlate.connect('x+', ['y+', 'z+', 'x-', 'y-', 'z-'], rad=True, C_con=[1, 1, 0, 1, 1])
    OrbitingPlate.connect('y+', ['z+', 'x-', 'y-', 'z-'], rad=True, C_con=[1, 1, 0, 1])
    OrbitingPlate.connect('z+', ['x-', 'y-', 'z-'], rad=True, C_con=[1, 1, 0])
    OrbitingPlate.connect('x-', ['y-', 'z-'], rad=[True, True], C_con=1)
    OrbitingPlate.connect('y-', 'z-', rad=True, C_con=1)
    OrbitingPlate.solve()

    # ESATAN data [[time, temperature1, temperature2, temperature3, temperature4, temperature5, temperature6]]
    # note: for the box model, the thickness must be set to 1 mm, as opposed to the 2 mm (somehow) to make the plates work.
    # inner surface must be set to "active", not "radiative".
    esatan = np.array(pd.read_csv(get_folder_file('ESATAN', 'BetaZero_Z_408km_box_rad_temps_1mmthick.csv'))).T[[0, 5, 7, 1, 9, 3, 11]].T[5:].astype(float)

    markers = ['o', 'x', '+', '*', '^', 'v']
    plt.figure()
    for i in range(OrbitingPlate.n):
        if i == 0 or i == 3:
            plt.plot(OrbitingPlate.t, OrbitingPlate.T[:, i], label=f'Node {i}, {OrbitingPlate.name[i]}, sim.', linewidth=2, zorder=10+i)
            # plt.plot(OrbitingPlateNoRad.t, OrbitingPlateNoRad.T[:, i], label=f'Node {i}, {OrbitingPlate.name[i]}, sim.; no rad.', linewidth=2, zorder=10+i, linestyle='dashed')
    for i in range(OrbitingPlate.n):  # separate for-loop for legend ordering in the figure
        # ::10 makes esatan data more sparse to see more easily
        # scatter plot instead of line plot because the lines almost perfectly overlap, making it invisible.
        if i == 0 or i == 3:
            plt.scatter(esatan[::10, 0], esatan[::10, i+1], label=f'Node {i}, ESATAN', marker=markers[i], color='k')

    plt.title('Temperatures of Orbiting Box\nWith Internal Radiation')
    plt.legend(loc='lower left').set_zorder(999)
    plt.grid()
    plt.xlabel('Time [s]')
    plt.ylabel(r'Temperature [$\degree$C]')
    plt.tight_layout()

    # compute RMSE
    xp_err, temp_new = calc_err(esatan[:, 1], OrbitingPlate.T[:, 0], esatan[:, 0], OrbitingPlate.t)
    yp_err, temp_new = calc_err(esatan[:, 2], OrbitingPlate.T[:, 1], esatan[:, 0], OrbitingPlate.t)
    zp_err, temp_new = calc_err(esatan[:, 3], OrbitingPlate.T[:, 2], esatan[:, 0], OrbitingPlate.t)
    xm_err, temp_new = calc_err(esatan[:, 4], OrbitingPlate.T[:, 3], esatan[:, 0], OrbitingPlate.t)
    ym_err, temp_new = calc_err(esatan[:, 5], OrbitingPlate.T[:, 4], esatan[:, 0], OrbitingPlate.t)
    zm_err, temp_new = calc_err(esatan[:, 6], OrbitingPlate.T[:, 5], esatan[:, 0], OrbitingPlate.t)
    xp_rmse, xp_outliers = calc_simple_rmse(xp_err, outlier_sens=10)
    yp_rmse, yp_outliers = calc_simple_rmse(yp_err, outlier_sens=10)
    zp_rmse, zp_outliers = calc_simple_rmse(zp_err, outlier_sens=10)
    xm_rmse, xm_outliers = calc_simple_rmse(xm_err, outlier_sens=10)
    ym_rmse, ym_outliers = calc_simple_rmse(ym_err, outlier_sens=10)
    zm_rmse, zm_outliers = calc_simple_rmse(zm_err, outlier_sens=10)

    figERR = plt.figure()
    axERR = figERR.add_subplot()
    axERR.set_title(f'Error in Temperatures\nWith Internal Radiation')
    axERR.plot(esatan[:, 0], xp_err, label=f'Node 0, x+, RMSE={xp_rmse:.2f}' + r' $\degree$C')
    axERR.scatter(esatan[:, 0][xp_outliers], xp_err[xp_outliers], c='r', marker='X', zorder=100, s=50)
    # axERR.plot(esatan[:, 0], yp_err, label=f'Node 1, y+, RMSE={yp_rmse:.2f}' + r' $\degree$C')
    # axERR.scatter(esatan[:, 0][yp_outliers], yp_err[yp_outliers], c='r', marker='X', zorder=100, s=50)
    # axERR.plot(esatan[:, 0], zp_err, label=f'Node 2, z+, RMSE={zp_rmse:.2f}' + r' $\degree$C')
    # axERR.scatter(esatan[:, 0][zp_outliers], zp_err[zp_outliers], c='r', marker='X', zorder=100, s=50)
    axERR.plot(esatan[:, 0], xm_err, label=f'Node 3, x-, RMSE={xm_rmse:.2f}' + r' $\degree$C', linestyle='dashed')
    axERR.scatter(esatan[:, 0][xm_outliers], xm_err[xm_outliers], c='r', marker='X', zorder=100, s=50, label='Outliers')
    # axERR.plot(esatan[:, 0], ym_err, label=f'Node 4, y-, RMSE={ym_rmse:.2f}' + r' $\degree$C')
    # axERR.scatter(esatan[:, 0][ym_outliers], ym_err[ym_outliers], c='r', marker='X', zorder=100, s=50)
    # axERR.plot(esatan[:, 0], zm_err, label=f'Node 5, z-, RMSE={zm_rmse:.2f}' + r' $\degree$C')
    # axERR.scatter(esatan[:, 0][zm_outliers], zm_err[zm_outliers], c='r', marker='X', zorder=100, s=50)
    axERR.set_xlabel('Time [s]')
    axERR.set_ylabel(r"Error in Temperature [$\degree$C]")
    axERR.grid()
    axERR.legend()
    figERR.tight_layout()


def verify_RectBoxIntRad():
    BoxOrbit = OrbitalModel(h=408e3, surfaces=['x+', 'y+', 'z+', 'x-', 'y-', 'z-'], beta=0., n_orbits=2)
    BoxOrbit.compute()
    t = BoxOrbit.t

    # still must multiply with optical properties
    alpha = 1.0
    epsilon = 1.0

    OrbitingPlate = NodalModel(t)
    OrbitingPlate.add_node(Node(C_cap=2000, T0=20, q_ext=BoxOrbit.get_heat('x+'), outer=True,
                           alpha=alpha, epsilon=epsilon, name='x+', origin=(0.5, 0, 0), geom=(0, 1, 2)))
    OrbitingPlate.add_node(Node(C_cap=2000, T0=20, q_ext=BoxOrbit.get_heat('y+'), outer=True,
                           alpha=alpha, epsilon=epsilon, name='y+', origin=(0, 0.5, 0), geom=(1, 0, 2)))
    OrbitingPlate.add_node(Node(C_cap=1000, T0=20, q_ext=BoxOrbit.get_heat('z+'), outer=True,
                           alpha=alpha, epsilon=epsilon, name='z+', origin=(0, 0, 1), geom=(1, 1, 0)))
    OrbitingPlate.add_node(Node(C_cap=2000, T0=20, q_ext=BoxOrbit.get_heat('x-'), outer=True,
                           alpha=alpha, epsilon=epsilon, name='x-', origin=(-0.5, 0, 0), geom=(0, 1, 2)))
    OrbitingPlate.add_node(Node(C_cap=2000, T0=20, q_ext=BoxOrbit.get_heat('y-'), outer=True,
                           alpha=alpha, epsilon=epsilon, name='y-', origin=(0, -0.5, 0), geom=(1, 0, 2)))
    OrbitingPlate.add_node(Node(C_cap=1000, T0=20, q_ext=BoxOrbit.get_heat('z-'), outer=True,
                           alpha=alpha, epsilon=epsilon, name='z-', origin=(0, 0, -1), geom=(1, 1, 0)))
    OrbitingPlate.connect('x+', ['y+', 'z+', 'x-', 'y-', 'z-'], rad=True, C_con=[2, 2/3, 0, 2, 2/3])
    OrbitingPlate.connect('y+', ['z+', 'x-', 'y-', 'z-'], rad=True, C_con=[2/3, 2, 0, 2/3])
    OrbitingPlate.connect('z+', ['x-', 'y-', 'z-'], rad=True, C_con=[2/3, 2/3, 0])
    OrbitingPlate.connect('x-', ['y-', 'z-'], rad=[True, True], C_con=[2, 2/3])
    OrbitingPlate.connect('y-', 'z-', rad=True, C_con=2/3)
    OrbitingPlate.solve()

    # ESATAN data [[time, temperature1, temperature2, temperature3, temperature4, temperature5, temperature6]]
    # note: for the box model, the thickness must be set to 1 mm, as opposed to the 2 mm (somehow) to make the plates work.
    # inner surface must be set to "active", not "radiative".
    esatan = np.array(pd.read_csv(get_folder_file('ESATAN', 'BetaZero_Z_408km_rectbox_rad_temps_1mmthick.csv'))).T[[0, 5, 7, 1, 9, 3, 11]].T[5:].astype(float)

    markers = ['o', 'x', '+', '*', '^', 'v']
    plt.figure()
    for i in range(OrbitingPlate.n):
        plt.plot(OrbitingPlate.t, OrbitingPlate.T[:, i], label=f'Node {i}, {OrbitingPlate.name[i]}, sim.', linewidth=2, zorder=10+i)
    for i in range(OrbitingPlate.n):  # separate for-loop for legend ordering in the figure
        # ::10 makes esatan data more sparse to see more easily
        # scatter plot instead of line plot because the lines almost perfectly overlap, making it invisible.
        plt.scatter(esatan[::10, 0], esatan[::10, i+1], label=f'Node {i}, ESATAN', marker=markers[i], color='k')

    plt.title('Temperatures of Orbiting Rectangular Box With Internal Radiation')
    plt.legend(loc='lower left')
    plt.grid()
    plt.xlabel('Time [s]')
    plt.ylabel(r'Temperature [$\degree$C]')


def verify_RectBoxIntRadHalfE():
    BoxOrbit = OrbitalModel(h=408e3, surfaces=['x+', 'y+', 'z+', 'x-', 'y-', 'z-'], beta=0., n_orbits=2)
    BoxOrbit.compute()
    t = BoxOrbit.t

    # still must multiply with optical properties
    alpha = 1.0
    epsilon = 0.5

    OrbitingPlate = NodalModel(t)
    OrbitingPlate.add_node(Node(C_cap=2000, T0=20, q_ext=BoxOrbit.get_heat('x+'), outer=True,
                           alpha=alpha, epsilon=epsilon, name='x+', origin=(0.5, 0, 0), geom=(0, 1, 2)))
    OrbitingPlate.add_node(Node(C_cap=2000, T0=20, q_ext=BoxOrbit.get_heat('y+'), outer=True,
                           alpha=alpha, epsilon=epsilon, name='y+', origin=(0, 0.5, 0), geom=(1, 0, 2)))
    OrbitingPlate.add_node(Node(C_cap=1000, T0=20, q_ext=BoxOrbit.get_heat('z+'), outer=True,
                           alpha=alpha, epsilon=epsilon, name='z+', origin=(0, 0, 1), geom=(1, 1, 0)))
    OrbitingPlate.add_node(Node(C_cap=2000, T0=20, q_ext=BoxOrbit.get_heat('x-'), outer=True,
                           alpha=alpha, epsilon=epsilon, name='x-', origin=(-0.5, 0, 0), geom=(0, 1, 2)))
    OrbitingPlate.add_node(Node(C_cap=2000, T0=20, q_ext=BoxOrbit.get_heat('y-'), outer=True,
                           alpha=alpha, epsilon=epsilon, name='y-', origin=(0, -0.5, 0), geom=(1, 0, 2)))
    OrbitingPlate.add_node(Node(C_cap=1000, T0=20, q_ext=BoxOrbit.get_heat('z-'), outer=True,
                           alpha=alpha, epsilon=epsilon, name='z-', origin=(0, 0, -1), geom=(1, 1, 0)))
    OrbitingPlate.connect('x+', ['y+', 'z+', 'x-', 'y-', 'z-'], rad=True, C_con=[2, 2/3, 0, 2, 2/3])
    OrbitingPlate.connect('y+', ['z+', 'x-', 'y-', 'z-'], rad=True, C_con=[2/3, 2, 0, 2/3])
    OrbitingPlate.connect('z+', ['x-', 'y-', 'z-'], rad=True, C_con=[2/3, 2/3, 0])
    OrbitingPlate.connect('x-', ['y-', 'z-'], rad=[True, True], C_con=[2, 2/3])
    OrbitingPlate.connect('y-', 'z-', rad=True, C_con=2/3)
    OrbitingPlate.solve()

    # ESATAN data [[time, temperature1, temperature2, temperature3, temperature4, temperature5, temperature6]]
    # note: for the box model, the thickness must be set to 1 mm, as opposed to the 2 mm (somehow) to make the plates work.
    # inner surface must be set to "active", not "radiative".
    esatan = np.array(pd.read_csv(get_folder_file('ESATAN', 'BetaZero_Z_408km_rectbox_rad_temps_1mmthick_halfemissivity.csv'))).T[[0, 5, 7, 1, 9, 3, 11]].T[5:].astype(float)

    markers = ['o', 'x', '+', '*', '^', 'v']
    plt.figure()
    for i in range(OrbitingPlate.n):
        plt.plot(OrbitingPlate.t, OrbitingPlate.T[:, i], label=f'Node {i}, {OrbitingPlate.name[i]}, sim.', linewidth=2, zorder=10+i)
    for i in range(OrbitingPlate.n):  # separate for-loop for legend ordering in the figure
        # ::10 makes esatan data more sparse to see more easily
        # scatter plot instead of line plot because the lines almost perfectly overlap, making it invisible.
        plt.scatter(esatan[::10, 0], esatan[::10, i+1], label=f'Node {i}, ESATAN', marker=markers[i], color='k')

    plt.title('Temperatures of Orbiting Rectangular Box With Internal Radiation\n and All Emissivities 0.5')
    plt.legend(loc='lower left')
    plt.grid()
    plt.xlabel('Time [s]')
    plt.ylabel(r'Temperature [$\degree$C]')


def verify_RectBoxIntRadLowE():
    BoxOrbit = OrbitalModel(h=408e3, surfaces=['x+', 'y+', 'z+', 'x-', 'y-', 'z-'], beta=0., n_orbits=2)
    BoxOrbit.compute()
    t = BoxOrbit.t
    # still must multiply with optical properties
    alpha = 1.0
    epsilon = 0.1

    OrbitingPlate = NodalModel(t)
    OrbitingPlate.add_node(Node(C_cap=2000, T0=20, q_ext=BoxOrbit.get_heat('x+'), outer=True,
                           alpha=alpha, epsilon=epsilon, name='x+', origin=(0.5, 0, 0), geom=(0, 1, 2)))
    OrbitingPlate.add_node(Node(C_cap=2000, T0=20, q_ext=BoxOrbit.get_heat('y+'), outer=True,
                           alpha=alpha, epsilon=epsilon, name='y+', origin=(0, 0.5, 0), geom=(1, 0, 2)))
    OrbitingPlate.add_node(Node(C_cap=1000, T0=20, q_ext=BoxOrbit.get_heat('z+'), outer=True,
                           alpha=alpha, epsilon=epsilon, name='z+', origin=(0, 0, 1), geom=(1, 1, 0)))
    OrbitingPlate.add_node(Node(C_cap=2000, T0=20, q_ext=BoxOrbit.get_heat('x-'), outer=True,
                           alpha=alpha, epsilon=epsilon, name='x-', origin=(-0.5, 0, 0), geom=(0, 1, 2)))
    OrbitingPlate.add_node(Node(C_cap=2000, T0=20, q_ext=BoxOrbit.get_heat('y-'), outer=True,
                           alpha=alpha, epsilon=epsilon, name='y-', origin=(0, -0.5, 0), geom=(1, 0, 2)))
    OrbitingPlate.add_node(Node(C_cap=1000, T0=20, q_ext=BoxOrbit.get_heat('z-'), outer=True,
                           alpha=alpha, epsilon=epsilon, name='z-', origin=(0, 0, -1), geom=(1, 1, 0)))
    OrbitingPlate.connect('x+', ['y+', 'z+', 'x-', 'y-', 'z-'], rad=True, C_con=[2, 2/3, 0, 2, 2/3])
    OrbitingPlate.connect('y+', ['z+', 'x-', 'y-', 'z-'], rad=True, C_con=[2/3, 2, 0, 2/3])
    OrbitingPlate.connect('z+', ['x-', 'y-', 'z-'], rad=True, C_con=[2/3, 2/3, 0])
    OrbitingPlate.connect('x-', ['y-', 'z-'], rad=[True, True], C_con=[2, 2/3])
    OrbitingPlate.connect('y-', 'z-', rad=True, C_con=2/3)
    OrbitingPlate.solve()

    # ESATAN data [[time, temperature1, temperature2, temperature3, temperature4, temperature5, temperature6]]
    # note: for the box model, the thickness must be set to 1 mm, as opposed to the 2 mm (somehow) to make the plates work.
    # inner surface must be set to "active", not "radiative".
    esatan = np.array(pd.read_csv(get_folder_file('ESATAN', 'BetaZero_Z_408km_rectbox_rad_temps_1mmthick_lowemissivity.csv'))).T[[0, 5, 7, 1, 9, 3, 11]].T[5:].astype(float)

    markers = ['o', 'x', '+', '*', '^', 'v']
    plt.figure()
    for i in range(OrbitingPlate.n):
        plt.plot(OrbitingPlate.t, OrbitingPlate.T[:, i], label=f'Node {i}, {OrbitingPlate.name[i]}, sim.', linewidth=2, zorder=10+i)
    for i in range(OrbitingPlate.n):  # separate for-loop for legend ordering in the figure
        # ::10 makes esatan data more sparse to see more easily
        # scatter plot instead of line plot because the lines almost perfectly overlap, making it invisible.
        plt.scatter(esatan[::10, 0], esatan[::10, i+1], label=f'Node {i}, ESATAN', marker=markers[i], color='k')

    plt.title('Temperatures of Orbiting Rectangular Box With Internal Radiation\n and All Emissivities 0.1')
    plt.legend(loc='lower left')
    plt.grid()
    plt.xlabel('Time [s]')
    plt.ylabel(r'Temperature [$\degree$C]')


def validate_FUNcube():
    # The FUNcube model must be solved before running this function.
    data_x_raw = np.array(pd.read_csv(get_folder_file('FUNcubeData', 'FUNcube_temps_x.csv')))
    data_xm_raw = np.array(pd.read_csv(get_folder_file('FUNcubeData', 'FUNcube_temps_xm.csv')))
    data_y_raw = np.array(pd.read_csv(get_folder_file('FUNcubeData', 'FUNcube_temps_y.csv')))
    data_ym_raw = np.array(pd.read_csv(get_folder_file('FUNcubeData', 'FUNcube_temps_ym.csv')))

    # Assign the face names such that they match the simulation.
    # FUNcube appears to fly "backwards" with z- face pointing in the flight direction.
    data_x = np.array(data_x_raw)
    data_xm = np.array(data_xm_raw)
    data_y = np.array(data_ym_raw)
    data_ym = np.array(data_y_raw)

    fig = plt.figure()
    fig.suptitle('Whole Orbit Data of FUNcube\n(Simulation and Flight Data)\nDate: 13 August 2024')
    ax = fig.add_subplot()
    ax.scatter(data_x[:, 0]+100, data_x[:, 1], label='Flight, x+', marker='o')
    ax.scatter(data_xm[:, 0]+100, data_xm[:, 1], label='Flight, x-', marker='x')
    ax.scatter(data_y[:, 0]+100, data_y[:, 1], label='Flight, y+', marker='^')
    ax.scatter(data_ym[:, 0]+100, data_ym[:, 1], label='Flight, y-', marker='s')

    indices = []
    desired_nodes = ['x+A', 'x-A', 'y+A', 'y-A']
    for i in range(len(desired_nodes)):
        indices.append(np.argwhere(FUNcubeModel.name == desired_nodes[i])[0, 0])

    t_dat = FUNcubeModel.t/60-77
    t_idx = np.argwhere(np.logical_and(-5 < t_dat, t_dat < 100))
    # auto-generated labels would be in for-loop with append f'Sim., {desired_nodes[i][:-1]}', but define manually to match FUNcube
    labels = ['Sim., x+', 'Sim., x-', 'Sim., y+', 'Sim., y-']
    linestyles = ['solid', 'dashed', 'dashdot', 'dotted']
    for i in range(len(indices)):  # different order to match the line order and colour with the flight data
        idx = indices[i]
        ax.plot(t_dat[t_idx][:, 0], FUNcubeModel.T[t_idx, idx][:, 0], label=labels[i], linestyle=linestyles[i])

    ax.set_xlabel('Time [min]')
    ax.set_ylabel(r'Temperature [$\degree$C]')
    ax.legend(loc='upper right')
    ax.grid()
    fig.tight_layout()

    # compute RMSE
    # Indices for xp & xm, and for yp & ym are swapped to match FUNcube's orientation
    xp_err, temp_new = calc_err(data_x[:, 1], FUNcubeModel.T[t_idx, indices[0]][:, 0], data_x[:, 0]+100, t_dat[t_idx][:, 0])
    xm_err, temp_new = calc_err(data_xm[:, 1], FUNcubeModel.T[t_idx, indices[1]][:, 0], data_xm[:, 0]+100, t_dat[t_idx][:, 0])
    yp_err, temp_new = calc_err(data_y[:, 1], FUNcubeModel.T[t_idx, indices[2]][:, 0], data_y[:, 0]+100, t_dat[t_idx][:, 0])
    ym_err, temp_new = calc_err(data_ym[:, 1], FUNcubeModel.T[t_idx, indices[3]][:, 0], data_ym[:, 0]+100, t_dat[t_idx][:, 0])
    xp_rmse, xp_outliers = calc_simple_rmse(xp_err, outlier_sens=10)
    xm_rmse, xm_outliers = calc_simple_rmse(xm_err, outlier_sens=10)
    yp_rmse, yp_outliers = calc_simple_rmse(yp_err, outlier_sens=10)
    ym_rmse, ym_outliers = calc_simple_rmse(ym_err, outlier_sens=10)

    figERR = plt.figure()
    axERR = figERR.add_subplot()
    axERR.set_title(f'Error in Temperatures\nDate: 13 August 2024')
    axERR.plot(data_x[:, 0]+100, xp_err, label=f'x+, RMSE={xp_rmse:.2f}' + r' $\degree$C')
    axERR.scatter(data_x[:, 0][xp_outliers]+100, xp_err[xp_outliers], c='r', marker='X', zorder=100, s=50)
    axERR.plot(data_xm[:, 0]+100, xm_err, label=f'x-, RMSE={xm_rmse:.2f}' + r' $\degree$C', linestyle='dotted')
    axERR.scatter(data_xm[:, 0][xm_outliers]+100, xm_err[xm_outliers], c='r', marker='X', zorder=100, s=50)
    axERR.plot(data_y[:, 0]+100, yp_err, label=f'y+, RMSE={yp_rmse:.2f}' + r' $\degree$C', linestyle='dashed')
    axERR.scatter(data_y[:, 0][yp_outliers]+100, yp_err[yp_outliers], c='r', marker='X', zorder=100, s=50)
    axERR.plot(data_ym[:, 0]+100, ym_err, label=f'y-, RMSE={ym_rmse:.2f}' + r' $\degree$C', linestyle='dashdot')
    axERR.scatter(data_ym[:, 0][ym_outliers]+100, ym_err[ym_outliers], c='r', marker='X', zorder=100, s=50, label='Outliers')
    axERR.set_xlabel('Time [s]')
    axERR.set_ylabel(r"Error in Temperature [$\degree$C]")
    axERR.grid()
    axERR.legend(loc='upper right')
    figERR.tight_layout()


    """# Plot the raw data of the 2016 flight data
    wholeday = np.array(pd.read_csv(get_folder_file('FUNcubeData', 'FUNcube_oneday.csv')))[:, 1:]
    t_wholeday = np.arange(0, wholeday.shape[0], 1)  # Time in minutes
    fig = plt.figure()
    fig.suptitle('Whole Orbit Data of FUNcube\n(Simulation and Flight Data)\nDate: 4 February 2016')
    ax = fig.add_subplot()
    ax.plot(t_wholeday, wholeday[:, 4], label='Flight, x+', marker='o')
    ax.plot(t_wholeday, wholeday[:, 5], label='Flight, x-', marker='x')
    ax.plot(t_wholeday, wholeday[:, 6], label='Flight, y+', marker='^')
    ax.plot(t_wholeday, wholeday[:, 7], label='Flight, y-', marker='s')
    ax.grid()
    ax.legend()"""

    # Run FUNcube simulation again, but now with angular rates (for older, whole day data)
    FUNcubeOrbit.modify(angular_rates_new=(0., 0., 2.), day_new=35, h_new=644e3)
    FUNcubeOrbit.compute()

    # Need to re-assign all heat/power parameters because the altitude changes the time array.
    FUNcubeModel.set_time(t=FUNcubeOrbit.t, erase=True)
    assign_q_ext_to_pcb(FUNcubeModel, FUNcubeOrbit)
    FUNcubeModel.modify_node('PCB1A', P_int_new=P_tot/5)
    FUNcubeModel.modify_node('PCB2A', P_int_new=P_tot/5)
    FUNcubeModel.modify_node('PCB3A', P_int_new=P_tot/5)
    FUNcubeModel.modify_node('PCB4A', P_int_new=P_tot/5)
    FUNcubeModel.modify_node('PCB5A', P_int_new=P_tot/5)

    FUNcubeModel.solve()

    wholeday = np.array(pd.read_csv(get_folder_file('FUNcubeData', 'FUNcube_oneday.csv')))[:, 1:]
    wholeday_x_raw = wholeday[:, 4]
    wholeday_xm_raw = wholeday[:, 5]
    wholeday_y_raw = wholeday[:, 6]
    wholeday_ym_raw = wholeday[:, 7]

    # Assign the face names such that they match the simulation.
    # FUNcube appears to fly "backwards" with z- face pointing in the flight direction.
    wholeday_x = np.array(wholeday_x_raw)
    wholeday_xm = np.array(wholeday_xm_raw)
    wholeday_y = np.array(wholeday_ym_raw)
    wholeday_ym = np.array(wholeday_y_raw)

    t_wholeday = np.arange(0, wholeday.shape[0], 1)  # Time in minutes
    fig = plt.figure()
    fig.suptitle('Whole Orbit Data of FUNcube\n(Simulation and Flight Data)\nDate: 4 February 2016')
    ax = fig.add_subplot()
    idx_start = 1254
    idx_end = 1350
    ax.plot(t_wholeday[idx_start:idx_end]-idx_start, wholeday_x[idx_start:idx_end], label='Flight, x+', marker='o')
    ax.plot(t_wholeday[idx_start:idx_end]-idx_start, wholeday_xm[idx_start:idx_end], label='Flight, x-', marker='x')
    ax.plot(t_wholeday[idx_start:idx_end]-idx_start, wholeday_y[idx_start:idx_end], label='Flight, y+', marker='^')
    ax.plot(t_wholeday[idx_start:idx_end]-idx_start, wholeday_ym[idx_start:idx_end], label='Flight, y-', marker='s')

    t_dat = FUNcubeModel.t/60-65
    t_idx = np.argwhere(np.logical_and(0 < t_dat, t_dat < 96))
    # auto-generated labels would be in for-loop with append f'Sim., {desired_nodes[i][:-1]}', but define manually to match FUNcube
    labels = ['Sim., x-', 'Sim., x+ ', 'Sim., y-', 'Sim., y+']
    linestyles = ['solid', 'dashed', 'dotted', 'dashdot']
    for i, idx in enumerate(indices):
        ax.plot(t_dat[t_idx], FUNcubeModel.T[t_idx, idx], label=labels[i], linestyle=linestyles[i])
    ax.set_xlabel('Time [min]')
    ax.set_ylabel(r'Temperature [$\degree$C]')

    ax.legend(loc='upper right')
    ax.grid()
    fig.tight_layout()

    # compute RMSE
    # Indices for xp & xm, and for yp & ym are swapped to match FUNcube's orientation
    xp_err, temp_new = calc_err(wholeday_x[idx_start:idx_end], FUNcubeModel.T[t_idx, indices[0]][:, 0], t_wholeday[idx_start:idx_end]-idx_start, t_dat[t_idx][:, 0])
    xm_err, temp_new = calc_err(wholeday_xm[idx_start:idx_end], FUNcubeModel.T[t_idx, indices[1]][:, 0], t_wholeday[idx_start:idx_end]-idx_start, t_dat[t_idx][:, 0])
    yp_err, temp_new = calc_err(wholeday_y[idx_start:idx_end], FUNcubeModel.T[t_idx, indices[2]][:, 0], t_wholeday[idx_start:idx_end]-idx_start, t_dat[t_idx][:, 0])
    ym_err, temp_new = calc_err(wholeday_ym[idx_start:idx_end], FUNcubeModel.T[t_idx, indices[3]][:, 0], t_wholeday[idx_start:idx_end]-idx_start, t_dat[t_idx][:, 0])
    xp_rmse, xp_outliers = calc_simple_rmse(xp_err, outlier_sens=10)
    xm_rmse, xm_outliers = calc_simple_rmse(xm_err, outlier_sens=10)
    yp_rmse, yp_outliers = calc_simple_rmse(yp_err, outlier_sens=10)
    ym_rmse, ym_outliers = calc_simple_rmse(ym_err, outlier_sens=10)

    figERR = plt.figure()
    axERR = figERR.add_subplot()
    axERR.set_title(f'Error in Temperatures\nDate: 4 February 2016')
    axERR.plot(t_wholeday[idx_start:idx_end]-idx_start, xp_err, label=f'x+, RMSE={xp_rmse:.2f}' + r' $\degree$C')
    axERR.scatter(t_wholeday[idx_start:idx_end][xp_outliers]-idx_start, xp_err[xp_outliers], c='r', marker='X', zorder=100, s=50)
    axERR.plot(t_wholeday[idx_start:idx_end]-idx_start, xm_err, label=f'x-, RMSE={xm_rmse:.2f}' + r' $\degree$C', linestyle='dotted')
    axERR.scatter(t_wholeday[idx_start:idx_end][xm_outliers]-idx_start, xm_err[xm_outliers], c='r', marker='X', zorder=100, s=50)
    axERR.plot(t_wholeday[idx_start:idx_end]-idx_start, yp_err, label=f'y+, RMSE={yp_rmse:.2f}' + r' $\degree$C', linestyle='dashed')
    axERR.scatter(t_wholeday[idx_start:idx_end][yp_outliers]-idx_start, yp_err[yp_outliers], c='r', marker='X', zorder=100, s=50)
    axERR.plot(t_wholeday[idx_start:idx_end]-idx_start, ym_err, label=f'y-, RMSE={ym_rmse:.2f}' + r' $\degree$C', linestyle='dashdot')
    axERR.scatter(t_wholeday[idx_start:idx_end][ym_outliers]-idx_start, ym_err[ym_outliers], c='r', marker='X', zorder=100, s=50, label='Outliers')
    axERR.set_xlabel('Time [s]')
    axERR.set_ylabel(r"Error in Temperature [$\degree$C]")
    axERR.grid()
    axERR.legend(loc='lower right')
    figERR.tight_layout()


# verify_viewfactors()
# verify_eclipse()
# verify_beta_from_parameters()
# verify_EnvironmentRadiation()
# verify_TransientAnalysis()
# verify_TwoPlatesNoIntRad()
# verify_BoxNoIntRad()
# verify_BoxIntRad()
# verify_RectBoxIntRad()
# verify_RectBoxIntRadHalfE()
# verify_RectBoxIntRadLowE()
validate_FUNcube()
plt.show()
