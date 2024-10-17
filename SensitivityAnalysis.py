# -*- coding: utf-8 -*-
"""
Created on Tue May 21 15:30:55 2024
Author: Frank Meijering (Delft University of Technology)

SensitivityAnalysis.py contains numerous functions to perform a sensitivity analysis on your custom models.
The 'run_...' functions compute the cases and store them in Pickle (.pkl) files. The 'plot_...' functions read
those files and plot them accordingly.
"""


import numpy as np
from matplotlib import pyplot as plt
from matplotlib import markers as mkr
import os
from ThermalBudget import NodalModel, Node, OrbitalModel, ensure_list
import pickle as pkl
import copy
from FUNcube import FUNcubeModel, FUNcubeOrbit


def get_file(file):
    """
    Returns the path of a file in the same directory as this Python file; works on most operating systems.

    :param file: Name (str) of the file to be retrieved.
    :return: Path of the file.
    """
    return os.path.join(os.path.dirname(__file__), file)


def get_folder_file(folder, file, subfolders=()):
    """
    Returns the path of a file in a folder within the same directory as this Python file;
    works on most operating systems.

    Subfolders can be an arbitrary number of sub-folders, such as:
    get_folder_file('folderA', 'file.txt', ('folderB', 'folderC'))
    --> C:/...your-directory.../folderA/folderB/folderC/file

    :param folder: Name (str) of the folder which the file is in.
    :param file: Name (str) of the file to be retrieved.
    :param subfolders: Tuple/list of names (str) of any sub-folders which the file is in.
    :return: Path of the file.
    """
    for i in reversed(range(len(subfolders))):
        file = os.path.join(subfolders[i], file)
    return get_file(os.path.join(folder, file))


def copy_orbital_nodalmodel(orbital_old, nodal_old):
    """
    Makes a deepcopy of an OrbitalModel and NodalModel object, while keeping the links between them. This is needed
    for the sensitivity analysis, since the original object should remain unchanged.

    :param orbital_old: OrbitalModel object to be copied.
    :param nodal_old: NodalModel object to be copied.
    :return: OrbitalModel copy, NodalModel copy.
    """
    orbital_new = copy.deepcopy(orbital_old)
    nodal_new = copy.deepcopy(nodal_old)
    new_surfs = []
    for surf in orbital_new.surfaces:  # for-loop always possible since surfaces is ensured to be a list
        if isinstance(surf, Node):
            idx = np.argwhere(surf.name == nodal_new.name)[0, 0]
            new_surfs.append(nodal_new.nodes[idx])
        else:
            new_surfs.append(surf)
    orbital_new.modify(surfaces_new=new_surfs)
    return orbital_new, nodal_new


def calc_rmse(y_short, y_long, multip):
    """
    Calculate the Root Mean Square Error (RMSE) between two differently shaped, but overlapping, arrays.
    The short array (large time step) only has fewer data points than the long array (small time step).
    If multip is 2, y_long is twice as long as y_short, i.e. the time step of y_long is twice as small as that of
    y_short. The x-values of y_short must be present in y_long; for example:
    x_short = [0, 2, 4] and x_long =[0, 1, 2, 3, 4] for multip = 2.

    :param y_short: Y-values of the short array (large time step).
    :param y_long: Y-values of the long array (small time step).
    :param multip: Integer value showing by which multiple the time step differs. See example above.
    :return: Root Mean Square Error (RMSE) between y_short and y_long.
    """
    y_long_short = y_long[::multip]
    if y_short.shape[0] != y_long_short.shape[0]:
        if y_short.shape[0] > y_long_short.shape[0]:
            y_short = y_short[:y_long_short.shape[0]]
        else:
            y_long_short = y_long_short[:y_short.shape[0]]
    return np.sqrt(np.sum((y_short-y_long_short)**2)/y_short.shape[0])


def read_folder(folder='SensitivityAnalysis', subfolders=()):
    """
    Reads all pickle files available in the given folder.

    :param folder: Name (str) of the folder where the pickle files are to be read from.
    :param subfolders: Tuple/list of names (str) of any sub-folders which the file is in.
    :return: cases_names, cases_results: list of names of the cases/files, and the solved NodalModels for all cases.
    """
    cases_results = []
    folder_listdir = f'{folder}'
    for subfolder in subfolders:
        folder_listdir = os.path.join(folder_listdir, subfolder)
    cases_names = os.listdir(folder_listdir)

    for filename in cases_names:
        with open(get_folder_file(folder, filename, subfolders), 'rb') as f:
            cases_results.append(pkl.load(f))

    return cases_names, cases_results


def run_case(variable, value, nodal_model, orbital_model, printing=True, solver='Radau', limit_step=False, interp_inputs=False):
    """
    Runs an orbital and transient analysis for the given input parameters.

    The parameter 'value' is the relative fraction for all parameters (except DAY, BETA, DT, and RAD),
    since the use of absolute numbers would potentially alter the entire nature of the NodalModel.

    If the altitude or time step is changed, the time array changes as well. Hence, the internal power cannot follow
    the exact same transient as before (except if it is constant), so an average internal power will be assumed along
    the entire timeline. If the power is constant, this is no issue and the same value is used.

    If any input parameter is by default zero, any multiplications (+-10% etc.) still result in zero; this could be a
    reason why the sensitivity analysis has no impact on certain parameters, e.g. if the internal power is zero.

    :param variable: Name (string) of the variable which is being changed.
    :param value: Multiplication factor if the variable is either of: [ALT, CAP, IR, ALB, SOL, POW, CON, EMI, ABS, ROT],
                  and the actual absolute value if the variable is either of: [DAY, BETA, DT, RAD].
                  For RAD, the value should be 'on' or 'off' (string).
    :param nodal_model: NodalModel to be subjected to the sensitivity analysis.
    :param orbital_model: OrbitalModel to be subjected to the sensitivity analysis.
    :param printing: Boolean indicating whether progress should be printed in the command line.
    :param solver: Numerical integration method, must be either of: ['rk45', 'dop853', 'radau'].
                   Default is Radau.
    :param limit_step: Boolean used for scipy's variable time stepping, indicating whether the solver can use its
                       optimisation algorithms to skip time steps and reduce computational time (limit_step=False),
                       or whether a higher accuracy is desired and no time steps are allowed to be skipped
                       (limit_step=True). Default is False.
    :param interp_inputs: Boolean indicating whether the environmental inputs (q_pla, q_alb, q_s, P_int) should
                          be interpolated during scipy's variable time stepping. Improves accuracy, but also
                          increases computational time. Default is False.
    :return: NodalModel object with the solved case.
    """
    if printing:
        print(f'{variable}_{value}'.replace('.', '_'))

    nodal_model.T0[0] = None  # De-activate any previously computed initial temperatures (those may be based on outdated data)
    orbital_model, nodal_model = copy_orbital_nodalmodel(orbital_model, nodal_model)  # make a copy to avoid mutability

    recompute_orbit = False  # only need to compute OrbitalModel again if orbit parameters are changed
    if variable == 'DAY':
        orbital_model.modify(day_new=value)
        recompute_orbit = True
    elif variable == 'ALT':
        orbital_model.modify(h_new=orbital_model.h*value)
        recompute_orbit = True
    elif variable == 'BETA':
        orbital_model.modify(beta_new=value)
        recompute_orbit = True
    elif variable == 'ROT':
        orbital_model.modify(angular_rates_new=tuple(np.array(orbital_model.angular_rates)*180./np.pi*value))  # takes deg as input, so must convert
        recompute_orbit = True
    elif variable == 'DT':
        orbital_model.modify(dt_new=value)
        recompute_orbit = True

    elif variable == 'CAP':
        for node in nodal_model.nodes:
            nodal_model.modify_node(node, C_cap_new=node.C_cap*value)
    elif variable == 'IR':
        for node in nodal_model.nodes:
            if node.q_pla is not None:
                nodal_model.modify_node(node, q_ext_new=tuple((node.q_pla*value, node.q_alb, node.q_s)))
    elif variable == 'ALB':
        for node in nodal_model.nodes:
            if node.q_alb is not None:
                nodal_model.modify_node(node, q_ext_new=tuple((node.q_pla, node.q_alb*value, node.q_s)))
    elif variable == 'SOL':
        for node in nodal_model.nodes:
            if node.q_s is not None:
                # not only solar power, but also albedo must be multiplied.
                nodal_model.modify_node(node, q_ext_new=tuple((node.q_pla, node.q_alb*value, node.q_s*value)))
    elif variable == 'POW':
        for node in nodal_model.nodes:
            if node.P_int is not None:
                nodal_model.modify_node(node, P_int_new=node.P_int*value)
    elif variable == 'CON':
        nodal_model.C_con *= value  # don't use .modify_node() here since this is about connections, not nodes.
    elif variable == 'EMI':
        for node in nodal_model.nodes:
            nodal_model.modify_node(node, epsilon_new=node.epsilon*value)
    elif variable == 'ABS':
        for node in nodal_model.nodes:
            nodal_model.modify_node(node, alpha_new=node.alpha*value)
    elif variable == 'RAD':
        if value.lower() == 'off':  # can only turn off radiation. Cannot turn it on, since it is unknown which nodes would be connected.
            nodal_model.rad *= 0.

    if recompute_orbit:
        orbital_model.compute()

        # Find which nodes receive external heat (orbital_model.surfaces cannot be used since they may not be
        # identical to the node names (e.g., surfaces = 'x+' and node name = 'PCB_x+A').
        outernodes = []
        for nod in nodal_model.nodes:
            if nod.outer:
                outernodes.append(nod)  # append the whole Node object

        if variable == 'ALT' or variable == 'DT':  # altitude change results in different time array.
            # Store P_int and re-apply it later, because it is erased by NodalModel.set_time.
            P_int_new = []
            for nod in nodal_model.nodes:
                if nod.P_int is None:
                    P_int_new.append(np.zeros(orbital_model.t.shape[0]))
                else:
                    # time array changes, so the best approximation is to take the average power.
                    P_int_new.append(np.mean(nod.P_int)*np.ones(orbital_model.t.shape[0]))
            nodal_model.set_time(t=orbital_model.t, erase=True)  # sets the correct time, but removes all time-dependent properties
            for num, nod in enumerate(nodal_model.nodes):
                nodal_model.modify_node(nod, P_int_new=P_int_new[num])

        for idx, outernode in enumerate(outernodes):
            direction = ''
            names_ = ['x+', 'x-', 'y+', 'y-', 'z+', 'z-']
            for name_ in names_:
                if name_ in outernode.name:  # Node name must include either of names_
                    direction = name_
            if direction == '':
                print(f"----------------ERROR----------------\n"
                      f"The sensitivity analysis could not be completed\n"
                      f"because outer node '{outernode.name}' does not have either of\n"
                      f"'x+', 'x-', 'y+', 'y-', 'z+', or 'z-' in their name.\n"
                      f"Ensure that all outer nodes contain a direction in their name.\n"
                      f"-------------------------------------\n")
            nodal_model.modify_node(outernode.name, q_ext_new=orbital_model.get_heat(direction))

    nodal_model.solve(solver=solver, printing=printing, limit_step=limit_step, interp_inputs=interp_inputs)
    return nodal_model


def run_all(dev1, dev2, nodal_model, orbital_model, overwrite=False, folder='SensitivityAnalysis',
            subfolders=(), printing=True, solver='Radau', limit_step=False, interp_inputs=False):
    """
    Runs all sensitivity analysis cases and writes them to pickle files.
    If overwrite is True, existing pickle files with the same name are overwritten.

    Subfolders can be an arbitrary number of sub-folders, such as:
    run_all(..., folder='folderA', subfolders=tuple('folderB', 'folderC'))
    --> C:/...your-directory.../folderA/folderB/folderC/file
    The file name is automatically generated.

    :param dev1: # +- small deviation (fraction) for ALT, CAP, IR, ALB, SOL, POW, CON, EMI, and ABS.
    :param dev2: # +- large deviation (fraction) for ALT, CAP, IR, ALB, SOL, POW, CON, EMI, and ABS.
    :param nodal_model: NodalModel to be subjected to the sensitivity analysis.
    :param orbital_model: OrbitalModel to be subjected to the sensitivity analysis.
    :param overwrite: Boolean to determine whether existing pickle files with the same name are to be overwritten.
    :param folder: Name (str) of the folder in which the files are to be exported.
    :param subfolders: Tuple/list of names (str) of any sub-folders which the file is in.
    :param printing: Boolean indicating whether progress should be printed in the command line.
    :param solver: Numerical integration method, must be either of: ['rk45', 'dop853', 'radau'].
                   Default is Radau.
    :param limit_step: Boolean used for scipy's variable time stepping, indicating whether the solver can use its
                       optimisation algorithms to skip time steps and reduce computational time (limit_step=False),
                       or whether a higher accuracy is desired and no time steps are allowed to be skipped
                       (limit_step=True). Default is False.
    :param interp_inputs: Boolean indicating whether the environmental inputs (q_pla, q_alb, q_s, P_int) should
                          be interpolated during scipy's variable time stepping. Improves accuracy, but also
                          increases computational time. Default is False.
    """
    print('Running Cases for Sensitivity Analysis...')
    if isinstance(subfolders, str):
        subfolders = (subfolders,)
    all_variables = ['ABS', 'ALB', 'ALT', 'CAP', 'CON', 'EMI', 'IR', 'POW', 'SOL']
    for casename in all_variables:
        cases = np.array([1.-dev2, 1.-dev1, 1., 1.+dev1, 1.+dev2])
        for case in cases:
            filename = f'{casename}_{case}'.replace('.', '_')
            folder_listdir = f'{folder}'
            for subfolder in subfolders:
                folder_listdir = os.path.join(folder_listdir, subfolder)
            if not os.path.exists(folder_listdir):
                os.makedirs(folder_listdir)
            if f'{filename}.pkl' not in os.listdir(folder_listdir) or overwrite:
                output = run_case(casename, case, nodal_model, orbital_model, printing, solver, limit_step,
                                  interp_inputs)
                with open(get_folder_file(folder, f'{filename}.pkl', subfolders), 'wb') as f:
                    pkl.dump(output, f)
    print('Finished Sensitivity Analysis.\n')


def run_variable(name, values, nodal_model, orbital_model, overwrite=False, folder='SensitivityAnalysis', subfolders=None,
                 printing=True, solver='Radau', limit_step=False, interp_inputs=False):
    """
    Runs arbitrary analysis cases for a given variable, using the other default values, and writes them to pickle files.
    The difference with the run_all function is that run_variable is not limited to two deviation percentages; any array
    with values can be used. Furthermore, it will result in different plots (plot_variable).
    If overwrite is True, existing pickle files with the same name are overwritten.

    The parameter 'value' is a list of the relative fractions with which the variable should be multiplied
    (except for DAY, BETA, DT, for which the values are the actual absolute values).

    Subfolders can be an arbitrary number of sub-folders, such as:
    run_variable(..., folder='folderA', subfolders=tuple('folderB', 'folderC'))
    --> C:/...your-directory.../folderA/folderB/folderC/file
    The file name is automatically generated.

    :param name: Name (str) of the variable. Must be one of: ['DAY', 'ALT', 'BETA', 'CAP', 'IR', 'ALB', 'SOL', 'POW',
                 'CON', 'EMI', 'ABS', 'ROT', 'DT', 'RAD'].
                 Internal radiation (RAD) is recommended to be used with its own function (plot_intrad).
    :param values: List/tuple/array of values. Is a multiplication factor if the variable is either of: [ALT, CAP, IR,
                   ALB, SOL, POW, CON, EMI, ABS, ROT], and the actual absolute value if the variable is either of: [DAY,
                   BETA, DT]. RAD would be either 'on' or 'off'.
    :param nodal_model: NodalModel to be subjected to the sensitivity analysis.
    :param orbital_model: OrbitalModel to be subjected to the sensitivity analysis.
    :param overwrite: Boolean to determine whether existing pickle files with the same name are to be overwritten.
    :param folder: Name (str) of the folder in which the files are to be exported.
    :param subfolders: Tuple/list of names (str) of any sub-folders which the file is in.
    :param printing: Boolean indicating whether progress should be printed in the command line.
    :param solver: Numerical integration method, must be either of: ['rk45', 'dop853', 'radau'].
                   Default is Radau.
    :param limit_step: Boolean used for scipy's variable time stepping, indicating whether the solver can use its
                       optimisation algorithms to skip time steps and reduce computational time (limit_step=False),
                       or whether a higher accuracy is desired and no time steps are allowed to be skipped
                       (limit_step=True). Default is False.
    :param interp_inputs: Boolean indicating whether the environmental inputs (q_pla, q_alb, q_s, P_int) should
                          be interpolated during scipy's variable time stepping. Improves accuracy, but also
                          increases computational time. Default is False.
    """
    print(f'Running Cases for {name} Sensitivity Analysis...')
    if subfolders is None:
        subfolders = (str(name),)
    if isinstance(subfolders, str):
        subfolders = (subfolders,)

    for value in values:
        filename = f'{name}_{value}'.replace('.', '_')
        folder_listdir = f'{folder}'
        for subfolder in subfolders:
            folder_listdir = os.path.join(folder_listdir, subfolder)
        if not os.path.exists(folder_listdir):
            os.makedirs(folder_listdir)
        if f'{filename}.pkl' not in os.listdir(folder_listdir) or overwrite:
            output = run_case(name, value, nodal_model, orbital_model, printing, solver, limit_step, interp_inputs)
            with open(get_folder_file(folder, f'{filename}.pkl', subfolders), 'wb') as f:
                pkl.dump(output, f)
    print(f'Finished {name} Sensitivity Analysis.\n')


def run_dt_rot(name, nodal_model, orbital_model, max_val=None, point_density=1., overwrite=False, folder='SensitivityAnalysis',
               subfolders=None, printing=True, solver='Radau', limit_step=False, interp_inputs=False):
    """
    Generates data points for a temperature versus angular velocity graph, then executes run_variable.
    Those data points (time steps or angular rates) are determined automatically, based on the expected time step needed
    to resolve a 90-degree rotation.

    For the angular rates analysis (name='ROT'), set those angular rates to (1., 1., 1.) so that the multiplication
    factor matches with the actual degrees per second.

    It is recommended to make extra sub-subfolders, such as: subfolders=('rot', 'dt10') or ('dt', 'rot-z').
    If overwrite is True, existing pickle files with the same name are overwritten.

    Subfolders can be an arbitrary number of sub-folders, such as:
    run_dt_rot(..., folder='folderA', subfolders=tuple('folderB', 'folderC'))
    --> C:/...your-directory.../folderA/folderB/folderC/file
    The file name is automatically generated.

    :param name: Name (str) of the variable. Must be either 'ROT' or 'DT'.
    :param nodal_model: NodalModel to be subjected to the sensitivity analysis.
    :param orbital_model: OrbitalModel to be subjected to the sensitivity analysis.
    :param max_val: Max value of the angular rate [deg/s] or time step [s] to be computed.
                    Default are 20 deg/s and 100 s.
    :param point_density: Multiplication factor for the number of points generated. Default of 1.0 means that the likely
                          most optimal number of points is used. Setting the value to 2.0 computes twice as many points,
                          or setting it to 0.5 halves the number of points, for example.
    :param overwrite: Boolean to determine whether existing pickle files with the same name are to be overwritten.
    :param folder: Name (str) of the folder in which the files are to be exported.
    :param subfolders: Tuple/list of names (str) of any sub-folders which the file is in.
    :param printing: Boolean indicating whether progress should be printed in the command line.
    :param solver: Numerical integration method, must be either of: ['rk45', 'dop853', 'radau'].
                   Default is Radau.
    :param limit_step: Boolean used for scipy's variable time stepping, indicating whether the solver can use its
                       optimisation algorithms to skip time steps and reduce computational time (limit_step=False),
                       or whether a higher accuracy is desired and no time steps are allowed to be skipped
                       (limit_step=True). Default is False.
    :param interp_inputs: Boolean indicating whether the environmental inputs (q_pla, q_alb, q_s, P_int) should
                          be interpolated during scipy's variable time stepping. Improves accuracy, but also
                          increases computational time. Default is False.
    """
    if name == 'ROT':
        rot_crit = 90./orbital_model.dt  # "critical" angular velocity [deg/s], 9 deg/s for dt=10s
        if max_val is None:
            max_val = 20.  # [deg/s]
        if max_val < rot_crit:
            print(f"---------------WARNING---------------\n"
                  f"The critical angular velocity ({rot_crit:.2f} deg/s) is not visible in the plot,\n"
                  f"since the maximum angular speed is smaller ({max_val:.2f} deg/s).\n"
                  f"Increase the maximum plotted value to see the full effects of the critical angular velocity.\n"
                  f"-------------------------------------\n")
        d_rot1 = 5./orbital_model.dt/point_density
        points_positive = np.arange(0., max_val+d_rot1, d_rot1)
        points_negative = -points_positive  # Also look at negative rotational values, see if it makes a difference.
        points = np.concatenate((points_positive, points_negative))
        points = np.sort(points)
        run_variable('ROT', points, nodal_model, orbital_model, overwrite=overwrite, folder=folder, subfolders=subfolders,
                     printing=printing, solver=solver, limit_step=limit_step, interp_inputs=interp_inputs)
    elif name == 'DT':
        dt_crit = 90./(np.max(np.array(orbital_model.angular_rates))*180/np.pi)
        if max_val is None:
            max_val = 100.  # [s] max time step to be computed
        if max_val < dt_crit:
            print(f"---------------WARNING---------------\n"
                  f"The critical time step ({dt_crit:.2f} s) is not visible in the plot,\n"
                  f"since the maximum time step is smaller ({max_val:.2f} s).\n"
                  f"Increase the maximum plotted value to see the full effects of the critical time step.\n"
                  f"-------------------------------------\n")
        rot1 = 5./(np.max(np.array(orbital_model.angular_rates))*180/np.pi)/point_density
        points = np.arange(rot1, max_val + rot1, rot1)
        run_variable('DT', points, nodal_model, orbital_model, overwrite=overwrite, folder=folder, subfolders=subfolders,
                     printing=printing, solver=solver, limit_step=limit_step, interp_inputs=interp_inputs)


def run_integrators(nodal_model, orbital_model, values=None, overwrite=False, folder='SensitivityAnalysis',
                    subfolders=('dt', 'integrators'), printing=True):
    """
    Run a number of cases with different integrators. Mainly used for thesis report.
    If overwrite is True, existing pickle files with the same name are overwritten.

    Subfolders can be an arbitrary number of sub-folders, such as:
    run_integrators(..., folder='folderA', subfolders=tuple('folderB', 'folderC'))
    --> C:/...your-directory.../folderA/folderB/folderC/file
    The file name is automatically generated.

    :param nodal_model: NodalModel to be subjected to the sensitivity analysis.
    :param orbital_model: OrbitalModel to be subjected to the sensitivity analysis.
    :param values: Time step values [s] at which to evaluate the models.
    :param overwrite: Boolean to determine whether existing pickle files with the same name are to be overwritten.
    :param folder: Name (str) of the folder in which the files are to be exported.
    :param subfolders: Tuple/list of names (str) of any sub-folders which the file is in.
    :param printing: Boolean indicating whether progress should be printed in the command line.
    """
    if values is None:
        values = 1/20*2**np.array(range(16, 0, -1))

    methods = ['radau', 'radau_limstep', 'radau_interp', 'radau_limstep_interp', 'rk45', 'dop853']
    subfolders_lst = []
    for i, m in enumerate(methods):
        f = ensure_list(subfolders)
        f.append(m)
        subfolders_lst.append(tuple(f))

    run_variable('DT', values=values, nodal_model=nodal_model, orbital_model=orbital_model, overwrite=overwrite, folder=folder,
                 subfolders=subfolders_lst[0], solver='Radau', limit_step=False, interp_inputs=False, printing=printing)
    run_variable('DT', values=values, nodal_model=nodal_model, orbital_model=orbital_model, overwrite=overwrite, folder=folder,
                 subfolders=subfolders_lst[1], solver='Radau', limit_step=True, interp_inputs=False, printing=printing)
    run_variable('DT', values=values, nodal_model=nodal_model, orbital_model=orbital_model, overwrite=overwrite, folder=folder,
                 subfolders=subfolders_lst[2], solver='Radau', limit_step=False, interp_inputs=True, printing=printing)
    run_variable('DT', values=values, nodal_model=nodal_model, orbital_model=orbital_model, overwrite=overwrite, folder=folder,
                 subfolders=subfolders_lst[3], solver='Radau', limit_step=True, interp_inputs=True, printing=printing)
    run_variable('DT', values=values, nodal_model=nodal_model, orbital_model=orbital_model, overwrite=overwrite, folder=folder,
                 subfolders=subfolders_lst[4], solver='rk45', limit_step=False, interp_inputs=False, printing=printing)
    run_variable('DT', values=values, nodal_model=nodal_model, orbital_model=orbital_model, overwrite=overwrite, folder=folder,
                 subfolders=subfolders_lst[5], solver='dop853', limit_step=False, interp_inputs=False, printing=printing)


def run_intrad(nodal_model=None, orbital_model=None, overwrite=False, folder='SensitivityAnalysis', subfolders=('rad',),
               printing=True, solver='Radau', limit_step=False, interp_inputs=False):
    """
    Runs the analysis cases for different internal radiation cases, using the other default values,
    and writes them to pickle files.
    If overwrite is True, existing pickle files with the same name are overwritten.

    Subfolders can be an arbitrary number of sub-folders, such as:
    run_intrad(..., folder='folderA', subfolders=tuple('folderB', 'folderC'))
    --> C:/...your-directory.../folderA/folderB/folderC/file
    The file name is automatically generated.

    :param nodal_model: NodalModel to be subjected to the sensitivity analysis.
    :param orbital_model: OrbitalModel to be subjected to the sensitivity analysis.
    :param overwrite: Boolean to determine whether existing pickle files with the same name are to be overwritten.
    :param folder: Name (str) of the folder in which the files are to be exported.
    :param subfolders: Tuple/list of names (str) of any sub-folders which the file is in.
    :param printing: Boolean indicating whether progress should be printed in the command line.
    :param solver: Numerical integration method, must be either of: ['rk45', 'dop853', 'radau'].
                   Default is Radau.
    :param limit_step: Boolean used for scipy's variable time stepping, indicating whether the solver can use its
                       optimisation algorithms to skip time steps and reduce computational time (limit_step=False),
                       or whether a higher accuracy is desired and no time steps are allowed to be skipped
                       (limit_step=True). Default is False.
    :param interp_inputs: Boolean indicating whether the environmental inputs (q_pla, q_alb, q_s, P_int) should
                          be interpolated during scipy's variable time stepping. Improves accuracy, but also
                          increases computational time. Default is False.
    """
    print('Running Cases for Internal Radiation Sensitivity Analysis...')
    if isinstance(subfolders, str):
        subfolders = (subfolders,)
    rad_cases = ['off', 'on']
    for rad in rad_cases:
        filename = f'RAD_{rad}'.replace('.', '_')
        folder_listdir = f'{folder}'
        for subfolder in subfolders:
            folder_listdir = os.path.join(folder_listdir, subfolder)
        if not os.path.exists(folder_listdir):
            os.makedirs(folder_listdir)
        if f'{filename}.pkl' not in os.listdir(folder_listdir) or overwrite:
            output = run_case('RAD', rad, nodal_model, orbital_model, printing, solver, limit_step, interp_inputs)
            with open(get_folder_file(folder, f'{filename}.pkl', subfolders), 'wb') as f:
                pkl.dump(output, f)
    print('Finished Internal Radiation Sensitivity Analysis.\n')


def plot_all(folder='SensitivityAnalysis', subfolders=()):
    """
    Plot all sensitivity analysis cases from the given (sub)folder.

    :param folder: Name (str) of the folder where the pickle files are to be read from.
    :param subfolders: Tuple/list of names (str) of any sub-folders which the file is in.
    """
    if isinstance(subfolders, str):
        subfolders = (subfolders,)
    filenames, results = read_folder(folder, subfolders)
    if results[0].celsius:
        temp_unit = r'$\degree$C'
    else:
        temp_unit = r'K'

    input_vals = [0.]*len(filenames)
    input_names = [0.]*len(filenames)
    for idx in range(len(filenames)):
        input_vals[idx] = float(str(filenames[idx][str(filenames[idx]).find('_')+1:-4]).replace('_', '.'))
        input_names[idx] = filenames[idx][:str(filenames[idx]).find('_')]
    input_names = list(np.unique(input_names))  # remove duplicates
    for i, name in enumerate(input_names):
        vals_ = input_vals[5*i:5*(i+1)]  # assuming each variable has 5 values: baseline, +-x% and +-y%
        results_ = results[5*i:5*(i+1)]  # assuming each variable has 5 values: baseline, +-x% and +-y%
        sorted_idx = np.argsort(vals_)
        vals_sorted = [vals_[i] for i in sorted_idx]
        results_sorted = [results_[i] for i in sorted_idx]
        input_vals[5*i:5*(i+1)] = vals_sorted
        results[5*i:5*(i+1)] = results_sorted

    temp_mean = np.array([])
    percent1 = (input_vals[3] - input_vals[2]) / input_vals[2] * 100.  # recompute the sensitivity %
    percent2 = (input_vals[4] - input_vals[2]) / input_vals[2] * 100.  # recompute the sensitivity %
    for result in results:
        temp_mean = np.append(temp_mean, np.mean(result.T))

    temp_mean_baseline = temp_mean[2::5]
    idx_pos = (np.argwhere(temp_mean[1::5] - temp_mean_baseline[0] >= 0)).flatten()  # where the parameter is increased
    idx_neg = (np.argwhere(temp_mean[1::5] - temp_mean_baseline[0] < 0)).flatten()  # where the parameter is decreased
    temp_mean_10min = temp_mean[1::5] - temp_mean_baseline[0]
    temp_mean_10plus = temp_mean[3::5] - temp_mean_baseline[0]
    temp_mean_20min = temp_mean[0::5] - temp_mean_baseline[0]
    temp_mean_20plus = temp_mean[4::5] - temp_mean_baseline[0]
    temp_mean_10min = np.abs(temp_mean_10min)
    temp_mean_20min = np.abs(temp_mean_20min)
    temp_mean_10plus = np.abs(temp_mean_10plus)
    temp_mean_20plus = np.abs(temp_mean_20plus)
    # In case the temperature decreases with increasing input parameter, swap some indices (is indicated in the figure)
    swap_10min = temp_mean_10min[idx_pos]
    swap_20min = temp_mean_20min[idx_pos]
    swap_10plus = temp_mean_10plus[idx_pos]
    swap_20plus = temp_mean_20plus[idx_pos]
    temp_mean_10min[idx_pos] = swap_10plus
    temp_mean_20min[idx_pos] = swap_20plus
    temp_mean_10plus[idx_pos] = swap_10min
    temp_mean_20plus[idx_pos] = swap_20min

    fig = plt.figure()
    ax = fig.add_subplot()
    if len(subfolders) != 0:
        ax.set_title(f'Average Temperature Change Due To\n{percent1:.0f}-{percent2:.0f}% Input Variations\n'+
                     r'$T_{avg_{baseline}}=$'+f'{temp_mean_baseline[0]:.1f}'+f'{temp_unit}'+f'\nfolder: {subfolders[-1]}')
    else:
        ax.set_title(f'Average Temperature Change Due To\n{percent1:.0f}-{percent2:.0f}% Input Variations\n'+
                     r'$T_{avg_{baseline}}=$'+f'{temp_mean_baseline[0]:.1f}'+f'[{temp_unit}]')
    ax.errorbar(x=input_names, y=temp_mean_baseline * 0., yerr=[temp_mean_10min, temp_mean_10plus], mec='k', mfc='k', fmt='o',
                linewidth=5, capsize=6, zorder=20, label=fr'$\pm${percent1:.0f}%', markersize=6)
    ax.errorbar(x=input_names, y=temp_mean_baseline * 0., yerr=[temp_mean_20min, temp_mean_20plus], mec='k', mfc='k', fmt='o',
                linewidth=2, capsize=6, zorder=10, label=fr'$\pm${percent2:.0f}%', markersize=6)
    plt.scatter(np.array(input_names)[idx_pos], -temp_mean_20min[idx_pos], marker=mkr.MarkerStyle(7), zorder=30, color='k', s=50)
    plt.scatter(np.array(input_names)[idx_neg], temp_mean_20plus[idx_neg], marker=mkr.MarkerStyle(6), zorder=30, color='k', s=50)
    ax.axhline(temp_mean_baseline[0] * 0., c='k', linestyle='dashed', label='baseline')
    ax.set_ylabel(r'Difference in $T_{avg}$'+'\n'+'compared to baseline '+f'[{temp_unit}]')
    ax.set_xlabel('Input parameter to be varied')
    ax.legend()
    ax.grid(zorder=0)
    fig.tight_layout()

    # Again, but instead of mean temperature, do temperature swing
    temp_minmax = np.array([])
    percent1 = (input_vals[3] - input_vals[2]) / input_vals[2] * 100.  # recompute the sensitivity %
    percent2 = (input_vals[4] - input_vals[2]) / input_vals[2] * 100.  # recompute the sensitivity %
    for result in results:
        temp_minmax = np.append(temp_minmax, np.abs(np.max(result.T)-np.min(result.T)))

    temp_minmax_baseline = temp_minmax[2::5]
    idx_pos = (np.argwhere(temp_minmax[1::5] - temp_minmax_baseline[0] >= 0)).flatten()  # where the parameter is increased
    idx_neg = (np.argwhere(temp_minmax[1::5] - temp_minmax_baseline[0] < 0)).flatten()  # where the parameter is decreased
    temp_minmax_10min = temp_minmax[1::5] - temp_minmax_baseline[0]
    temp_minmax_10plus = temp_minmax[3::5] - temp_minmax_baseline[0]
    temp_minmax_20min = temp_minmax[0::5] - temp_minmax_baseline[0]
    temp_minmax_20plus = temp_minmax[4::5] - temp_minmax_baseline[0]
    temp_minmax_10min = np.abs(temp_minmax_10min)
    temp_minmax_20min = np.abs(temp_minmax_20min)
    temp_minmax_10plus = np.abs(temp_minmax_10plus)
    temp_minmax_20plus = np.abs(temp_minmax_20plus)
    # In case the temperature decreases with increasing input parameter, swap some indices (is indicated in the figure)
    swap_10min = temp_minmax_10min[idx_pos]
    swap_20min = temp_minmax_20min[idx_pos]
    swap_10plus = temp_minmax_10plus[idx_pos]
    swap_20plus = temp_minmax_20plus[idx_pos]
    temp_minmax_10min[idx_pos] = swap_10plus
    temp_minmax_20min[idx_pos] = swap_20plus
    temp_minmax_10plus[idx_pos] = swap_10min
    temp_minmax_20plus[idx_pos] = swap_20min

    fig = plt.figure()
    ax = fig.add_subplot()
    if len(subfolders) != 0:
        ax.set_title(f'Temperature Swing Change Due To\n{percent1:.0f}-{percent2:.0f}% Input Variations\n'+
                     r'$\left(T_{max}-T_{min}\right)_{baseline}=$'+f'{temp_minmax_baseline[0]:.1f}' + f'{temp_unit}'+f'\nfolder: {subfolders[-1]}')
    else:
        ax.set_title(f'Temperature Swing Change Due To\n{percent1:.0f}-{percent2:.0f}% Input Variations\n'+
                     r'$\left(T_{max}-T_{min}\right)_{baseline}=$'+f'{temp_minmax_baseline[0]:.1f}' + f'{temp_unit}')
    ax.errorbar(x=input_names, y=temp_minmax_baseline * 0., yerr=[temp_minmax_10min, temp_minmax_10plus], mec='k', mfc='k', fmt='o',
                linewidth=5, capsize=6, zorder=20, label=fr'$\pm${percent1:.0f}%', markersize=6)
    ax.errorbar(x=input_names, y=temp_minmax_baseline * 0., yerr=[temp_minmax_20min, temp_minmax_20plus], mec='k', mfc='k', fmt='o',
                linewidth=2, capsize=6, zorder=10, label=fr'$\pm${percent2:.0f}%', markersize=6)
    plt.scatter(np.array(input_names)[idx_pos], -temp_minmax_20min[idx_pos], marker=mkr.MarkerStyle(7), zorder=30, color='k', s=50)
    plt.scatter(np.array(input_names)[idx_neg], temp_minmax_20plus[idx_neg], marker=mkr.MarkerStyle(6), zorder=30, color='k', s=50)
    ax.axhline(temp_minmax_baseline[0] * 0., c='k', linestyle='dashed', label='baseline')
    ax.set_ylabel(r'Difference in $\left(T_{max}-T_{min}\right)$'+'\n'+'compared to baseline '+f'[{temp_unit}]')
    ax.set_xlabel('Input parameter to be varied')
    ax.legend()
    ax.grid(zorder=0)
    fig.tight_layout()


def plot_variable(name, folder='SensitivityAnalysis', subfolders=(), whichnode=None):
    """
    Plot temperature data for varying values of the given variable name, from the given (sub)folder.
    The default node to be plotted for the transient (temperature-time and heat flux-time) graphs is the first node of
    the model (node 0). It is recommended that a node is chosen which is reasonably representative of the spacecraft
    temperature. For all other plots, the temperatures of all nodes are used to extract min/mean/max values.

    :param name: Name (str) of the variable. Must be one of: ['DAY', 'ALT', 'BETA', 'CAP', 'IR', 'ALB', 'SOL', 'POW',
                 'CON', 'EMI', 'ABS', 'ROT', 'DT', 'RAD'].
                 Internal radiation (RAD) is recommended to be used with its own function (plot_intrad).
    :param folder: Name (str) of the folder where the pickle files are to be read from.
    :param subfolders: Tuple/list of names (str) of any sub-folders which the file is in.
    :param whichnode: Node object, string, or integer indicating the node to be plotted for the transient graphs.
    """
    if isinstance(subfolders, str):
        subfolders = (subfolders,)
    filenames, results = read_folder(folder, subfolders)
    if name != 'RAD':
        vals = [0.]*len(filenames)
        for idx in range(len(filenames)):
            vals[idx] = float(str(filenames[idx][len(name)+1:-4]).replace('_', '.'))
        sorted_idx = np.argsort(vals)
        vals = [vals[i] for i in sorted_idx]
        results = [results[i] for i in sorted_idx]
    else:
        vals = ['']*len(filenames)
        for idx in range(len(filenames)):
            vals[idx] = str(filenames[idx][len(name) + 1:-4]).replace('_', '.')

    if results[0].celsius:
        temp_unit = r'$\degree$C'
    else:
        temp_unit = r'K'

    if whichnode is None:
        whichnode = 0
    nodename = ''
    if isinstance(whichnode, str):
        nodename = whichnode
        whichnode = np.argwhere(results[0].name == whichnode)[0, 0]
    elif isinstance(whichnode, Node):
        nodename = whichnode.name
        whichnode = np.argwhere(results[0].name == whichnode.name)[0, 0]
    elif isinstance(whichnode, int):
        nodename = results[0].name[whichnode]

    # Min/mean/max temperature plots
    fig1 = plt.figure()
    ax1 = fig1.add_subplot()
    abbrev = ['DAY', 'ALT', 'BETA', 'CAP', 'IR', 'ALB', 'SOL', 'POW', 'CON', 'EMI', 'ABS', 'ROT', 'DT', 'RAD']
    indx = np.argwhere(np.array(abbrev) == name)[0, 0]
    proper_names = ['Day', 'Altitude', 'Beta Angle', 'Heat Capacity', 'Earth IR', 'Albedo', 'Solar Flux',
                    'Internal Power', 'Conductance', 'Emissivity', 'Absorptivity', 'Angular Velocity',
                    'Time Step', 'Internal Radiation']
    symbols = [r'day', r'$h$', r'$\beta$', r'$C_{cap}$', r'$P_{pla}$', r'$P_{alb}$', r'$P_{s}$', r'$P_{int}$',
               r'$C_{con}$', r'$\varepsilon$', r'$\alpha$', r'$\omega$', r'$dt$', 'Rad']
    units = ['', 'm', r'$\degree$', 'J/K', 'W', 'W', 'W', 'W', 'W/K', '', '', r'$\degree$/s', 's', '']
    if len(subfolders) != 0:
        fig1.suptitle(f'Temperature Variance Per {proper_names[indx]}\nfolder: "{subfolders[-1]}"')
    else:
        fig1.suptitle(f'Temperature Variance Per {proper_names[indx]}')

    valsmean = np.zeros(len(results))
    valsmin = np.zeros(len(results))
    valsmax = np.zeros(len(results))
    for i, output in enumerate(results):
        maxTemp = np.max(output.T)
        minTemp = np.min(output.T)
        avgTemp = np.mean(output.T)
        valsmean[i] = avgTemp
        valsmin[i] = minTemp
        valsmax[i] = maxTemp

    ax1.plot(vals, valsmax, marker='^', c='k', label='max')
    ax1.plot(vals, valsmean, marker='o', c='k', label='mean')
    ax1.plot(vals, valsmin, marker='v', c='k', label='min')
    ax1.fill_between(vals, valsmin, valsmean, color='deepskyblue')
    ax1.fill_between(vals, valsmean, valsmax, color='tab:red')
    ax1.set_ylabel(r'Minimum/Mean/Maximum Temperature '+f'[{temp_unit}]')
    if name in ['ABS', 'ALB', 'ALT', 'CAP', 'CON', 'EMI', 'IR', 'POW', 'SOL', 'ROT']:
        ax1.set_xlabel(f'Multiplication Factor on {proper_names[indx]}')
    else:
        if units[indx] == '':
            ax1.set_xlabel(f'{proper_names[indx]}')
        else:
            ax1.set_xlabel(f'{proper_names[indx]} [{units[indx]}]')
    ax1.grid()
    ax1.legend()
    fig1.tight_layout()

    # Transient temperature plots
    figA = plt.figure()
    axA = figA.add_subplot()
    if len(subfolders) != 0:
        axA.set_title(r'Solar heat'+f'\nfolder: "{subfolders[-1]}"\nnode: "{nodename}"')
    else:
        axA.set_title(r'Solar heat'+f'\nnode: "{nodename}"')
    axA.set_ylabel(r'Solar heat [W]')
    figC = plt.figure()
    axC = figC.add_subplot()
    if len(subfolders) != 0:
        axC.set_title(r'Albedo heat'+f'\nfolder: "{subfolders[-1]}"\nnode: "{nodename}"')
    else:
        axC.set_title(r'Albedo heat'+f'\nnode: "{nodename}"')
    axC.set_ylabel(r'Albedo heat [W]')
    figD = plt.figure()
    axD = figD.add_subplot()
    if len(subfolders) != 0:
        axD.set_title(r'Earth IR heat'+f'\nfolder: "{subfolders[-1]}"\nnode: "{nodename}"')
    else:
        axD.set_title(r'Earth IR heat'+f'\nnode: "{nodename}"')
    axD.set_ylabel(r'Earth IR heat [W]')
    figB = plt.figure()
    axB = figB.add_subplot()
    if len(subfolders) != 0:
        axB.set_title(f'Temperature For Varying {proper_names[indx]}\nfolder: "{subfolders[-1]}"\nnode: "{nodename}"')
    else:
        axB.set_title(f'Temperature For Varying {proper_names[indx]}'+f'\nnode: "{nodename}"')

    axA.set_xlabel('Time [s]')
    axB.set_xlabel('Time [s]')
    axC.set_xlabel('Time [s]')
    axD.set_xlabel('Time [s]')

    axB.set_ylabel(f'Temperature [{temp_unit}]')
    colours_extended = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink',
                        'tab:gray', 'tab:olive', 'tab:cyan', 'blue', 'lime', 'red', 'mediumorchid', 'maroon',
                        'deeppink', 'silver', 'gold', 'cyan']
    axA.set_prop_cycle('color', colours_extended)
    axB.set_prop_cycle('color', colours_extended)
    axC.set_prop_cycle('color', colours_extended)
    axD.set_prop_cycle('color', colours_extended)
    for i, output in enumerate(results):
        if whichnode is None:
            whichnode = 0
        if isinstance(whichnode, str):
            whichnode = np.argwhere(output.name == whichnode)[0, 0]
        elif isinstance(whichnode, Node):
            whichnode = np.argwhere(output.name == whichnode.name)[0, 0]

        nod = whichnode  # Standard node to be shown
        if name in ['ABS', 'ALB', 'ALT', 'CAP', 'CON', 'EMI', 'IR', 'POW', 'SOL', 'ROT']:  # multiplication factor
            axA.plot(output.t, output.q_s[:, nod]*output.alpha[nod]*output.area[nod], label=f'{vals[i]}'+r'$\cdot$'+fr'{symbols[indx]}')
            axC.plot(output.t, output.q_alb[:, nod]*output.alpha[nod]*output.area[nod], label=f'{vals[i]}'+r'$\cdot$'+fr'{symbols[indx]}')
            axD.plot(output.t, output.q_pla[:, nod]*output.epsilon[nod]*output.area[nod], label=f'{vals[i]}'+r'$\cdot$'+fr'{symbols[indx]}')
            axB.plot(output.t, output.T[:, nod], label=f'{vals[i]}'+r'$\cdot$'+fr'{symbols[indx]}')
        else:  # actual value
            axA.plot(output.t, output.q_s[:, nod]*output.alpha[nod]*output.area[nod], label=fr'{symbols[indx]}=' + f'{vals[i]}'+f' {units[indx]}')
            axC.plot(output.t, output.q_alb[:, nod]*output.alpha[nod]*output.area[nod], label=fr'{symbols[indx]}=' + f'{vals[i]}'+f' {units[indx]}')
            axD.plot(output.t, output.q_pla[:, nod]*output.epsilon[nod]*output.area[nod], label=fr'{symbols[indx]}=' + f'{vals[i]}'+f' {units[indx]}')
            axB.plot(output.t, output.T[:, nod], label=fr'{symbols[indx]}=' + f'{vals[i]}'+f' {units[indx]}')

    axA.grid()
    axA.legend()
    axB.grid()
    axB.legend()
    axC.grid()
    axC.legend()
    axD.grid()
    axD.legend()
    figA.tight_layout()
    figB.tight_layout()
    figC.tight_layout()
    figD.tight_layout()


def plot_dt_rot(name, folder='SensitivityAnalysis', subfolders=()):
    """
    Plot temperature data for varying values of the given variable name, from the given (sub)folder.

    :param name: Name (str) of the variable. Must be either 'ROT' or 'DT'.
    :param folder: Name (str) of the folder where the pickle files are to be read from.
    :param subfolders: Tuple/list of names (str) of any sub-folders which the file is in.
    """
    if isinstance(subfolders, str):
        subfolders = (subfolders,)
    filenames, results = read_folder(folder, subfolders)
    vals = [0.]*len(filenames)
    for idx in range(len(filenames)):
        vals[idx] = float(str(filenames[idx][len(name)+1:-4]).replace('_', '.'))
    sorted_idx = np.argsort(vals)
    vals = [vals[i] for i in sorted_idx]
    results = [results[i] for i in sorted_idx]

    if results[0].celsius:
        temp_unit = r'$\degree$C'
    else:
        temp_unit = r'K'

    # Min/mean/max temperature plots
    fig1 = plt.figure()
    ax1 = fig1.add_subplot()
    abbrev = ['DAY', 'ALT', 'BETA', 'CAP', 'IR', 'ALB', 'SOL', 'POW', 'CON', 'EMI', 'ABS', 'ROT', 'DT']
    indx = np.argwhere(np.array(abbrev) == name)[0, 0]
    proper_names = ['Day', 'Altitude', 'Beta Angle', 'Heat Capacity', 'Earth IR', 'Albedo', 'Solar Flux',
                    'Internal Power', 'Conductance', 'Emissivity', 'Absorptivity', 'Angular Velocity',
                    'Time Step']
    units = ['', 'm', r'$\degree$', 'J/K', 'W', 'W', 'W', 'W', 'W/K', '', '', r'$\degree$/s', 's']
    if len(subfolders) != 0:
        fig1.suptitle(f'Temperature Variance Per {proper_names[indx]}\nfolder: "{subfolders[-1]}"')
    else:
        fig1.suptitle(f'Temperature Variance Per {proper_names[indx]}')

    valsmean = np.zeros(len(results))
    valsmin = np.zeros(len(results))
    valsmax = np.zeros(len(results))
    for i, output in enumerate(results):
        maxTemp = np.max(output.T)
        minTemp = np.min(output.T)
        avgTemp = np.mean(output.T)
        valsmean[i] = avgTemp
        valsmin[i] = minTemp
        valsmax[i] = maxTemp

    ax1.plot(vals, valsmax, marker='^', c='k', label='max')
    ax1.plot(vals, valsmean, marker='o', c='k', label='mean')
    ax1.plot(vals, valsmin, marker='v', c='k', label='min')

    ax1.fill_between(vals, valsmin, valsmean, color='deepskyblue')
    ax1.fill_between(vals, valsmean, valsmax, color='tab:red')
    ax1.set_ylabel(f'Minimum/Mean/Maximum Temperature [{temp_unit}]')
    if name == 'DT':
        ax1.set_xlabel(f'{proper_names[indx]} [{units[indx]}]')
    else:  # name == 'ROT'
        ax1.set_xlabel(f'Multiplication Factor on {proper_names[indx]}\n' +
                       r'(= actual deg/s if $\omega_{default}$=1 deg/s)')
    ax1.grid()
    ax1.legend()
    fig1.tight_layout()


def plot_integrators(benchmark_filename, benchmark_folder, plot_extra=False, folder='SensitivityAnalysis',
                     subfolders=('dt', 'integrators')):
    """
    Plot convergence data for different time steps. Mainly used for the thesis report.

    :param benchmark_filename: Name of the file which is considered as the benchmark computation.
    :param benchmark_folder: Name of the folder in which the benchmark is located (is assumed to be in the same
                             directory as the subfolders parameter).
    :param plot_extra: Plot extra figures showing the transient temperatures for each time step. Each time step is shown
                       in a separate figure (many figures will appear).
    :param folder: Name (str) of the folder where the pickle files are to be read from.
    :param subfolders: Tuple/list of names (str) of any sub-folders which the file is in.
    """
    name = 'DT'
    if isinstance(subfolders, str):
        subfolders = (subfolders,)

    bm_folder = ensure_list(subfolders)
    bm_folder.append(benchmark_folder)
    bm_folder = tuple(bm_folder)
    with open(get_folder_file('SensitivityAnalysis', benchmark_filename, subfolders=bm_folder), 'rb') as f:
        benchmark_output = pkl.load(f)
        benchmark_val = float(str(benchmark_filename[len(name)+1:-4]).replace('_', '.'))

    if benchmark_output.celsius:
        temp_unit = r'$\degree$C'
    else:
        temp_unit = r'K'

    folder_listdir = f'{folder}'
    for subfolder in subfolders:
        folder_listdir = os.path.join(folder_listdir, subfolder)
    solvers = os.listdir(folder_listdir)

    fig1 = plt.figure()
    ax1 = fig1.add_subplot()
    fig1.suptitle(f'Temperature Variance Per Time Step\nCompared to Benchmark')
    ax1.loglog()
    ax1.set_xlabel(f'Time Step [s]')
    ax1.set_ylabel(f'RMS Error in Temperature [{temp_unit}]')
    ax1.grid()

    fig_benchmark = plt.figure()
    ax_benchmark = fig_benchmark.add_subplot()
    fig_benchmark.suptitle(f'Temperature Variance Compared To\nPrevious Time Step\n(Benchmark: {benchmark_folder})')
    ax_benchmark.loglog()
    ax_benchmark.set_xlabel(f'Time Step [s]')
    ax_benchmark.set_ylabel(f'RMS Error in Temperature [{temp_unit}]')
    ax_benchmark.grid()
    markers = ['o', '^', 'v', 's', '<', '>', 'x', '+']
    for j, solver in enumerate(solvers):  # Each solver has its own folder
        filenames, results = read_folder(folder_listdir, (solver,))
        vals = [0.] * len(filenames)
        for idx in range(len(filenames)):
            vals[idx] = float(str(filenames[idx][len(name) + 1:-4]).replace('_', '.'))
        sorted_idx = np.argsort(vals)
        vals = [vals[i] for i in sorted_idx]
        results = [results[i] for i in sorted_idx]
        filenames = [filenames[i] for i in sorted_idx]

        if solver == benchmark_folder:  # compare values of the benchmark to its own previous time step (relative)
            rms_errs = np.zeros(len(vals)-1)
            for i in range(len(vals)-1):  # first cases in the list are the small time steps
                T_smalldt = results[i].T[:, 0]
                t_smalldt = results[i].t
                T_largedt = results[i+1].T[:, 0]
                t_largedt = results[i+1].t
                rms_errs[i] = calc_rmse(T_largedt, T_smalldt, round(vals[i+1]/vals[i]))
                if plot_extra:
                    plt.figure()
                    plt.title(f'{solver}, file: "{filenames[i][:-4]}"')
                    plt.plot(t_smalldt, T_smalldt, label='small')
                    plt.plot(t_largedt, T_largedt, label='large')
                    plt.legend()
                    plt.xlabel('Time [s]')
                    plt.ylabel(f'Temperature [{temp_unit}]')
                    plt.grid()
            ax_benchmark.plot(vals[1:], rms_errs, marker='o', c='k')
        else:  # compare values of the solver directly to the smallest benchmark (absolute)
            T_smalldt = benchmark_output.T[:, 0]
            t_smalldt = benchmark_output.t
            rms_errs = np.zeros(len(vals))
            for i in range(len(vals)):
                T_largedt = results[i].T[:, 0]
                t_largedt = results[i].t
                rms_errs[i] = calc_rmse(T_largedt, T_smalldt, round(vals[i]/benchmark_val))
                if plot_extra:
                    plt.figure()
                    plt.title(f'{solver}, file: "{filenames[i][:-4]}"')
                    plt.plot(t_smalldt, T_smalldt, label='small')
                    plt.plot(t_largedt, T_largedt, label='large')
                    plt.legend()
                    plt.xlabel('Time [s]')
                    plt.ylabel(f'Temperature [{temp_unit}]')
                    plt.grid()
            ax1.plot(vals, rms_errs, marker=markers[j], label=f'{solver}')

    ax1.legend()
    ax1.invert_xaxis()
    fig1.tight_layout()
    ax_benchmark.invert_xaxis()
    fig_benchmark.tight_layout()


def plot_intrad(whichnodes=None, folder='SensitivityAnalysis', subfolders=()):
    """
    Plot temperature data for varying internal radiation conditions from the given (sub)folder.

    :param whichnodes: String or integer (not Node) indicating which nodes are to be shown in all plots. This
                       also means that only connections between the selected nodes are shown.
    :param folder: Name (str) of the folder where the pickle files are to be read from.
    :param subfolders: Tuple/list of names (str) of any sub-folders which the file is in.
    """
    if isinstance(subfolders, str):
        subfolders = (subfolders,)
    filenames, results = read_folder(folder, subfolders)
    rad_vals = ['a']*len(filenames)
    linestyles_ = ['solid']*len(filenames)
    for idx in range(len(filenames)):
        rad_vals[idx] = str(filenames[idx][4:-4])  # Remove the .pkl extension
        if rad_vals[idx].lower() == 'off':
            linestyles_[idx] = 'dashed'

    if results[0].celsius:
        temp_unit = r'$\degree$C'
    else:
        temp_unit = r'K'

    if whichnodes is None:
        whichnodes = range(results[0].n)
    else:
        whichnodes = ensure_list(whichnodes)
        for x in range(len(whichnodes)):
            if isinstance(whichnodes[x], str):
                whichnodes[x] = np.argwhere(results[0].name == whichnodes[x])[0, 0]
            elif not isinstance(whichnodes[x], int):
                print(f"---------------WARNING---------------\n"
                      f"Selected nodes to plot were given in the wrong format.\n"
                      f"All nodes are shown.\n"
                      f"Enter a list with either the node number(s), name(s), or Node object(s).\n"
                      f"-------------------------------------\n")
                whichnodes = range(results[0].n)

    # Transient temperature plots
    figB = plt.figure()
    axB = figB.add_subplot()
    if len(subfolders) != 0:
        axB.set_title(f'Temperature For Varying Internal Radiation\nfolder: "{subfolders[-1]}"')
    else:
        axB.set_title(f'Temperature For Varying Internal Radiation')
    axB.set_xlabel('Time [s]')
    axB.set_ylabel(f'Temperature [{temp_unit}]')
    colours_extended = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink',
                        'tab:gray', 'tab:olive', 'tab:cyan', 'blue', 'lime', 'red', 'mediumorchid', 'maroon',
                        'deeppink', 'silver', 'gold', 'cyan']
    axB.set_prop_cycle('color', colours_extended)
    for i, output in enumerate(results):
        for nod in whichnodes:
            axB.plot(output.t, output.T[:, nod], label=r'rad=' + f'{rad_vals[i]}, node {nod}, "{results[0].name[nod]}"',
                     linestyle=linestyles_[i])
    axB.grid()
    axB.legend()
    figB.tight_layout()

    # Temperature difference plots
    figC = plt.figure()
    axC = figC.add_subplot()
    if len(subfolders) != 0:
        axC.set_title(f'Temperature Difference Between Radiative\nand Non-Radiative Simulation\nfolder: "{subfolders[-1]}"')
    else:
        axC.set_title(f'Temperature Difference Between Radiative\nand Non-Radiative Simulation')
    axC.set_xlabel('Time [s]')
    axC.set_ylabel(f'Temperature Difference\n(Rad. "on" minus "off") [{temp_unit}]')
    colours_extended = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink',
                        'tab:gray', 'tab:olive', 'tab:cyan', 'blue', 'lime', 'red', 'mediumorchid', 'maroon',
                        'deeppink', 'silver', 'gold', 'cyan']
    axC.set_prop_cycle('color', colours_extended)
    # for i, output in enumerate(results):
    for nod in whichnodes:  # off comes first, then on
        axC.plot(results[0].t, results[1].T[:, nod]-results[0].T[:, nod], label=r'rad=' + f'node {nod}, "{results[0].name[nod]}"')
    axC.grid()
    axC.legend()
    figC.tight_layout()


# --------------------- Examples with FUNcube model ------------------------
# run_all(0.1, 0.2, nodal_model=FUNcubeModel, orbital_model=FUNcubeOrbit, overwrite=False, subfolders=('FUNcube1', 'FUNsensitivity'))
# run_intrad(nodal_model=FUNcubeModel, orbital_model=FUNcubeOrbit, overwrite=False, subfolders=('FUNcube1', 'FUNintrad'))
# run_variable('ABS', values=[0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0], nodal_model=FUNcubeModel, orbital_model=FUNcubeOrbit,
#              overwrite=False, subfolders=('FUNcube1', 'FUNabs'))
# run_variable('BETA', values=[0., 10., 20., 30., 40., 50., 60., 70., 80., 90.], nodal_model=FUNcubeModel, orbital_model=FUNcubeOrbit,
#              overwrite=False, subfolders=('FUNcube1', 'FUNbeta'))
# run_variable('CAP', values=[0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0], nodal_model=FUNcubeModel, orbital_model=FUNcubeOrbit, overwrite=False,
#              subfolders=('FUNcube1', 'FUNcap'))
# run_variable('DAY', values=[1, 50, 100, 150, 200, 250, 300, 350], nodal_model=FUNcubeModel, orbital_model=FUNcubeOrbit, overwrite=False,
#              subfolders=('FUNcube1', 'FUNday'))
# run_variable('ALT', values=[0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0], nodal_model=FUNcubeModel, orbital_model=FUNcubeOrbit, overwrite=False,
#              subfolders=('FUNcube1', 'FUNalt'))
# run_variable('IR', values=[0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0], nodal_model=FUNcubeModel, orbital_model=FUNcubeOrbit, overwrite=False,
#              subfolders=('FUNcube1', 'FUNir'))
# run_variable('ALB', values=[0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0], nodal_model=FUNcubeModel, orbital_model=FUNcubeOrbit, overwrite=False,
#              subfolders=('FUNcube1', 'FUNalb'))
# run_variable('SOL', values=[0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0], nodal_model=FUNcubeModel, orbital_model=FUNcubeOrbit, overwrite=False,
#              subfolders=('FUNcube1', 'FUNsol'))
# run_variable('POW', values=[0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0], nodal_model=FUNcubeModel, orbital_model=FUNcubeOrbit, overwrite=False,
#              subfolders=('FUNcube1', 'FUNpow'))
# run_variable('CON', values=[0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0], nodal_model=FUNcubeModel, orbital_model=FUNcubeOrbit, overwrite=False,
#              subfolders=('FUNcube1', 'FUNcon'))
# run_variable('EMI', values=[0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0], nodal_model=FUNcubeModel, orbital_model=FUNcubeOrbit, overwrite=False,
#              subfolders=('FUNcube1', 'FUNemi'))

# plot_all(subfolders=('FUNcube1', 'FUNsensitivity'))
# plot_intrad(subfolders=('FUNcube1', 'FUNintrad'), whichnodes=['z-A', 'PCB2bat'])
# plot_variable('ABS', subfolders=('FUNcube1', 'FUNabs'), whichnode='PCB2bat')
# plot_variable('BETA', subfolders=('FUNcube1', 'FUNbeta'), whichnode='PCB2bat')
# plot_variable('CAP', subfolders=('FUNcube1', 'FUNcap'), whichnode='PCB2bat')
# plot_variable('DAY', subfolders=('FUNcube1', 'FUNday'), whichnode='PCB2bat')
# plot_variable('ALT', subfolders=('FUNcube1', 'FUNalt'), whichnode='PCB2bat')
# plot_variable('IR', subfolders=('FUNcube1', 'FUNir'), whichnode='PCB2bat')
# plot_variable('ALB', subfolders=('FUNcube1', 'FUNalb'), whichnode='PCB2bat')
# plot_variable('SOL', subfolders=('FUNcube1', 'FUNsol'), whichnode='PCB2bat')
# plot_variable('POW', subfolders=('FUNcube1', 'FUNpow'), whichnode='PCB2bat')
# plot_variable('CON', subfolders=('FUNcube1', 'FUNcon'), whichnode='PCB2bat')
# plot_variable('EMI', subfolders=('FUNcube1', 'FUNemi'), whichnode='PCB2bat')

plt.show()
