# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 14:36:24 2024
Author: Frank Meijering (Delft University of Technology)

ThermalBudget.py is the backbone of this thermal simulation tool. The three major classes (Node, NodalModel, and
OrbitalModel) are defined here. It is recommended to import those classes in a separate file, and build your models
there. Also import the show_available_materials() function and run it at the start of your code (not obligatory, but
useful); it shows a summary of available materials from the Materials.csv file.
"""


import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from Constants import const_lst
from scipy.integrate import solve_ivp
from scipy.optimize import root
from scipy.spatial.transform import Rotation
import time
from EnvironmentRadiation import heat_received, t_to_theta, theta_to_t, beta_angle, tau_phi, cartesian_to_spherical, \
    spherical_to_cartesian
from Materials import Material, Coating, Contact, mat_lst, coat_lst, con_lst
import numbers


def dT_dt(t, T_prev, const, n, C_cap, C_con, q_pla, q_alb, q_s, P_int, outer, alpha, epsilon, area, celsius, rad,
          t_array, cnt, printing=True, interp_inputs=False):
    """
    Calculates the derivative of the temperature of all nodes with respect to time, to be integrated afterward.

    :param t: Time instance [s], is unused but needed for scipy.integrate.solve_ivp.
    :param T_prev: Temperatures of the previous time step i-1 in [K] or [deg C].
    :param const: Dictionary of universal constants or Earth properties.
    :param n: Total number of nodes.
    :param C_cap: Heat capacities of the nodes [J/K].
    :param C_con: Conductance between nodes [W/K].
    :param q_pla: Earth IR heat flux [W/m^2]. These powers must stay separated since they have different spectra.
    :param q_alb: Albedo heat flux [W/m^2].
    :param q_s: Solar heat flux [W/m^2].
    :param P_int: Internal power [W] on each node (can be internal, external, or both).
    :param outer: Boolean array, whether the node radiates freely to space or not.
    :param alpha: Absorptivities in the solar spectrum of all nodes.
    :param epsilon: Emissivities in the IR spectrum of all nodes.
    :param area: Surface area [m^2] of all nodes.
    :param celsius: Boolean, whether the units are in Celsius (True) or Kelvin (False).
    :param rad: Radiative exchange factors between nodes. Same matrix shape as C_con, also symmetric.
    :param t_array: Entire time array (numpy array) of the simulation, used to track progress within scipy.solve_ivp.
    :param cnt: List with a zero value (as: cnt = [0]), used for printing progress in the console.
    :param printing: Boolean indicating whether the progress should be printed in the console.
    :param interp_inputs: Boolean indicating whether the environmental inputs (q_pla, q_alb, q_s, P_int) should
                          be interpolated during scipy's variable time stepping. Improves accuracy, but also
                          increases computational time. Default is False.
    :return: Derivative of temperature with respect to time in [K/s].
    """
    idx = np.argwhere(t >= t_array)[-1, 0]  # Take the closest time index (needed for variable timestep in solve_ivp)

    temporary = np.tile(T_prev, (n, 1))  # temporary variable to compute all temperature differences between all nodes
    dT_spatial = temporary-temporary.T  # all temperature differences between all nodes
    temporary2 = np.tile(T_prev+const['kelvin']*celsius, (n, 1))**4  # temporary variable to compute the (Ti^4-Tj^4) component for radiation.
    dT4_spatial = temporary2-temporary2.T  # differences between all T^4 of the nodes.

    # Progress bar
    progress = idx / (t_array.shape[0] - 1) * 100
    if printing and isinstance(cnt, list):
        if idx/t_array.shape[0]*100-cnt[-1] > 10:
            print(f'{progress:.0f}%')
            cnt.append(cnt[-1]+10)
            cnt.pop(0)

    if interp_inputs:
        q_pla_interp = np.zeros(n)
        q_alb_interp = np.zeros(n)
        q_s_interp = np.zeros(n)
        P_int_interp = np.zeros(n)
        for indx in range(n):
            q_pla_interp[indx] = np.interp(t, t_array, q_pla[:, indx])
            q_alb_interp[indx] = np.interp(t, t_array, q_alb[:, indx])
            q_s_interp[indx] = np.interp(t, t_array, q_s[:, indx])
            P_int_interp[indx] = np.interp(t, t_array, P_int[:, indx])
        return 1/C_cap*(P_int_interp+area*(epsilon*q_pla_interp+alpha*(q_alb_interp+q_s_interp))+np.sum(dT_spatial*C_con, axis=1)-
                        outer*area*epsilon*const['stefan']*((T_prev+const['kelvin']*celsius)**4-const['T_space']**4)+
                        const['stefan']*np.sum(dT4_spatial*rad, axis=1))  # internal, external, conduction, radiation to space, inner rad.
    else:
        return 1/C_cap*(P_int[idx]+area*(epsilon*q_pla[idx]+alpha*(q_alb[idx]+q_s[idx]))+np.sum(dT_spatial*C_con, axis=1)-
                        outer*area*epsilon*const['stefan']*((T_prev+const['kelvin']*celsius)**4-const['T_space']**4)+
                        const['stefan']*np.sum(dT4_spatial*rad, axis=1))  # internal, external, conduction, radiation to space, inner rad.


def view_factor_perp(a2, b, c1):
    """
    Computes the view factor between two perpendicular plates with a common edge (b).

    :param a2: Width [m] of the receiving plate.
    :param b: Length [m] of the common edge of both plates (must be equal).
    :param c1: Width [m] of the emitting plate.
    :return: View factor [-] from plate 1 to 2.
    """
    N = a2/b
    L = c1/b
    return 1/(np.pi*L)*(L*np.arctan(1/L)+N*np.arctan(1/N)-np.sqrt(N**2+L**2)*np.arctan(1/np.sqrt(N**2+L**2))+
                        1/4*np.log(((1+N**2)*(1+L**2)/(1+N**2+L**2))*(L**2*(1+N**2+L**2)/((1+L**2)*(N**2+L**2)))**(L**2)
                                   *(N**2*(1+N**2+L**2)/((1+N**2)*(N**2+L**2)))**(N**2)))


def view_factor_par(a_width, b_depth, c_dist):
    """
    Computes the view factor between two parallel plates with the same size, exactly placed above/under each other.
    Parameters a_width and b_depth are interchangeable due to the symmetry in the formula.

    :param a_width: Width [m] of both plates (must be equal).
    :param b_depth: Depth [m] of both plates (must be equal).
    :param c_dist: Distance [m] between both plates.
    :return: View factor [-] between the two plates (F12 = F21).
    """
    X = a_width/c_dist
    Y = b_depth/c_dist
    return 2/(np.pi*X*Y)*(np.log(np.sqrt((1+X**2)*(1+Y**2)/(1+X**2+Y**2)))+
                          X*np.sqrt(1+Y**2)*np.arctan(X/np.sqrt(1+Y**2))+
                          Y*np.sqrt(1+X**2)*np.arctan(Y/np.sqrt(1+X**2))-
                          X*np.arctan(X)-Y*np.arctan(Y))


def plot_plate(origin, size, ax, n, name):
    """
    Plot the outer lines of a flat, rectangular plate, as well as the origin.

    :param origin: (x, y, z) position of the centre of the plate.
    :param size: (x, y, z) dimensions of the plate. The size in the normal direction of the surface must be set to zero,
                 and the other two dimensions (width and height) must be nonzero.
    :param ax: Matplotlib axis object (mpl_toolkits.mplot3d.axes3d.Axes3D) which can be defined with:
               matplotlib.pyplot.figure().add_subplot(projection='3d').
    :param n: Node number of the drawn plate.
    :param name: Name (string) of the drawn plate.
    """
    if size[0] == 0:  # surface is in the Y-Z plane, pointing towards X.
        points = np.array([[origin[0], origin[1]-size[1]/2, origin[2]-size[2]/2],
                           [origin[0], origin[1]+size[1]/2, origin[2]-size[2]/2],
                           [origin[0], origin[1]+size[1]/2, origin[2]+size[2]/2],
                           [origin[0], origin[1]-size[1]/2, origin[2]+size[2]/2]])
    elif size[1] == 0:  # surface is in the X-Z plane, pointing towards Y.
        points = np.array([[origin[0]-size[0]/2, origin[1], origin[2]-size[2]/2],
                           [origin[0]+size[0]/2, origin[1], origin[2]-size[2]/2],
                           [origin[0]+size[0]/2, origin[1], origin[2]+size[2]/2],
                           [origin[0]-size[0]/2, origin[1], origin[2]+size[2]/2]])
    else:  # surface is in the X-Y plane, pointing towards Z.
        points = np.array([[origin[0]-size[0]/2, origin[1]-size[1]/2, origin[2]],
                           [origin[0]+size[0]/2, origin[1]-size[1]/2, origin[2]],
                           [origin[0]+size[0]/2, origin[1]+size[1]/2, origin[2]],
                           [origin[0]-size[0]/2, origin[1]+size[1]/2, origin[2]]])
    ax.plot([points[0, 0], points[1, 0]], [points[0, 1], points[1, 1]], [points[0, 2], points[1, 2]], c='k')
    ax.plot([points[1, 0], points[2, 0]], [points[1, 1], points[2, 1]], [points[1, 2], points[2, 2]], c='k')
    ax.plot([points[2, 0], points[3, 0]], [points[2, 1], points[3, 1]], [points[2, 2], points[3, 2]], c='k')
    ax.plot([points[3, 0], points[0, 0]], [points[3, 1], points[0, 1]], [points[3, 2], points[0, 2]], c='k')
    ax.scatter(origin[0], origin[1], origin[2], color='grey', s=30)
    ax.text(origin[0], origin[1], origin[2], f'{n}, "{name}"', size=11)


def plot_point(origin, ax, n, name):
    """
    Plot the origin of a node.

    :param origin: (x, y, z) position of the centre of the plate.
    :param ax: Matplotlib axis object (mpl_toolkits.mplot3d.axes3d.Axes3D) which can be defined with:
               matplotlib.pyplot.figure().add_subplot(projection='3d').
    :param n: Node number of the drawn plate.
    :param name: Name (string) of the drawn plate.
    """
    ax.scatter(origin[0], origin[1], origin[2], color='grey', s=30)
    ax.text(origin[0], origin[1], origin[2], f'{n}, "{name}"', size=11)


def plot_connections(point1, point2, axis, radiative=False):
    """
    Plot a line representing a conductive or radiative link between two nodes.

    :param point1: (x, y, z) position of the first node.
    :param point2: (x, y, z) position of the second node.
    :param axis: Matplotlib axis object (mpl_toolkits.mplot3d.axes3d.Axes3D) which can be defined with:
                 matplotlib.pyplot.figure().add_subplot(projection='3d').
    :param radiative: Boolean indicating whether it is a radiative (True) or conductive (False) connection.
    """
    if radiative:
        colour = 'limegreen'
        lin = (0, (5, 5))
        zord = 2
    else:
        colour = 'r'
        lin = 'solid'
        zord = 1
    axis.plot([point1[0], point2[0]], [point1[1], point2[1]], [point1[2], point2[2]], c=colour, linestyle=lin,
              zorder=zord)


def plot3dline(point1, point2, axis, colour='k', lbl='', style='solid', ordr=None):
    """
    Converts two points into a line and plots it on the given axis.

    :param colour: Colour of the desired plot.
    :param point1: Starting coordinate.
    :param point2: Ending coordinate.
    :param axis: Figure axis to be plotted on.
    :param lbl: Label of the axis.
    :param style: Linestyle.
    :param ordr: Zorder of the line (how far it is above or below others).
    """
    if ordr is not None:
        if lbl == '':
            axis.plot([point1[0], point2[0]], [point1[1], point2[1]], [point1[2], point2[2]], c=colour, linestyle=style,
                      zorder=ordr)
        else:
            axis.plot([point1[0], point2[0]], [point1[1], point2[1]], [point1[2], point2[2]], c=colour, label=lbl,
                      linestyle=style, zorder=ordr)
    else:
        if lbl == '':
            axis.plot([point1[0], point2[0]], [point1[1], point2[1]], [point1[2], point2[2]], c=colour, linestyle=style)
        else:
            axis.plot([point1[0], point2[0]], [point1[1], point2[1]], [point1[2], point2[2]], c=colour, label=lbl,
                      linestyle=style)


def plot3dframe(frame, axis, colour1='k', colour2='k', colour3='k', lbl1='', lbl2='', lbl3='', style='solid', ordr=None):
    """
    Plots a reference frame with three axes in a 3D plot.

    :param frame: 3x3 Numpy array containing the three unit vectors in the desired directions.
    :param axis: Matplotlib axis to be plotted on.
    :param colour1: Colour of the X-axis to be plotted.
    :param colour2: Colour of the Y-axis to be plotted.
    :param colour3: Colour of the Y-axis to be plotted.
    :param lbl1: Label of the X-axis to be shown in the legend.
    :param lbl2: Label of the Y-axis to be shown in the legend.
    :param lbl3: Label of the Z-axis to be shown in the legend.
    :param style: Linestyle.
    :param ordr: Zorder of the line (how far it is above or below others).
    """
    origin = np.array([0, 0, 0])
    for i in range(3):
        if i == 0:
            plot3dline(origin, frame[i], axis, colour1, lbl1, style, ordr)
        elif i == 1:
            plot3dline(origin, frame[i], axis, colour2, lbl2, style, ordr)
        else:
            plot3dline(origin, frame[i], axis, colour3, lbl3, style, ordr)


def ensure_list(x, keep_entity=False):
    """
    If the input is a list, it will return itself; if the input is not a list, it will return the input as a list.
    NOTE: be careful when using tuples and numpy arrays: when keep_entity is False, they will be transformed to a list
    type such as [x, y, z]; when keep_entity is True, they will be put into a list such as [(x, y, z)].

    :param x: Arbitrary input.
    :param keep_entity: Boolean indicating whether tuples/numpy arrays are transformed to a list, or put into a list.
    :return: Input x inside a Python list.
    """
    if isinstance(x, list):
        return x
    elif isinstance(x, (tuple, np.ndarray)) and not keep_entity:
        return list(x)
    else:
        return [x]


def nonzero(x):
    """
    Checks whether input x is a nonzero number.

    :param x: Arbitrary input.
    :return: Boolean (True if x is a nonzero number; False if x is 0, 0.0, None, np.nan, 'string', ...).
    """
    if isinstance(x, numbers.Number):
        if np.abs(x) < 1e-10:
            return False
        elif not np.isfinite(x):  # rule out np.nan since it is considered a number
            return False
        else:
            return True
    else:
        return False


def show_tree(model, depth=0):
    """
    Prints a model tree in the command line, showing the order of (sub)NodalModels and Nodes.
    Only the model argument should be entered by the user (do not enter a value for depth).

    :param model: Mother NodalModel object.
    :param depth: At what sublevel the NodalModel is. Used to automatically track the recursive depth of this function.
    """
    if depth == 0:
        print('\nMODEL TREE:')
        print(model.title)
    else:
        print('    '*(depth-1)+'\u2514---'+model.title)
    for node in model.directnodes:
        print('    '*depth+'\u2514---'+f'{node.name}')
    for submodel in model.submodels:
        depth += 1
        show_tree(submodel, depth)
        depth -= 1
    if depth == 0:
        print('\n')


def show_available_materials():
    """
    Print the available materials, coatings, and contact connections from Materials.csv in the console. These names can
    be copy-pasted into the material, coating, and connect arguments of Nodes and NodalModels.
    """
    print('AVAILABLE MATERIALS:')
    for i in mat_lst:
        print(i)
    print('\nAVAILABLE COATINGS:')
    for i in coat_lst:
        print(i)
    print('\nAVAILABLE CONTACT INTERFACES:')
    for i in con_lst:
        print(i)
    print('')


class Node:
    """
    The Node class is an object that defines one node.
    It contains the thermal and physical properties of a node.
    """
    def __init__(self, name='', material=None, coating=None, C_cap=None, mass=None, volume=None, T0=None, q_ext=None,
                 P_int=None, outer=None, alpha=None, epsilon=None, area=None, origin=(np.nan, np.nan, np.nan),
                 geom=(np.nan, np.nan, np.nan)):
        """
        Initiates a Node object. Beware of over-defining parameters; for example, applying a coating whilst also
        manually assigning alpha and epsilon results in two possible values for both alpha and epsilon. In such cases,
        the code will always prioritise the manually defined value.

        :param name: Unique name used to identify the node within a NodalModel. If not provided, it will be assigned
                     a number when added to a NodalModel.
        :param material: Material object or string containing the bulk properties (to compute C_cap) of the node.
        :param coating: Coating object or string containing the absorptivity (alpha) and emissivity (epsilon) of the node.
        :param C_cap: Heat capacity [J/K].
        :param mass: Mass of the node [kg], only needed if a material is applied and no C_cap is given.
        :param volume: Volume of the node [m^3], only needed if a material is applied and no C_cap or mass is given.
        :param T0: Initial temperature [deg C] or [K].
        :param q_ext: Tuple (q_pla, q_alb, q_s) of external heat fluxes [W/m^2] throughout time. Will be converted to
                      three separate Node attributes called Node.q_pla, Node.q_alb, Node.q_s.
        :param P_int: Internal power [W], throughout time.
        :param outer: Whether the node radiates freely to space or not. Default is False.
        :param alpha: Absorptivity in the solar spectrum. Default is 1.
        :param epsilon: Emissivity in the IR spectrum. Default is 1.
        :param area: Surface area [m^2] of the added node. Default is 1.
        :param origin: Tuple (x, y, z) of origin coordinates of the node. Default is (0, 0, 0).
        :param geom: Tuple (x_width, y_width, z_width) of the rectangular shape of the node. Default is (0, 0, 0).
        """
        self.warnings = False
        self.name = name
        if mass is not None:
            self.mass = mass
        else:
            self.mass = None
        if volume is not None:
            self.volume = volume
        else:
            self.volume = None
        if material is not None:
            if isinstance(material, Material) or (isinstance(material, str) and material in mat_lst):
                if isinstance(material, str) and material in mat_lst:
                    material = mat_lst[material]
                self.material = material
                if mass is not None:  # mass overrides volume if both are given
                    self.C_cap = mass*material.c_cap
                elif volume is not None:
                    self.C_cap = volume*material.density*material.c_cap
                elif C_cap is None:
                    self.C_cap = None
                    print(f"----------------ERROR----------------\n"
                          f"The heat capacity of node {self.name} could not be computed\n"
                          f"because the node also requires a mass when a Material object is assigned,\n"
                          f"to compute the heat capacity from the specific heat capacity.\n"
                          f"Assign a mass to the node, or manually define the heat capacity C_cap.\n"
                          f"-------------------------------------\n")
                    self.warnings = True
            elif C_cap is not None:
                self.C_cap = C_cap
                self.material = None
                print(f"---------------WARNING---------------\n"
                      f"Material assignment of node {self.name} has failed.\n"
                      f"However, the heat capacity was set to the manually entered C_cap.\n"
                      f"If a material is desired to be assigned, assure it is a 'Material' object,\n"
                      f"or a string object that is a valid entry of the mat_lst dictionary.\n"
                      f"-------------------------------------\n")
                self.warnings = True
            else:
                self.C_cap = None
                self.material = None
                print(f"----------------ERROR----------------\n"
                      f"Material assignment of node {self.name} has failed.\n"
                      f"The material assignment must be a 'Material' object,\n"
                      f"or a string object that is a valid entry of the mat_lst dictionary.\n"
                      f"-------------------------------------\n")
                self.warnings = True
        else:
            self.material = None
        if C_cap is not None:  # manually defined overrides any other calculated value
            self.C_cap = C_cap
        elif C_cap is None and material is None:
            self.C_cap = None
            print(f"----------------ERROR----------------\n"
                  f"The heat capacity of node {self.name} could not be assigned\n"
                  f"since no material nor heat capacity was given.\n"
                  f"Either assign a heat capacity C_cap, or a material and a mass/volume.\n"
                  f"-------------------------------------\n")
            self.warnings = True
        if coating is not None:
            if isinstance(coating, Coating) or (isinstance(coating, str) and coating in coat_lst):
                if isinstance(coating, str) and coating in coat_lst:
                    coating = coat_lst[coating]
                self.coating = coating
                self.alpha = coating.alpha
                self.epsilon = coating.epsilon
            elif alpha is not None and epsilon is not None:
                self.coating = None
                self.alpha = alpha
                self.epsilon = epsilon
                print(f"---------------WARNING---------------\n"
                      f"Coating assignment of node {self.name} has failed.\n"
                      f"However, the absorptivity and emissivity were set to the manually entered alpha and epsilon.\n"
                      f"If a coating is desired to be assigned, assure it is a 'Coating' object,\n"
                      f"or a string object that is a valid entry of the coat_lst dictionary.\n"
                      f"-------------------------------------\n")
                self.warnings = True
            else:
                self.coating = None
                self.alpha = 1.
                self.epsilon = 1.
                print(f"---------------WARNING---------------\n"
                      f"Coating assignment of node {self.name} has failed,\n"
                      f"and no other absorptivity and emissivity were assigned.\n"
                      f"Alpha and epsilon were set to 1.\n"
                      f"If a coating is desired to be assigned, assure it is a 'Coating' object,\n"
                      f"or a string object that is a valid entry of the coat_lst dictionary.\n"
                      f"Alternatively, define alpha and emissivity manually.\n"
                      f"-------------------------------------\n")
                self.warnings = True
        else:
            self.coating = None
        if alpha is not None and epsilon is not None:  # manually defined overrides any other calculated value
            self.alpha = alpha
            self.epsilon = epsilon
        elif epsilon is not None:
            self.epsilon = epsilon
            if coating is None:
                self.alpha = 1.
                print(f"---------------WARNING---------------\n"
                      f"The absorptivity of node {self.name} was not given,\n"
                      f"hence it was set to 1.\n"
                      f"Define your own value or apply a coating.\n"
                      f"-------------------------------------\n")
                self.warnings = True
            else:
                print(f"---------------WARNING---------------\n"
                      f"Node {self.name} was given both a coating and\n"
                      f"a separate emissivity definition.\n"
                      f"The manually defined emissivity overrules the coating emissivity.\n"
                      f"-------------------------------------\n")
                self.warnings = True
        elif alpha is not None:
            self.alpha = alpha
            if coating is None:
                self.epsilon = 1.
                print(f"---------------WARNING---------------\n"
                      f"The emissivity of node {self.name} was not given,\n"
                      f"hence it was set to 1.\n"
                      f"Define your own value or apply a coating.\n"
                      f"-------------------------------------\n")
                self.warnings = True
            else:
                print(f"---------------WARNING---------------\n"
                      f"Node {self.name} was given both a coating and\n"
                      f"a separate absorptivity definition.\n"
                      f"The manually defined absorptivity overrules the coating absorptivity.\n"
                      f"-------------------------------------\n")
                self.warnings = True
        elif self.coating is None:
            self.alpha = 1.
            self.epsilon = 1.
            print(f"---------------WARNING---------------\n"
                  f"The absorptivity and emissivity of node {self.name} were not given,\n"
                  f"hence they were set to 1.\n"
                  f"This warning can be ignored if the node does not emit/receive any radiation.\n"
                  f"-------------------------------------\n")
            self.warnings = True

        self.T0 = T0
        self.T = T0
        if outer is None:
            self.outer = False
        else:
            self.outer = outer
        self.origin = origin
        self.geom = geom
        if np.any(~np.isnan(geom)):  # if the node has geometry assigned
            axis = np.argwhere(np.abs(geom) < 1e-10)[0, 0]  # axis of zero thickness
            if axis == 0:
                self.area = geom[1] * geom[2]
            elif axis == 1:
                self.area = geom[0] * geom[2]
            elif axis == 2:
                self.area = geom[0] * geom[1]
            else:
                self.area = 1.
                if area is None:
                    print(f"----------------ERROR----------------\n"
                          f"The area of node {self.name} could not be computed\n"
                          f"due to a geometry definition error.\n"
                          f"The area of the node was set to 1 m2.\n"
                          f"Ensure that the node has a nonzero width and height, \n"
                          f"and a zero thickness in the direction of the surface normal.\n"
                          f"-------------------------------------\n")
                    self.warnings = True
        elif area is None:
            # No warning shown because if there would be radiation, the defined geometry would define the area anyway.
            self.area = 1.
        if area is not None:  # manually defined overrides any other calculated value
            self.area = area
        if q_ext is not None:
            self.q_pla = q_ext[0]
            self.q_alb = q_ext[1]
            self.q_s = q_ext[2]
            if not all(np.all(np.abs(x) < 1e-10) for x in q_ext) and outer is None:  # if q_ext is not zero
                # Only change outer if it is not given by the user.
                self.outer = True  # if the surface receives nonzero radiation, it must be an outer node
        else:
            self.q_pla = None
            self.q_alb = None
            self.q_s = None
        self.P_int = P_int
        self.number = None  # Node only has a number in the context of a NodalModel

    def modify(self, name_new=None, material_new=None, coating_new=None, C_cap_new=None, mass_new=None, volume_new=None,
               T0_new=None, q_ext_new=None, P_int_new=None, outer_new=None, alpha_new=None, epsilon_new=None,
               area_new=None, origin_new=None, geom_new=None):
        """
        Adapt values of the Node object. Beware of over-defining parameters; for example, applying a coating whilst also
        manually assigning alpha and epsilon results in two possible values for both alpha and epsilon. In such cases,
        the code will always prioritise the manually defined value.

        :param name_new: Unique name used to identify the node within a NodalModel.
        :param material_new: Material object or string containing the bulk properties (C_cap) of the node.
        :param coating_new: Coating object or string containing the absorptivity (alpha) and emissivity (epsilon) of the node.
        :param C_cap_new: Heat capacity [J/K]
        :param mass_new: Mass of the node [kg], only needed if a material is applied and no C_cap is given.
        :param volume_new: Volume of the node [m^3], only needed if a material is applied and no C_cap or mass is given.
        :param T0_new: Initial temperature [deg C] or [K]. Unit must be compatible with the unit of the NodalModel.
        :param q_ext_new: Tuple (q_pla, q_alb, q_s) of external heat fluxes [W/m^2] throughout time. Will be converted to
                          three separate Node attributes called Node.q_pla, Node.q_alb, Node.q_s.
        :param P_int_new: Internal power [W], throughout time.
        :param outer_new: Whether the node radiates freely to space or not. Default is False.
        :param alpha_new: Absorptivity in the solar spectrum. Default is 1.
        :param epsilon_new: Emissivity in the IR spectrum. Default is 1.
        :param area_new: Surface area [m^2] of the added node. Default is 1.
        :param origin_new: Tuple (x, y, z) of origin coordinates of the node. Default is (0, 0, 0).
        :param geom_new: Tuple (x_width, y_width, z_width) of the rectangular shape of the node. Default is (0, 0, 0).
        """
        if name_new is not None:
            self.name = name_new
        if mass_new is not None:
            self.mass = mass_new
        if volume_new is not None:
            self.volume = volume_new
        if material_new is not None:
            if isinstance(material_new, Material) or (isinstance(material_new, str) and material_new in mat_lst):
                if isinstance(material_new, str) and material_new in mat_lst:
                    material_new = mat_lst[material_new]
                self.material = material_new
                if self.mass is not None:  # if mass_new is given, it will automatically assign self.mass as well
                    self.C_cap = self.mass*material_new.c_cap
                elif self.volume is not None:
                    self.C_cap = self.volume*material_new.density*material_new.c_cap
                else:
                    print(f"---------------WARNING---------------\n"
                          f"Material assignment of node {self.name} has failed.\n"
                          f"The node has no mass or volume available.\n"
                          f"-------------------------------------\n")
                    self.warnings = True
            elif C_cap_new is not None:
                self.C_cap = C_cap_new
                print(f"---------------WARNING---------------\n"
                      f"Material assignment of node {self.name} has failed.\n"
                      f"However, the heat capacity was set to the manually entered C_cap.\n"
                      f"If a material is desired to be assigned, assure it is a 'Material' object,\n"
                      f"or a string object that is a valid entry of the mat_lst dictionary.\n"
                      f"-------------------------------------\n")
                self.warnings = True
            else:
                print(f"---------------WARNING---------------\n"
                      f"Material assignment of node {self.name} has failed.\n"
                      f"The material assignment must be a 'Material' object,\n"
                      f"or a string object that is a valid entry of the mat_lst dictionary.\n"
                      f"-------------------------------------\n")
                self.warnings = True
        if C_cap_new is not None:  # manually defined overrides any other calculated value
            self.C_cap = C_cap_new
        if coating_new is not None:
            if isinstance(coating_new, Coating) or (isinstance(coating_new, str) and coating_new in coat_lst):  # coating properties override manually assigned alpha & epsilon
                if isinstance(coating_new, str) and coating_new in coat_lst:
                    coating_new = coat_lst[coating_new]
                self.coating = coating_new
                self.alpha = coating_new.alpha
                self.epsilon = coating_new.epsilon
            elif alpha_new is not None and epsilon_new is not None:
                self.alpha = alpha_new
                self.epsilon = epsilon_new
                print(f"---------------WARNING---------------\n"
                      f"Coating assignment of node {self.name} has failed.\n"
                      f"However, the absorptivity and emissivity were set to the manually entered alpha and epsilon.\n"
                      f"If a coating is desired to be assigned, assure it is a 'Coating' object,\n"
                      f"or a string object that is a valid entry of the coat_lst dictionary.\n"
                      f"-------------------------------------\n")
                self.warnings = True
            else:
                print(f"---------------WARNING---------------\n"
                      f"Coating assignment of node {self.name} has failed.\n"
                      f"The coating assignment must be a 'Coating' object,\n"
                      f"or a string object that is a valid entry of the coat_lst dictionary.\n"
                      f"-------------------------------------\n")
                self.warnings = True
        if alpha_new is not None:  # manually defined overrides any other calculated value
            self.alpha = alpha_new
        if epsilon_new is not None:  # manually defined overrides any other calculated value
            self.epsilon = epsilon_new

        if T0_new is not None:
            self.T0 = T0_new
            self.T = T0_new
        if q_ext_new is not None:
            self.q_pla = q_ext_new[0]
            self.q_alb = q_ext_new[1]
            self.q_s = q_ext_new[2]
            if not all(np.all(np.abs(x) < 1e-10) for x in q_ext_new) and outer_new is None:  # if q_ext is not zero
                # Only change outer if it is not given by the user.
                self.outer = True  # if the surface receives nonzero environment radiation, it must be an outer node
        if P_int_new is not None:
            self.P_int = P_int_new
        if outer_new is not None:
            self.outer = outer_new
        if origin_new is not None:
            self.origin = origin_new
        if geom_new is not None:
            self.geom = geom_new
            # Geometry overrides manually assigned area
            axis = np.argwhere(np.abs(geom_new) < 1e-10)[0, 0]  # axis of zero thickness
            if axis == 0:
                self.area = geom_new[1] * geom_new[2]
                if area_new is None:
                    print(f"---------------WARNING---------------\n"
                          f"The newly assigned geometry of node {self.name} has overridden the area as well.\n"
                          f"If this is not desired, assign a new area manually.\n"
                          f"-------------------------------------\n")
                    self.warnings = True
            elif axis == 1:
                self.area = geom_new[0] * geom_new[2]
                if area_new is None:
                    print(f"---------------WARNING---------------\n"
                          f"The newly assigned geometry of node {self.name} has overridden the area as well.\n"
                          f"If this is not desired, assign a new area manually.\n"
                          f"-------------------------------------\n")
                    self.warnings = True
            elif axis == 2:
                self.area = geom_new[0] * geom_new[1]
                if area_new is None:
                    print(f"---------------WARNING---------------\n"
                          f"The newly assigned geometry of node {self.name} has overridden the area as well.\n"
                          f"If this is not desired, assign a new area manually.\n"
                          f"-------------------------------------\n")
                    self.warnings = True
            if area_new is not None:  # manually defined overrides any other calculated value
                self.area = area_new


class NodalModel:
    """
    The NodalModel class is an object that defines one nodal model.
    It contains functions to define and expand the model, and to solve the model.
    """
    def __init__(self, t=None, celsius=True, title=''):
        """
        Initialises the NodalModel object.
        The temperatures can be defined in either degrees Celsius or Kelvin; the output unit matches the input.

        :param t: Time vector [s].
        :param celsius: Boolean, whether the temperatures are in Celsius (True) or Kelvin (False). Default is Celsius.
        :param title: Title/name that the NodalModel has.
        """
        if t is not None:
            self.t = t
            self.dt = t[1] - t[0]
            self.P_int = np.zeros((self.t.shape[0], 0))
            self.q_pla = np.zeros((self.t.shape[0], 0))
            self.q_alb = np.zeros((self.t.shape[0], 0))
            self.q_s = np.zeros((self.t.shape[0], 0))
            self.T = np.zeros((self.t.shape[0], 0))
            self.solvable = True  # Used to prevent solving the model without a time array etc.
        else:
            self.t = None
            self.dt = None
            self.P_int = None
            self.q_pla = None
            self.q_alb = None
            self.q_s = None
            self.T = None
            self.solvable = False  # Used to prevent solving the model without a time array etc.

        self.n = 0
        self.C_cap = np.array([])
        self.C_con = np.array([[]])
        self.T0 = np.array([])
        self.outer = np.array([])
        self.alpha = np.array([])
        self.epsilon = np.array([])
        self.area = np.array([])
        self.celsius = celsius
        self.name = np.array([])
        self.rad = np.array([[]])
        self.origin = np.zeros((0, 3))
        self.geom = np.zeros((0, 3))
        self.title = title
        self.nodes = []
        self.connections = []
        self.submodels = []
        self.directnodes = []  # Nodes in the main NodalModel but are not part of a sub-NodalModel. Used for show_tree.
        self.warnings = False
        self.solved = False  # Used to prevent trying to plot a model with no solved data.
        self.heatflow = None

    def add_node(self, node_or_model):
        """
        Adds a node (Node object) or collection of nodes (NodalModel object) to the current NodalModel object.
        Can also add multiple objects at once by feeding a list of objects into the function.

        :param node_or_model: Node or NodalModel object. May also be a list of Nodes and/or NodalModels.
        """
        node_or_model = ensure_list(node_or_model)
        for n_or_m in node_or_model:
            if isinstance(n_or_m, Node):
                node = n_or_m
                if node.warnings:
                    self.warnings = True
                self.nodes.append(node)
                self.directnodes.append(node)
                node.number = self.n  # assign before n+=1 because the numbering starts from zero
                self.n += 1
                self.C_cap = np.append(self.C_cap, node.C_cap)
                self.T0 = np.append(self.T0, node.T0)
                self.outer = np.append(self.outer, node.outer)
                self.epsilon = np.append(self.epsilon, node.epsilon)
                self.alpha = np.append(self.alpha, node.alpha)
                if node.name == '':
                    node.name = str(node.number)
                if node.name in self.name:
                    print("----------------ERROR----------------\n"
                          f"The name of the added Node ('{node.name}') already exists in the main NodalModel.\n"
                          "This might result in incorrect computations due to wrongly identified Nodes.\n"
                          "Ensure that the added Node's name is not yet present in the NodalModel.\n"
                          "-------------------------------------\n")
                    self.warnings = True
                self.name = np.append(self.name, node.name)
                self.origin = np.vstack((self.origin, np.array(node.origin)))
                self.geom = np.vstack((self.geom, np.array(node.geom)))
                self.area = np.append(self.area, node.area)

                if self.solvable:  # Time-array and time-varying data available
                    self.T = np.pad(self.T, [(0, 0), (0, 1)], mode='constant', constant_values=0)
                    self.T[0, -1] = node.T0

                    err = False  # variable to ensure that the error is only displayed once
                    if node.q_pla is None:
                        node.q_pla = np.zeros(self.t.shape[0])
                    if node.q_pla.shape[0] == self.t.shape[0] and node.q_pla.shape != self.t.shape:  # if the power array is entered in the wrong format
                        self.q_pla = np.append(self.q_pla, node.q_pla, axis=1)
                    else:
                        if self.t.shape[0] in node.q_pla.shape:
                            self.q_pla = np.append(self.q_pla, np.reshape(node.q_pla, (self.t.shape[0], 1)), axis=1)
                        else:
                            print("----------------ERROR----------------\n"
                                  "The main NodalModel and added Node have different time lengths, hence cannot be broadcast together.\n"
                                  "The time-varying arrays of the Node were added to the NodalModel as empty arrays.\n"
                                  "Ensure that all time-dependent arrays have identical shapes.\n"
                                  "-------------------------------------\n")
                            self.warnings = True
                            err = True
                            self.q_pla = np.append(self.q_pla, np.zeros((self.t.shape[0], 1)), axis=1)

                    if node.q_alb is None:
                        node.q_alb = np.zeros(self.t.shape[0])
                    if node.q_alb.shape[0] == self.t.shape[0] and node.q_alb.shape != self.t.shape:  # if the power array is entered in the wrong format
                        self.q_alb = np.append(self.q_alb, node.q_alb, axis=1)
                    else:
                        if self.t.shape[0] in node.q_alb.shape:
                            self.q_alb = np.append(self.q_alb, np.reshape(node.q_alb, (self.t.shape[0], 1)), axis=1)
                        else:
                            if not err:
                                print("----------------ERROR----------------\n"
                                      "The main NodalModel and added Node have different time lengths, hence cannot be broadcast together.\n"
                                      "The time-varying arrays of the Node were added to the NodalModel as empty arrays.\n"
                                      "Ensure that all time-dependent arrays have identical shapes.\n"
                                      "-------------------------------------\n")
                                self.warnings = True
                                err = True
                            self.q_alb = np.append(self.q_alb, np.zeros((self.t.shape[0], 1)), axis=1)

                    if node.q_s is None:
                        node.q_s = np.zeros(self.t.shape[0])
                    if node.q_s.shape[0] == self.t.shape[0] and node.q_s.shape != self.t.shape:  # if the power array is entered in the wrong format
                        self.q_s = np.append(self.q_s, node.q_s, axis=1)
                    else:
                        if self.t.shape[0] in node.q_s.shape:
                            self.q_s = np.append(self.q_s, np.reshape(node.q_s, (self.t.shape[0], 1)), axis=1)
                        else:
                            if not err:
                                print("----------------ERROR----------------\n"
                                      "The main NodalModel and added Node have different time lengths, hence cannot be broadcast together.\n"
                                      "The time-varying arrays of the Node were added to the NodalModel as empty arrays.\n"
                                      "Ensure that all time-dependent arrays have identical shapes.\n"
                                      "-------------------------------------\n")
                                self.warnings = True
                                err = True
                            self.q_s = np.append(self.q_s, np.zeros((self.t.shape[0], 1)), axis=1)
                    if node.P_int is None:
                        node.P_int = np.zeros(self.t.shape[0])
                    if isinstance(node.P_int, numbers.Number):
                        if np.isfinite(node.P_int):  # np.nan is also considered a number so add an extra check (beware: isfinite does not work with None)
                            node.P_int *= np.ones(self.t.shape[0])
                    if node.P_int.shape[0] == self.t.shape[0] and node.P_int.shape != self.t.shape:  # if the power array is entered in the wrong format
                        self.P_int = np.append(self.P_int, node.P_int, axis=1)
                    else:
                        if self.t.shape[0] in node.P_int.shape:
                            self.P_int = np.append(self.P_int, np.reshape(node.P_int, (self.t.shape[0], 1)), axis=1)
                        else:
                            if not err:
                                print("----------------ERROR----------------\n"
                                      "The main NodalModel and added Node have different time lengths, hence cannot be broadcast together.\n"
                                      "The time-varying arrays of the Node were added to the NodalModel as empty arrays.\n"
                                      "Ensure that all time-dependent arrays have identical shapes.\n"
                                      "-------------------------------------\n")
                                self.warnings = True
                            self.P_int = np.append(self.P_int, np.zeros((self.t.shape[0], 1)), axis=1)
                else:  # No time array and time-varying data available
                    if node.q_pla is not None or node.q_alb is not None or node.q_s is not None or \
                            node.P_int is not None:
                        print("----------------ERROR----------------\n"
                              "Added object contains power fluxes/inputs, but the NodalModel has no time array assigned.\n"
                              "The object's power fluxes/inputs were not added to the model.\n"
                              "First assign a time array to the NodalModel, then assign the power fluxes/inputs again,\n"
                              "while ensuring that the power fluxes are compatible with the time array.\n"
                              "-------------------------------------\n")
                        self.warnings = True

                if self.n == 1:
                    self.C_con = np.zeros((1, 1))
                    self.rad = np.zeros((1, 1))
                else:
                    self.C_con = np.pad(self.C_con, [(0, 1), (0, 1)], mode='constant', constant_values=0)
                    self.rad = np.pad(self.rad, [(0, 1), (0, 1)], mode='constant', constant_values=0)
            elif isinstance(n_or_m, NodalModel):  # one "level" of depth is sufficient because the model represents all its subnodes and submodels.
                def add_non_time_variables():
                    # Keep the connections within the submodel the same
                    if self.n == 0:
                        self.C_con = nodal_model.C_con
                        self.rad = nodal_model.rad
                    else:
                        self.C_con = np.pad(self.C_con, [(0, nodal_model.n), (0, nodal_model.n)], mode='constant',
                                            constant_values=0)
                        self.C_con[-nodal_model.n:, -nodal_model.n:] = nodal_model.C_con
                        self.rad = np.pad(self.rad, [(0, nodal_model.n), (0, nodal_model.n)], mode='constant',
                                          constant_values=0)
                        self.rad[-nodal_model.n:, -nodal_model.n:] = nodal_model.rad

                    re_number = []
                    for idx, subnode in enumerate(nodal_model.nodes):
                        oldname = subnode.name
                        change = False
                        if subnode.name == str(subnode.number):
                            change = True
                        subnode.number = self.n  # assign before n+=1 because the numbering starts from zero
                        self.n += 1
                        if subnode.name == '' or change:  # node names/numbers are changed here, permanently changing the names of connections as well
                            subnode.name = str(subnode.number)
                            nodal_model.name[idx] = subnode.name
                            for indx, conn in enumerate(nodal_model.connections):
                                if oldname in conn:
                                    if oldname == conn[0]:
                                        re_number.append(tuple((subnode.name, indx, 0)))
                                    else:
                                        re_number.append(tuple((subnode.name, indx, 1)))
                        if subnode.name in self.name:
                            print("----------------ERROR----------------\n"
                                  f"The name of the added Node ('{subnode.name}') already exists in the main NodalModel.\n"
                                  "This might result in incorrect computations due to wrongly identified Nodes.\n"
                                  "Ensure that the added Node's name is not yet present in the NodalModel.\n"
                                  "-------------------------------------\n")
                            self.warnings = True

                    for newnumber in re_number:  # the automatically assigned name must change to the new node number
                        oldtup = list(nodal_model.connections[newnumber[1]])
                        if newnumber[2] == 0:
                            oldtup[0] = newnumber[0]
                        else:
                            oldtup[1] = newnumber[0]
                        nodal_model.connections[newnumber[1]] = tuple(oldtup)

                    self.submodels.append(nodal_model)
                    self.nodes.extend(nodal_model.nodes)  # Use .extend instead of .append
                    self.C_cap = np.concatenate((self.C_cap, nodal_model.C_cap))
                    self.T0 = np.concatenate((self.T0, nodal_model.T0))
                    self.outer = np.concatenate((self.outer, nodal_model.outer))
                    self.epsilon = np.concatenate((self.epsilon, nodal_model.epsilon))
                    self.alpha = np.concatenate((self.alpha, nodal_model.alpha))
                    self.name = np.concatenate((self.name, nodal_model.name))
                    self.origin = np.vstack((self.origin, nodal_model.origin))
                    self.geom = np.vstack((self.geom, nodal_model.geom))
                    self.area = np.concatenate((self.area, nodal_model.area))
                    self.connections.extend(nodal_model.connections)

                def add_time_variables():
                    self.T = np.pad(self.T, [(0, 0), (0, nodal_model.n)], mode='constant', constant_values=0)
                    self.T[0, -nodal_model.n:] = nodal_model.T0

                    # Do not need to do "if ... is None" checks because was already done when making the sub NodalModel
                    self.q_pla = np.hstack((self.q_pla, nodal_model.q_pla))
                    self.q_alb = np.hstack((self.q_alb, nodal_model.q_alb))
                    self.q_s = np.hstack((self.q_s, nodal_model.q_s))
                    self.P_int = np.hstack((self.P_int, nodal_model.P_int))

                nodal_model = n_or_m
                if nodal_model.warnings:
                    self.warnings = True
                if self.solvable and nodal_model.solvable:  # Time-array and time-varying data available
                    if self.t.shape != nodal_model.t.shape:
                        print("----------------ERROR----------------\n"
                              "NodalModels have different time arrays, hence cannot be broadcast together.\n"
                              "Ensure that all time-dependent arrays have identical shapes.\n"
                              "-------------------------------------\n")
                        self.warnings = True
                    elif np.any(np.abs(self.t - nodal_model.t) > 1e-10):
                        print("----------------ERROR----------------\n"
                              "NodalModels have different time arrays, hence cannot be broadcast together.\n"
                              "Ensure that all time-dependent arrays have identical shapes.\n"
                              "-------------------------------------\n")
                        self.warnings = True
                    else:
                        add_non_time_variables()
                        add_time_variables()
                elif not self.solvable and nodal_model.solvable:  # No time-array and time-varying data available for the self NodalModel
                    print("---------------WARNING---------------\n"
                          "The main NodalModel has no time array assigned, while the added NodalModel does.\n"
                          "The time array from the added NodalModel was implemented into the main NodalModel.\n"
                          "Alternatively, manually assign a time array to the main model before adding the new model.\n"
                          "-------------------------------------\n")
                    self.warnings = True

                    self.set_time(nodal_model.t)
                    add_non_time_variables()
                    add_time_variables()
                    self.solvable = True
                elif not nodal_model.solvable and self.solvable:  # No time-array and time-varying data available for the added NodalModel
                    print("---------------WARNING---------------\n"
                          "The added NodalModel has no time array assigned, while the main NodalModel does.\n"
                          "The time array from the main NodalModel was implemented into the added NodalModel.\n"
                          "Alternatively, manually assign a time array to the added model before adding the new model.\n"
                          "-------------------------------------\n")
                    self.warnings = True

                    nodal_model.set_time(self.t)
                    add_non_time_variables()
                    add_time_variables()
                    nodal_model.solvable = True
                else:  # Both models have no time arrays assigned, adding them is doable.
                    add_non_time_variables()
            else:
                print("----------------ERROR----------------\n"
                      "Added object must be a Node or NodalModel object,\n"
                      "or a list of multiple Nodes and/or NodalModels.\n"
                      "-------------------------------------\n")
                self.warnings = True

    def set_time(self, t, erase=False, print_err=True):
        """
        If the NodalModel did not yet have time-varying parameters assigned, this function adds a time array and adds
        zero-valued entries for all other time-varying parameters. Also sets the same time for sub-NodalModels.
        If any previous time data is to be removed and replaced by the new data, provide the input erase=True.

        :param t: Time vector [s].
        :param erase: Boolean to indicate whether previously existing time data is allowed to be erased and replaced by
                      the new data.
        :param print_err: To be ignored by the user. Boolean to avoid printing the same error multiple times.
        """
        if self.solvable and not erase and print_err:  # user may have not realised old time data is available
            print("---------------WARNING---------------\n"
                  "NodalModel already has a time array assigned, set_time() was aborted.\n"
                  "Assigning a new time array would have erased all previous time data.\n"
                  "Provide the parameter erase=True if old time data is to be erased.\n"
                  "-------------------------------------\n")
            self.warnings = True
            print_err = False
        else:
            self.t = t
            self.dt = t[1] - t[0]
            self.P_int = np.zeros((self.t.shape[0], self.n))
            self.q_pla = np.zeros((self.t.shape[0], self.n))
            self.q_alb = np.zeros((self.t.shape[0], self.n))
            self.q_s = np.zeros((self.t.shape[0], self.n))
            self.T = np.zeros((self.t.shape[0], self.n))
            self.solvable = True  # Used to prevent solving the model without a time array etc.
        for submodel in self.submodels:
            submodel.set_time(t, erase, print_err)

    def connect(self, node1, nodes2, contact_obj=None, rad=None, C_con=None, k_through=None, L_through=None,
                h_contact=None, A=None):
        """
        Defines connections between the first node to be entered (node1) and the specified other nodes (nodes2).
        This function can be used to define a single connection between two nodes, or define multiple connections.
        Multiple connections are defined using a single node1, and a list of nodes2 and corresponding parameters.
        The function adds the connections as: [node1<->nodes2[0], node1<->nodes2[1], node1<->nodes2[2], ...].

        If a list of multiple connections is used, type None for the connections that are not desired to be made; for
        example, for two connections (one conductive, one radiative), C_con would look like C_con=[value, None].

        If it is a through-connection and both nodes have the same Material assigned, only L_through and A need to be
        provided.

        :param node1: Node object, string, or integer indicating the first connected node.
        :param nodes2: (List of) Node object(s), string(s), or integer(s) indicating the second connected node(s).
        :param contact_obj: (List of) Contact object(s), using a pre-defined contact interface.
        :param rad: (List of) Boolean(s) indicating whether the connections include radiative heat transfer.
        :param C_con: (List of) conductance value(s) [W/K] for the corresponding connections.
        :param k_through: (List of) thermal conductivity(ies) [W/(m.K)] between the nodes, for a through-material
                          connection. Does not need to be defined if the nodes have the same Material object assigned.
        :param L_through: (List of) distance(s) between nodes [m], for a through-material connection.
        :param h_contact: (List of) heat transfer coefficient(s) [W/(m^2.K)], for a contact connection.
        :param A: (List of) orthogonal surface area(s) [m^2] between the nodes.
        """
        # Ensure the for-loop always works, and just runs once if only one connection is being made
        nodes2 = ensure_list(nodes2)
        contact_obj = ensure_list(contact_obj)
        C_con = ensure_list(C_con)
        rad = ensure_list(rad)
        k_through = ensure_list(k_through)
        L_through = ensure_list(L_through)
        h_contact = ensure_list(h_contact)
        A = ensure_list(A)

        # Ensure all lists have the correct length
        if len(contact_obj) == 1:
            contact_obj *= len(nodes2)  # duplicates the list until it has the same length as nodes2
        if len(C_con) == 1:
            C_con *= len(nodes2)
        if len(rad) == 1:
            rad *= len(nodes2)
        if len(k_through) == 1:
            k_through *= len(nodes2)
        if len(L_through) == 1:
            L_through *= len(nodes2)
        if len(h_contact) == 1:
            h_contact *= len(nodes2)
        if len(A) == 1:
            A *= len(nodes2)

        for i in range(len(nodes2)):
            # Converting the type of node to an identification/index number
            if isinstance(node1, str):
                node1 = np.argwhere(self.name == node1)[0, 0]
            elif isinstance(node1, Node):
                node1 = np.argwhere(self.name == node1.name)[0, 0]
            if isinstance(nodes2[i], str):
                nodes2[i] = np.argwhere(self.name == nodes2[i])[0, 0]
            elif isinstance(nodes2[i], Node):
                nodes2[i] = np.argwhere(self.name == nodes2[i].name)[0, 0]

            # Check method of conduction calculation
            con_changed = False
            con_set_zero = False
            err = False
            if not nonzero(C_con[i]):  # C_con is either zero or it was not entered
                if C_con[i] is not None:  # If a zero was entered, it must be set to zero.
                    C_con[i] = 0.
                    con_set_zero = True
                    con_changed = True
                else:
                    if self.nodes[node1].material is not None and self.nodes[nodes2[i]].material is not None:
                        # a through-connection is desired (L and A defined) and k is not overridden
                        if k_through[i] is None and nonzero(L_through[i]) and nonzero(A[i]):
                            if self.nodes[node1].material.name == self.nodes[nodes2[i]].material.name:
                                k_through[i] = self.nodes[node1].material.k_through
                            else:
                                print(f"----------------ERROR----------------\n"
                                      f"The through-conduction of nodes '{self.name[node1]}' and '{self.name[nodes2[i]]}' could not be computed,\n"
                                      f"since the nodes have different materials assigned.\n"
                                      f"Ensure that both materials are identical, or manually define k_through.\n"
                                      f"-------------------------------------\n")
                                self.warnings = True
                                err = True
                    cont = False
                    if h_contact[i] is None and nonzero(A[i]) and contact_obj[i] is not None:
                        # a contact connection is desired (A and contact_obj defined) and h is not overridden
                        if isinstance(contact_obj[i], Contact) or (isinstance(contact_obj[i], str) and contact_obj[i] in con_lst):
                            if isinstance(contact_obj[i], str) and contact_obj[i] in con_lst:
                                contact_obj[i] = con_lst[contact_obj[i]]
                            h_contact[i] = contact_obj[i].h_contact
                            cont = True
                        else:
                            print(f"----------------ERROR----------------\n"
                                  f"The contact conduction of nodes '{self.name[node1]}' and '{self.name[nodes2[i]]}' could not be computed,\n"
                                  f"since 'contact_obj' is not of type Contact,\n"
                                  f"or a string object that is a valid entry of the con_lst dictionary.\n"
                                  f"-------------------------------------\n")
                            self.warnings = True
                            err = True
                    if nonzero(k_through[i]) and nonzero(L_through[i]) and nonzero(A[i]):  # through conduction
                        C_con[i] = k_through[i]*A[i]/L_through[i]
                        con_changed = True
                    elif (nonzero(h_contact[i]) or cont) and nonzero(A[i]):  # contact conduction
                        C_con[i] = h_contact[i]*A[i]
                        con_changed = True
            else:
                con_changed = True

            if con_changed:  # New conductive connection present
                if con_set_zero:  # Connection was set to zero, hence must be removed from the list
                    if tuple((self.nodes[node1].name, self.nodes[nodes2[i]].name, 'conductive')) in self.connections:
                        self.connections.remove((self.nodes[node1].name, self.nodes[nodes2[i]].name, 'conductive'))
                    elif tuple((self.nodes[nodes2[i]].name, self.nodes[node1].name, 'conductive')) in self.connections:
                        self.connections.remove((self.nodes[nodes2[i]].name, self.nodes[node1].name, 'conductive'))
                else:
                    if tuple((self.nodes[node1].name, self.nodes[nodes2[i]].name, 'conductive')) not in self.connections \
                            and tuple((self.nodes[nodes2[i]].name, self.nodes[node1].name, 'conductive')) not in self.connections:
                        self.connections.append((self.nodes[node1].name, self.nodes[nodes2[i]].name, 'conductive'))
                self.C_con[node1, nodes2[i]] = C_con[i]
                self.C_con[nodes2[i], node1] = C_con[i]
            else:
                if rad[i] is None and not err:  # no connection was made and no error has been printed yet
                    print(f"----------------ERROR----------------\n"
                          f"The conduction between nodes '{self.name[node1]}' and '{self.name[nodes2[i]]}' could not be computed,\n"
                          f"since the correct parameters were not given.\n"
                          f"For a through-connection, give k, L, and A (or just L and A if the Nodes have materials).\n"
                          f"For a contact connection, give h and A (or contact_obj and A).\n"
                          f"-------------------------------------\n")
                    self.warnings = True

            if rad[i] is not None:  # ensure the 'if rad[i]' works, instead of relying on if None == False
                if rad[i]:
                    if tuple((self.nodes[node1].name, self.nodes[nodes2[i]].name, 'radiative')) not in self.connections and \
                            tuple((self.nodes[nodes2[i]].name, self.nodes[node1].name, 'radiative')) not in self.connections:
                        self.connections.append((self.nodes[node1].name, self.nodes[nodes2[i]].name, 'radiative'))
                    if np.any(np.isnan(self.nodes[node1].origin)) or np.any(np.isnan(self.nodes[node1].geom))\
                            or np.any(np.isnan(self.nodes[nodes2[i]].origin)) or np.any(np.isnan(self.nodes[nodes2[i]].geom)):
                        print(f"----------------ERROR----------------\n"
                              f"The radiative link between nodes '{self.name[node1]}' and '{self.name[nodes2[i]]}'\n"
                              f"could not be computed because either of/both nodes have\n"
                              f"no origin and/or geometry assigned.\n"
                              f"Add an origin and geometry to the node to allow for radiative computations.\n"
                              f"-------------------------------------\n")
                        self.warnings = True
                    else:
                        if np.any(np.abs(self.nodes[node1].geom) < 1e-10):  # if there is a zero thickness in the emitting node
                            axis1 = np.argwhere(np.abs(self.nodes[node1].geom) < 1e-10)[0, 0]  # normal direction of emitting plate
                            if np.any(np.abs(self.nodes[nodes2[i]].geom) < 1e-10):  # if there is a zero thickness in the receiving node
                                axis2 = np.argwhere(np.abs(self.nodes[nodes2[i]].geom) < 1e-10)[0, 0]  # normal direction of receiving plate
                                if axis1 == axis2:
                                    direc = [0, 1, 2]
                                    direc.remove(axis1)  # the two axes that must be the same size remain
                                    if np.abs(self.nodes[node1].geom[direc[0]] - self.nodes[nodes2[i]].geom[direc[0]]) > 1e-10 \
                                            or np.abs(self.nodes[node1].geom[direc[1]] - self.nodes[nodes2[i]].geom[direc[1]]) > 1e-10:  # plates have different shapes
                                        print(f"---------------WARNING---------------\n"
                                              f"The two parallel plates of nodes '{self.name[node1]}' and '{self.name[nodes2[i]]}'"
                                              f" have different sizes.\n"
                                              f"For more accurate view factor computations,\n"
                                              f"their sizes should be equal.\n"
                                              f"-------------------------------------\n")
                                        self.warnings = True
                                    if np.abs(self.nodes[node1].origin[direc[0]] - self.nodes[nodes2[i]].origin[direc[0]]) > 1e-10 \
                                            or np.abs(self.nodes[node1].origin[direc[1]] - self.nodes[nodes2[i]].origin[direc[1]]) > 1e-10:  # plates are not aligned
                                        print(f"---------------WARNING---------------\n"
                                              f"The two parallel plates of nodes '{self.name[node1]}' and '{self.name[nodes2[i]]}'"
                                              f" are not aligned properly.\n"
                                              f"For more accurate view factor computations,\n"
                                              f"they must be perfectly below/above or beside each other.\n"
                                              f"-------------------------------------\n")
                                        self.warnings = True
                                    geom_nonzero = np.array(self.nodes[node1].geom)[np.argwhere(np.abs(self.nodes[node1].geom) > 1e-10)][:, 0]
                                    vf = view_factor_par(geom_nonzero[0], geom_nonzero[1],
                                                         np.abs(self.nodes[node1].origin[axis1] - self.nodes[nodes2[i]].origin[axis2]))
                                else:
                                    common_axis = 3 - axis1 - axis2  # axis of connected edge
                                    axes_letters = ['x', 'y', 'z']
                                    if np.abs(self.nodes[node1].geom[common_axis] - self.nodes[nodes2[i]].geom[common_axis]) > 1e-10:  # plates have different length common axis
                                        print(f"---------------WARNING---------------\n"
                                              f"The common edge (axis {axes_letters[common_axis]}) of the two perpendicular nodes\n"
                                              f"'{self.name[node1]}' and '{self.name[nodes2[i]]}'"
                                              f" have different sizes.\n"
                                              f"For more accurate view factor computations,\n"
                                              f"their common edge length should be equal.\n"
                                              f"-------------------------------------\n")
                                        self.warnings = True
                                    if np.abs(self.nodes[node1].origin[common_axis] - self.nodes[nodes2[i]].origin[common_axis]) > 1e-10:  # plates have offset along common axis
                                        print(f"---------------WARNING---------------\n"
                                              f"The common edge (axis {axes_letters[common_axis]}) of the two perpendicular nodes\n"
                                              f"'{self.name[node1]}' and '{self.name[nodes2[i]]}'"
                                              f" are not aligned properly.\n"
                                              f"For more accurate view factor computations,\n"
                                              f"they must be perfectly centered with respect to each other.\n"
                                              f"-------------------------------------\n")
                                        self.warnings = True
                                    if np.abs((self.nodes[nodes2[i]].origin[3-common_axis-axis2]-self.nodes[nodes2[i]].geom[3-common_axis-axis2]/2)
                                              - self.nodes[node1].origin[3-common_axis-axis2]) > 1e-10 \
                                            and np.abs((self.nodes[nodes2[i]].origin[3-common_axis-axis2]+self.nodes[nodes2[i]].geom[3-common_axis-axis2]/2)
                                                       - self.nodes[node1].origin[3-common_axis-axis2]) > 1e-10:  # plates are misaligned, case 1
                                        print(f"---------------WARNING---------------\n"
                                              f"The common edge (axis {axes_letters[common_axis]}) of the two perpendicular nodes\n"
                                              f"'{self.name[node1]}' and '{self.name[nodes2[i]]}'"
                                              f" do not line up in the {axes_letters[3-common_axis-axis2]} direction.\n"
                                              f"For more accurate view factor computations,\n"
                                              f"their common edges should line up perfectly.\n"
                                              f"-------------------------------------\n")
                                        self.warnings = True
                                    if np.abs((self.nodes[node1].origin[3-common_axis-axis1]-self.nodes[node1].geom[3-common_axis-axis1]/2)
                                              - self.nodes[nodes2[i]].origin[3-common_axis-axis1]) > 1e-10 \
                                            and np.abs((self.nodes[node1].origin[3-common_axis-axis1]+self.nodes[node1].geom[3-common_axis-axis1]/2)
                                                       - self.nodes[nodes2[i]].origin[3-common_axis-axis1]) > 1e-10:  # plates are misaligned, case 2
                                        print(f"---------------WARNING---------------\n"
                                              f"The common edge (axis {axes_letters[common_axis]}) of the two perpendicular nodes\n"
                                              f"'{self.name[node1]}' and '{self.name[nodes2[i]]}'"
                                              f" do not line up in the {axes_letters[3-common_axis-axis1]} direction.\n"
                                              f"For more accurate view factor computations,\n"
                                              f"their common edges should line up perfectly.\n"
                                              f"-------------------------------------\n")
                                        self.warnings = True
                                    b = self.nodes[node1].geom[common_axis]  # length of connected edge
                                    a2 = self.nodes[nodes2[i]].geom[3 - common_axis - axis2]
                                    c1 = self.nodes[node1].geom[3 - common_axis - axis1]
                                    vf = view_factor_perp(a2, b, c1)
                            else:
                                print(f"----------------ERROR----------------\n"
                                      f"The radiation received by node '{self.name[nodes2[i]]}' could not be computed\n"
                                      f"due to a geometry definition error.\n"
                                      f"The view factor from node '{self.name[node1]}' to '{self.name[nodes2[i]]}' was set to zero.\n"
                                      f"Ensure that the node has a nonzero width and height, "
                                      f"and a zero thickness in the direction of the surface normal.\n"
                                      f"-------------------------------------\n")
                                self.warnings = True
                                vf = 0.
                        else:
                            print(f"----------------ERROR----------------\n"
                                  f"The radiation emitted by node '{self.name[node1]}' could not be computed\n"
                                  f"due to a geometry definition error.\n"
                                  f"The view factor from node '{self.name[node1]}' to '{self.name[nodes2[i]]}' was set to zero.\n"
                                  f"Ensure that the node has a nonzero width and height, "
                                  f"and a zero thickness in the direction of the surface normal.\n"
                                  f"-------------------------------------\n")
                            self.warnings = True
                            vf = 0.

                        self.rad[node1, nodes2[i]] = vf * self.nodes[node1].area * self.nodes[node1].epsilon
                        self.rad[nodes2[i], node1] = vf * self.nodes[node1].area * self.nodes[node1].epsilon  # using e1A1F12 = e2A2F21
                else:
                    self.rad[node1, nodes2[i]] = 0.
                    self.rad[nodes2[i], node1] = 0.
                    if tuple((self.nodes[node1].name, self.nodes[nodes2[i]].name, 'radiative')) in self.connections:
                        self.connections.remove((self.nodes[node1].name, self.nodes[nodes2[i]].name, 'radiative'))
                    elif tuple((self.nodes[nodes2[i]].name, self.nodes[node1].name, 'radiative')) in self.connections:
                        self.connections.remove((self.nodes[nodes2[i]].name, self.nodes[node1].name, 'radiative'))

    def modify_node(self, node_mod, name_new=None, material_new=None, coating_new=None, C_cap_new=None, mass_new=None,
                    volume_new=None, T0_new=None, q_ext_new=None, P_int_new=None, outer_new=None, alpha_new=None,
                    epsilon_new=None, area_new=None, origin_new=None, geom_new=None):
        """
        Adapt values of an existing Node in the NodalModel. The Node object itself will be changed, as well as
        all its values within the NodalModel matrices/arrays.

        :param node_mod: Node object, string, or integer indicating the node to be modified.
        :param name_new: Unique name used to identify the node within a NodalModel.
        :param material_new: Material object or string containing the bulk properties (C_cap) of the node.
        :param coating_new: Coating object or string containing the absorptivity (alpha) and emissivity (epsilon) of the node.
        :param C_cap_new: Heat capacity [J/K]
        :param mass_new: Mass of the node [kg], only needed if a material is applied and no C_cap is given.
        :param volume_new: Volume of the node [m^3], only needed if a material is applied and no C_cap or mass is given.
        :param T0_new: Initial temperature [deg C] or [K]. Unit must be compatible with the unit of the NodalModel.
        :param q_ext_new: Tuple (q_pla, q_alb, q_s) of external heat fluxes [W/m^2] throughout time. Will be converted to
                          three separate Node attributes called Node.q_pla, Node.q_alb, Node.q_s.
        :param P_int_new: Internal power [W], throughout time.
        :param outer_new: Whether the node radiates freely to space or not. Default is False.
        :param alpha_new: Absorptivity in the solar spectrum. Default is 1.
        :param epsilon_new: Emissivity in the IR spectrum. Default is 1.
        :param area_new: Surface area [m^2] of the added node. Default is 1.
        :param origin_new: Tuple (x, y, z) of origin coordinates of the node. Default is (0, 0, 0).
        :param geom_new: Tuple (x_width, y_width, z_width) of the rectangular shape of the node. Default is (0, 0, 0).
        """
        # Converting the type of node to an identification/index number
        if isinstance(node_mod, str):
            node_mod = np.argwhere(self.name == node_mod)[0, 0]
        elif isinstance(node_mod, Node):
            node_mod = np.argwhere(self.name == node_mod.name)[0, 0]

        if name_new is not None:
            self.name[node_mod] = name_new
        if material_new is not None:
            if isinstance(material_new, Material) or (isinstance(material_new, str) and material_new in mat_lst):  # error for 'else' will be thrown in the Node.modify function
                if isinstance(material_new, str) and material_new in mat_lst:
                    material_new = mat_lst[material_new]
                if mass_new is not None:
                    self.C_cap[node_mod] = mass_new*material_new.c_cap
                elif self.nodes[node_mod].material is not None:  # previously assigned node mass still available
                    self.C_cap[node_mod] = self.nodes[node_mod].mass*material_new.c_cap
                elif volume_new is not None:
                    self.C_cap[node_mod] = mass_new*material_new.density*material_new.c_cap
                elif self.nodes[node_mod].volume is not None:
                    self.C_cap[node_mod] = self.nodes[node_mod].volume*material_new.density*material_new.c_cap
        if C_cap_new is not None:  # manually defined overrides any other calculated value
            self.C_cap[node_mod] = C_cap_new
        if coating_new is not None:
            if isinstance(coating_new, Coating) or (isinstance(coating_new, str) and coating_new in coat_lst):
                if isinstance(coating_new, str) and coating_new in coat_lst:
                    coating_new = coat_lst[coating_new]
                self.alpha[node_mod] = coating_new.alpha
                self.epsilon[node_mod] = coating_new.epsilon
            elif alpha_new is not None and epsilon_new is not None:
                self.alpha[node_mod] = alpha_new
                self.epsilon[node_mod] = alpha_new
                print(f"---------------WARNING---------------\n"
                      f"Coating assignment of node {self.name[node_mod]} has failed.\n"
                      f"However, the absorptivity and emissivity were set to the manually entered alpha and epsilon.\n"
                      f"If a coating is desired to be assigned, assure it is a 'Coating' object,\n"
                      f"or a string object that is a valid entry of the coat_lst dictionary.\n"
                      f"-------------------------------------\n")
                self.warnings = True
            else:
                print(f"---------------WARNING---------------\n"
                      f"Coating assignment of node {self.name[node_mod]} has failed.\n"
                      f"The coating assignment must be a 'Coating' object,\n"
                      f"or a string object that is a valid entry of the coat_lst dictionary.\n"
                      f"-------------------------------------\n")
                self.warnings = True
        if alpha_new is not None:  # manually defined overrides any other calculated value
            self.alpha[node_mod] = alpha_new
        if epsilon_new is not None:  # manually defined overrides any other calculated value
            self.epsilon[node_mod] = epsilon_new

        if T0_new is not None:
            self.T0[node_mod] = T0_new
            self.T[0, node_mod] = T0_new
        if q_ext_new is not None:
            self.q_pla[:, node_mod] = q_ext_new[0]
            self.q_alb[:, node_mod] = q_ext_new[1]
            self.q_s[:, node_mod] = q_ext_new[2]
            if not all(np.all(np.abs(x) < 1e-10) for x in q_ext_new) and outer_new is None:  # if q_ext is not zero
                # Only change outer if it is not given by the user.
                self.outer[node_mod] = True  # if the surface receives nonzero radiation, it must be an outer node
        if P_int_new is not None:
            self.P_int[:, node_mod] = P_int_new
        if outer_new is not None:
            self.outer[node_mod] = outer_new
        if origin_new is not None:
            self.origin[node_mod, :] = np.array(origin_new)
        if geom_new is not None:
            self.geom[node_mod, :] = np.array(geom_new)
            # Geometry overrides manually assigned area
            axis = np.argwhere(np.abs(geom_new) < 1e-10)[0, 0]  # axis of zero thickness
            if axis == 0:
                self.area[node_mod] = geom_new[1] * geom_new[2]
                if area_new is None:
                    print(f"---------------WARNING---------------\n"
                          f"The newly assigned geometry of node {self.name[node_mod]} has overridden the area as well.\n"
                          f"If this is not desired, assign a new area manually.\n"
                          f"-------------------------------------\n")
                    self.warnings = True
            elif axis == 1:
                self.area[node_mod] = geom_new[0] * geom_new[2]
                if area_new is None:
                    print(f"---------------WARNING---------------\n"
                          f"The newly assigned geometry of node {self.name[node_mod]} has overridden the area as well.\n"
                          f"If this is not desired, assign a new area manually.\n"
                          f"-------------------------------------\n")
                    self.warnings = True
            elif axis == 2:
                self.area[node_mod] = geom_new[0] * geom_new[1]
                if area_new is None:
                    print(f"---------------WARNING---------------\n"
                          f"The newly assigned geometry of node {self.name[node_mod]} has overridden the area as well.\n"
                          f"If this is not desired, assign a new area manually.\n"
                          f"-------------------------------------\n")
                    self.warnings = True
        if area_new is not None:
            self.area[node_mod] = area_new

        self.nodes[node_mod].modify(name_new=name_new, material_new=material_new, coating_new=coating_new,
                                    C_cap_new=C_cap_new, mass_new=mass_new, volume_new=volume_new, T0_new=T0_new,
                                    q_ext_new=q_ext_new, P_int_new=P_int_new, outer_new=outer_new, alpha_new=alpha_new,
                                    epsilon_new=epsilon_new, area_new=area_new, origin_new=origin_new, geom_new=geom_new)

        self.solved = False

    def solve(self, solver='Radau', printing=True, T0=None, limit_step=False, interp_inputs=False):
        """
        Solves the nodal model for the given time vector. The solution is stored in the NodalModel.T array.

        :param solver: Numerical integration method, must be either of: ['rk45', 'dop853', 'radau'].
                       Default is Radau.
        :param printing: Boolean indicating whether the progress should be printed in the console.
        :param T0: Starts the entire model at this T0. Must be the same unit (deg C or K) as the NodalModel.
                   If the parameter is not entered, the general rules apply (using Node.T0 or calculate steady state).
        :param limit_step: Boolean used for scipy's variable time stepping, indicating whether the solver can use its
                           optimisation algorithms to skip time steps and reduce computational time (limit_step=False),
                           or whether a higher accuracy is desired and no time steps are allowed to be skipped
                           (limit_step=True). Default is False.
        :param interp_inputs: Boolean indicating whether the environmental inputs (q_pla, q_alb, q_s, P_int) should
                              be interpolated during scipy's variable time stepping. Improves accuracy, but also
                              increases computational time. Default is False.
        """
        if self.solvable:
            if self.n != 0:
                solvers = ['rk45', 'dop853', 'radau']
                solver = solver.lower()
                if solver not in solvers:
                    print(f"----------------ERROR----------------\n"
                          f"Attempted solver '{solver}' is invalid.\n"
                          f"Try any of the following: {solvers}.\n"
                          f"-------------------------------------\n")
                    self.warnings = True
                else:
                    time0 = time.time()
                    if None in self.T0 or np.any([np.isnan(x) for x in self.T0]) and T0 is None:  # No initial temperature provided, so start with a steady-state analysis.
                        if printing:
                            print('Finding initial steady-state...')
                        self.T0 = np.zeros(self.T0.shape)+const_lst['kelvin']*(1-self.celsius)
                        self.T[0] = self.T0

                        def equilibrium_temp(T):
                            """
                            Wrapper for finding the root of the dT/dt function. Uses the average powers throughout the orbit.

                            :param T: Temperatures of all nodes in [K] or [deg C] through time. Is zero before the solver is run.
                            :return: Result of dT_dt function for the input T.
                            """
                            return dT_dt(0., T, const_lst, self.n, self.C_cap, self.C_con,
                                         np.mean(self.q_pla, axis=0)*np.ones_like(self.q_pla),
                                         np.mean(self.q_alb, axis=0)*np.ones_like(self.q_alb),
                                         np.mean(self.q_s, axis=0)*np.ones_like(self.q_s),
                                         np.mean(self.P_int, axis=0)*np.ones_like(self.P_int), self.outer, self.alpha,
                                         self.epsilon, self.area, self.celsius, self.rad, self.t, [0], False, False)

                        sol = root(equilibrium_temp, self.T0)
                        steady_state = sol.x
                        steady_reached = sol.success

                        if not steady_reached:  # if no steady-state is reached, the default setting for T = 0 deg C is kept.
                            if printing:
                                print('---------------WARNING---------------')
                                print(f'Steady-state was not reached.')
                                print(f'Initial temperatures will be set to 0 degrees Celsius.')
                                print('-------------------------------------\n')
                                self.warnings = True
                        else:
                            if printing:
                                print('Initial steady-state successfully found.\n')
                            self.T0 = steady_state
                            self.T[0] = self.T0
                    elif T0 is not None:  # if the T0 is desired to be overwritten for the whole model
                        self.T[0] = np.ones(self.T0.shape)*T0  # do not overwrite self.T0, only use the new T0 temporarily.
                    else:
                        self.T[0] = self.T0  # ensuring the correct T0 is used

                    self.heatflow = np.zeros((self.t.shape[0], len(self.connections)))

                    if printing:
                        print('Starting transient analysis...')
                    if limit_step:
                        max_dt = self.dt
                    else:
                        max_dt = np.inf
                    heatflow_con_temporary = np.zeros((self.t.shape[0], self.C_con.shape[0], self.C_con.shape[1]))
                    heatflow_rad_temporary = np.zeros((self.t.shape[0], self.rad.shape[0], self.rad.shape[1]))
                    cnt = [0]
                    if solver != 'radau':
                        solver = solver.upper()
                    else:
                        solver = 'Radau'  # Need correct writing style (first letter caps) of the method for scipy
                    self.T = solve_ivp(dT_dt, [0., self.t[-1]], self.T[0], method=solver, t_eval=self.t, max_step=max_dt,
                                       args=(const_lst, self.n, self.C_cap, self.C_con, self.q_pla, self.q_alb, self.q_s,
                                             self.P_int, self.outer, self.alpha, self.epsilon, self.area, self.celsius,
                                             self.rad, self.t, cnt, printing, interp_inputs)).y.T

                    for i in range(self.t.shape[0]):  # Re-compute the heat flows from the solution, to store them properly. Would be difficult without for-loop.
                        temporary = np.tile(self.T[i], (self.n, 1))  # temporary variable to compute all temperature differences between all nodes
                        dT_spatial = temporary - temporary.T  # all temperature differences between all nodes
                        temporary2 = np.tile(self.T[i] + const_lst['kelvin'] * self.celsius, (self.n, 1)) ** 4  # temporary variable to compute the (Ti^4-Tj^4) component for radiation.
                        dT4_spatial = temporary2 - temporary2.T  # differences between all T^4 of the nodes.
                        heatflow_con_temporary[i] = dT_spatial*self.C_con
                        heatflow_rad_temporary[i] = const_lst['stefan']*dT4_spatial*self.rad

                    for idx, conn in enumerate(self.connections):
                        node1 = np.argwhere(self.name == conn[0])[0, 0]
                        node2 = np.argwhere(self.name == conn[1])[0, 0]
                        for i in range(self.t.shape[0]-1):
                            if conn[2] == 'conductive':
                                self.heatflow[i+1, idx] = heatflow_con_temporary[i, node2, node1]  # from node 1 to node 2
                            else:  # radiative
                                self.heatflow[i+1, idx] = heatflow_rad_temporary[i, node2, node1]
                            if i == 1:
                                self.heatflow[0] = self.heatflow[1]  # since the first time stamp is not computed, approximate.
                    self.solved = True

                    if printing:
                        print('Transient analysis finished.')
                        print(f'Runtime: {time.time() - time0:.5f} seconds\n')
            else:
                print("----------------ERROR----------------\n"
                      "Model has zero nodes.\n"
                      "Add at least one node before running the solver.\n"
                      "-------------------------------------\n")
                self.warnings = True
        else:
            print("----------------ERROR----------------\n"
                  "Model has no time array assigned.\n"
                  "Add a time array with the function set_time() before running the solver.\n"
                  "-------------------------------------\n")
            self.warnings = True

    def show_plots(self, showrad=True, shownodes=True, showtemps=True, showflows=True, whichnodes=None):
        """
        Standard method to plot the temperatures, external radiation, heat flows, a 3d plot of the nodes, and a
        hierarchical tree of the NodalModel.

        :param showrad: Boolean whether the external heat fluxes are shown.
        :param shownodes: Boolean whether the 3d plot with nodes and hierarchical tree (printed in console) are shown.
        :param showtemps: Boolean whether the transient temperatures are shown.
        :param showflows: Boolean whether the conductive and radiative heat flows are shown.
        :param whichnodes: Node object, string, or integer indicating which nodes are to be shown in all plots. This
                           also means that only connections between the selected nodes are shown.
        """
        if whichnodes is None:
            whichnodes = range(self.n)
        else:
            whichnodes = ensure_list(whichnodes)
            for x in range(len(whichnodes)):
                if isinstance(whichnodes[x], str):
                    whichnodes[x] = np.argwhere(self.name == whichnodes[x])[0, 0]
                elif isinstance(whichnodes[x], Node):
                    whichnodes[x] = np.argwhere(self.name == whichnodes[x].name)[0, 0]
                elif not isinstance(whichnodes[x], int):
                    print(f"---------------WARNING---------------\n"
                          f"Selected nodes to plot were given in the wrong format.\n"
                          f"All nodes are shown.\n"
                          f"Enter a list with either the node number(s), name(s), or Node object(s).\n"
                          f"-------------------------------------\n")
                    self.warnings = True
                    whichnodes = range(self.n)

        if shownodes:  # plot geometry and thermal connections; as well as a NodalModel tree structure
            # 3D plot
            fig = plt.figure()
            fig.suptitle(f'Geometry of Nodal Model "{self.title}"')
            ax = fig.add_subplot(projection='3d')

            # plot plates & points
            for i in np.argwhere(~np.isnan(self.origin))[::3, 0]:  # for the nodes that have an origin assigned
                if i in whichnodes:
                    if np.any(~np.isnan(self.geom[i])):  # if the node also has geometry assigned
                        plot_plate(self.origin[i], self.geom[i], ax, i, self.name[i])
                    else:
                        plot_point(self.origin[i], ax, i, self.name[i])
            if np.any(np.isnan(self.origin)):
                print(f'----------------WARNING----------------\n'
                      f'Not all nodes could be visualised.\n'
                      f'Any nodes without a specified origin will not appear in the 3D plot with nodes.\n'
                      f'---------------------------------------\n')
                self.warnings = True

            # plot connections
            for i in np.unique(np.sort(np.argwhere(np.abs(self.rad[:]) > 1e-10), axis=1), axis=1):  # radiation
                if i[0] in whichnodes and i[1] in whichnodes:
                    plot_connections(self.origin[i[0]], self.origin[i[1]], ax, radiative=True)
            for i in np.unique(np.sort(np.argwhere(np.abs(self.C_con[:]) > 1e-10), axis=1), axis=1):  # conduction
                if i[0] in whichnodes and i[1] in whichnodes:
                    plot_connections(self.origin[i[0]], self.origin[i[1]], ax, radiative=False)

            custom_lines = [Line2D([0], [0], color='r'),
                            Line2D([0], [0], color='limegreen', linestyle=(0, (5, 5)))]
            ax.legend(custom_lines, ['Conductive', 'Radiative'])

            ax.axis('equal')
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('z')
            plt.tight_layout()

            # Tree plot
            show_tree(self)

        if showrad and self.solvable:
            # plot input heat flux
            plt.figure()
            plt.title(f'Heat Flow from Environment and Internal Power\nof Nodal Model "{self.title}"')
            plot_legend = False
            for i in whichnodes:
                if self.outer[i]:
                    plt.plot(self.t, self.q_pla[:, i]*self.area[i]*self.epsilon[i], label=f'node {i}, "{self.name[i]}": P_pla')
                    plt.plot(self.t, self.q_alb[:, i]*self.area[i]*self.alpha[i], label=f'node {i}, "{self.name[i]}": P_alb')
                    plt.plot(self.t, self.q_s[:, i]*self.area[i]*self.alpha[i], label=f'node {i}, "{self.name[i]}": P_s')
                    plot_legend = True
                if np.any(np.abs(self.P_int[:, i]) > 1e-10):
                    plt.plot(self.t, self.P_int[:, i], label=f'node {i}, "{self.name[i]}": P_int')
                    plot_legend = True
            if plot_legend:
                plt.legend()
            plt.grid()
            plt.xlabel('Time [s]')
            plt.ylabel('Power Into Node [W]')
        elif showrad and not self.solvable:
            print(f'----------------WARNING----------------\n'
                  f'External radiation data could not be shown because\n'
                  f'the NodalModel does not have a time array assigned.\n'
                  f'Add a time array with the function set_time() before running the solver.\n'
                  f'---------------------------------------\n')
            self.warnings = True

        # plot transient temperature results
        if showtemps or showflows:
            if self.solved and self.solvable:
                if showtemps:
                    # plot temperatures
                    plt.figure()
                    plt.title(f'Temperatures of the Nodal Model "{self.title}"')
                    if self.n == 1:
                        plt.plot(self.t, self.T, label=f'node 0')
                    else:
                        for i in whichnodes:
                            plt.plot(self.t, self.T[:, i], label=f'node {i}, "{self.name[i]}"')
                    plt.legend()
                    plt.grid()
                    plt.xlabel('Time [s]')
                    if self.celsius:
                        plt.ylabel(r'Temperature [$\degree$C]')
                    else:
                        plt.ylabel('Temperature [K]')
                if showflows:
                    # plot power flow between all nodes
                    fig_con = plt.figure()
                    fig_con.suptitle(f'Internal Conductive Heat Flows\nof the Nodal Model "{self.title}"')
                    ax_con = fig_con.add_subplot()
                    fig_rad = plt.figure()
                    fig_rad.suptitle(f'Internal Radiative Heat Flows\nof the Nodal Model "{self.title}"')
                    ax_rad = fig_rad.add_subplot()
                    plot_legend_con = False
                    plot_legend_rad = False
                    for j, conn in enumerate(self.connections):
                        node1 = np.argwhere(self.name == conn[0])[0, 0]
                        node2 = np.argwhere(self.name == conn[1])[0, 0]
                        if node1 in whichnodes and node2 in whichnodes:
                            if conn[2] == 'conductive':
                                ax_con.plot(self.t, self.heatflow[:, j],
                                            label=f'nodes {node1}'+r'$\rightarrow$'+f'{node2}, ("{self.name[node1]}"'+
                                                  r'$\rightarrow$'+f'"{self.name[node2]}")')
                                plot_legend_con = True
                            else:  # radiative
                                ax_rad.plot(self.t, self.heatflow[:, j],
                                            label=f'nodes {node1}'+r'$\rightarrow$'+f'{node2}, ("{self.name[node1]}"'+
                                                  r'$\rightarrow$'+f'"{self.name[node2]}")')
                                plot_legend_rad = True
                    if plot_legend_con:
                        ax_con.legend()
                    if plot_legend_rad:
                        ax_rad.legend()
                    ax_con.grid()
                    ax_rad.grid()
                    ax_con.set_xlabel('Time [s]')
                    ax_rad.set_xlabel('Time [s]')
                    ax_con.set_ylabel('Conductive Power From Node i to j [W]')
                    ax_rad.set_ylabel('Radiative Power From Node i to j [W]')
            elif not self.solved:
                print("---------------WARNING---------------\n"
                      "Simulation has not been run yet, no temperature/heat flow plots were shown.\n"
                      "Run the solve() function at least once to generate data.\n"
                      "-------------------------------------\n")
                self.warnings = True
            else:
                print(f'----------------WARNING----------------\n'
                      f'Temperature/heat flow data could not be shown because\n'
                      f'the NodalModel does not have a time array assigned.\n'
                      f'Add a time array with the function set_time() before running the solver.\n'
                      f'---------------------------------------\n')
                self.warnings = True

        if self.warnings:
            print("-------------------------!ATTENTION!----------------------------"
                  "\nThere are errors/warnings to view in the console.\n"
                  "Always view the entire console to not miss any errors/warnings.\n"
                  "----------------------------------------------------------------")

        plt.show()


class OrbitalModel:
    """
    The OrbitalModel class is an object that defines an orbit, its discretisation,
    and any (angled) plates that receive radiation. It can then calculate the planet IR, albedo, and solar fluxes.
    """
    def __init__(self, h, surfaces, beta=None, RAAN=None, incl=None, day=1, n_orbits=1., dtheta=None, dt=None,
                 angular_rates=None):
        """
        Initialises an OrbitalModel object. Beta angle can be specified directly, or via the RAAN and inclination.
        If surfaces is/contains a Node object, its property Node.name must include the direction, so for example
        'x+' would work, but also 'x+01' would work.

        :param h: Orbital altitude [m].
        :param surfaces: Either a (list of) Node object(s), or string with direction ('x+' or 'y-' etc.),
         or a tuple/list with the tau and phi spherical angles. If it is a Node object, its property Node.name must
         include the direction, so for example 'x+' would work, but also 'x+01' would also work.
        :param beta: Solar declination [deg]. MUST BE DEGREES, will be converted to radians.
        :param RAAN: Right ascension of the ascending node [deg] of the satellite's orbit. MUST BE DEGREES.
        :param incl: Inclination [deg] of the satellite's orbit with respect to the Earth's equator. MUST BE DEGREES.
        :param day: Number of days from January 1 (January 1 itself is day 1).
        :param n_orbits: Number of orbits to be calculated. Default is 1.
        :param dtheta: Step size [deg] in true anomaly (theta). MUST BE DEGREES, will be converted to radians.
                       Default is 0.01 rad.
        :param dt: Step size [s] in time. Can be entered instead of dtheta. If both are given, dtheta overrides dt.
        :param angular_rates: Angular velocities [deg/s] around the x, y, and z axes. MUST BE DEGREES / S.
        """
        # RAAN & inclination converted to beta angle, but this RAAN & incl is used in case the modify function is used
        # to change the orbit.
        if RAAN is None:
            self.RAAN = RAAN
        else:
            self.RAAN = RAAN*np.pi/180.
        if incl is None:
            self.incl = incl
        else:
            self.incl = incl*np.pi/180.
        self.h = h
        keep_entity = False  # variable to distinguish tuples with objects and tuples with tau and phi angles
        if isinstance(surfaces, tuple):
            for i in surfaces:
                if isinstance(i, numbers.Number) and len(surfaces) == 2:  # then it is a tuple with tau & phi
                    if np.isfinite(i):  # np.nan is also considered a number so add an extra check (beware: isfinite does not work with None)
                        keep_entity = True
        self.surfaces = ensure_list(surfaces, keep_entity=keep_entity)
        self.n_orbits = n_orbits
        self.day = day

        if dtheta is None:
            if dt is None:
                self.dtheta = 0.01  # [rad]
                self.dt = theta_to_t(self.h, self.dtheta)
            else:
                self.dt = dt
                self.dtheta = t_to_theta(self.h, self.dt)
        else:
            # overwrite the time step (if given) by the step in true anomaly
            self.dtheta = dtheta*np.pi/180.
            self.dt = theta_to_t(self.h, self.dtheta)
        self.theta = np.arange(0., self.n_orbits*2*np.pi+self.dtheta, self.dtheta)
        self.t = theta_to_t(self.h, self.theta)

        if beta is None:
            if RAAN is not None and incl is not None:
                self.beta = beta_angle(self.day, RAAN * np.pi / 180., incl * np.pi / 180.)
            else:
                self.beta = 0.  # standard beta angle assumed zero
        else:
            self.beta = beta*np.pi/180.

        self.q_pla = np.zeros((self.t.shape[0], len(self.surfaces)))
        self.q_alb = np.zeros((self.t.shape[0], len(self.surfaces)))
        self.q_s = np.zeros((self.t.shape[0], len(self.surfaces)))

        if angular_rates is None:
            self.angular_rates = angular_rates
        else:
            self.angular_rates = ensure_list(angular_rates)
            for k in range(len(self.angular_rates)):
                self.angular_rates[k] *= np.pi/180.

        self.attitude = np.zeros((self.t.shape[0], len(self.surfaces), 3))
        self.rotation_obj = None
        self.names = []
        for j, surf in enumerate(self.surfaces):
            if isinstance(surf, Node):
                self.names.append(surf.name)
            elif isinstance(surf, str):
                self.names.append(surf)
            else:
                self.names.append(j)

    def modify(self, h_new=None, surfaces_new=None, beta_new=None, RAAN_new=None, incl_new=None, day_new=None,
               n_orbits_new=None, dtheta_new=None, dt_new=None, angular_rates_new=None):
        """
        Adapt values of the OrbitalModel object. Beware that using this function (particularly surfaces_new, dtheta_new,
        or dt_new) will reset previously computed outputs, since the shape of the heat flux array must change to
        accommodate the changes (hence, the previous results are not discarded).

        :param h_new: Orbital altitude [m].
        :param surfaces_new: Either a (list of) Node object(s), or string with direction ('x+' or 'y-' etc.),
                             or a tuple/list with the tau and phi spherical angles [deg]. If it is a Node object, its
                             property Node.name must include the direction, so for example 'x+' would work, but also
                             'x+01' would also work.
        :param beta_new: Solar declination [deg]. MUST BE DEGREES, will be converted to radians.
        :param RAAN_new: Right ascension of the ascending node [deg] of the satellite's orbit. MUST BE DEGREES.
        :param incl_new: Inclination [deg] of the satellite's orbit with respect to the Earth's equator. MUST BE DEGREES.
        :param day_new: Number of days from January 1 (January 1 itself is day 1).
        :param n_orbits_new: Number of orbits to be calculated. Default is 1.
        :param dtheta_new: Step size [deg] in true anomaly (theta). MUST BE DEGREES, will be converted to radians.
        :param dt_new: Step size [s] in time. Can be entered instead of dtheta. If both are given, dtheta overrides dt.
        :param angular_rates_new: Angular velocities [deg/s] around the x, y, and z axes. MUST BE DEGREES / S.
        """
        time_changed = False
        surf_changed = False
        ang_rates_changed = False
        alt_changed = False
        if h_new is not None:
            alt_changed = True
            self.h = h_new
            self.dtheta = t_to_theta(self.h, self.dt)  # keep dt the same; probably most likely that the user set dt, not dtheta
            self.theta = np.arange(0., self.n_orbits*2*np.pi+self.dtheta, self.dtheta)
            self.t = theta_to_t(self.h, self.theta)
        if surfaces_new is not None:
            surf_changed = True
            keep_entity = False  # variable to distinguish tuples with objects and tuples with tau and phi angles
            if isinstance(surfaces_new, tuple):
                for i in surfaces_new:
                    if isinstance(i, numbers.Number) and len(surfaces_new) == 2:  # then it is a tuple with tau & phi
                        if np.isfinite(i):  # np.nan is also considered a number so add an extra check (beware: isfinite does not work with None)
                            keep_entity = True
            self.surfaces = ensure_list(surfaces_new, keep_entity=keep_entity)
        if n_orbits_new is not None:
            time_changed = True
            self.n_orbits = n_orbits_new
            self.theta = np.arange(0., self.n_orbits*2*np.pi+self.dtheta, self.dtheta)
            self.t = theta_to_t(self.h, self.theta)
        if day_new is not None:
            self.day = day_new
            if RAAN_new is not None and incl_new is not None:
                self.beta = beta_angle(self.day, RAAN_new*np.pi/180., incl_new*np.pi/180.)
            elif RAAN_new is None and self.RAAN is not None and incl_new is not None:
                self.beta = beta_angle(self.day, self.RAAN, incl_new*np.pi/180.)
            elif incl_new is None and self.incl is not None and RAAN_new is not None:
                self.beta = beta_angle(self.day, RAAN_new*np.pi/180., self.incl)
            elif self.RAAN is not None and self.incl is not None:
                self.beta = beta_angle(self.day, self.RAAN, self.incl)
            else:
                print("---------------WARNING---------------\n"
                      "The day in the year was changed, but since there are no RAAN and/or inclination available,\n"
                      "the beta angle remains the same as before, or it will take the new value\n"
                      "if any is given in the modify(beta_new=...) function.\n"
                      "-------------------------------------\n")

        if dtheta_new is not None or dt_new is not None:
            time_changed = True
            if dtheta_new is None:
                self.dt = dt_new
                self.dtheta = t_to_theta(self.h, self.dt)
            else:
                # overwrite the time step (if given) by the step in true anomaly
                self.dtheta = dtheta_new*np.pi/180.
                self.dt = theta_to_t(self.h, self.dtheta)
            self.theta = np.arange(0., self.n_orbits*2*np.pi+self.dtheta, self.dtheta)
            self.t = theta_to_t(self.h, self.theta)

        if beta_new is not None:
            self.beta = beta_new*np.pi/180.
        elif RAAN_new is not None and incl_new is not None:
            self.beta = beta_angle(self.day, RAAN_new*np.pi/180., incl_new*np.pi/180.)

        if time_changed or surf_changed or alt_changed:  # must discard old data to renew these arrays
            self.q_pla = np.zeros((self.t.shape[0], len(self.surfaces)))
            self.q_alb = np.zeros((self.t.shape[0], len(self.surfaces)))
            self.q_s = np.zeros((self.t.shape[0], len(self.surfaces)))

        if angular_rates_new is not None:
            ang_rates_changed = True
            self.angular_rates = ensure_list(angular_rates_new)
            for k in range(len(self.angular_rates)):
                self.angular_rates[k] *= np.pi/180.

        if ang_rates_changed or time_changed or surf_changed:
            self.attitude = np.zeros((self.t.shape[0], len(self.surfaces), 3))
            self.rotation_obj = None

        if surf_changed:
            self.names = []
            for j, surf in enumerate(self.surfaces):
                if isinstance(surf, Node):
                    self.names.append(surf.name)
                elif isinstance(surf, str):
                    self.names.append(surf)
                else:
                    self.names.append(j)

    def animate_attitude(self, speed=1, realtime=False):
        """
        Shows an animation (previous plotting windows must be closed) of a line or collection of lines that are rotating
        throughout time.

        :param speed: Default is 1, showing all time frames. If it is increased, time frames are skipped to speed up
                      the animation.
        :param realtime: If True, the duration between each time step is set to the dt of the simulation.
        """
        if self.rotation_obj is not None:
            fig1 = plt.figure()
            ax1 = fig1.add_subplot(projection='3d')
            ax1.set_xlabel('x')
            ax1.set_ylabel('y')
            ax1.set_zlabel('z')

            timestep = 0
            N_steps = self.t.shape[0]

            rot_all = Rotation.from_euler('xyz', [0., -90., 0.], degrees=True)  # make nadir downwards
            while plt.fignum_exists(fig1.number) and timestep < N_steps:  # Use fignum to be able to close window normally
                if timestep % speed == 0:
                    ax1.clear()
                    hours = (timestep * self.dt % (24 * 3600)) // 3600
                    minutes = (timestep * self.dt % 3600) // 60
                    seconds = (timestep * self.dt % 60)
                    plt.suptitle(f'Time passed: {hours:.0f} hrs, {minutes:.0f} mins, {seconds:.2f} seconds\n'
                                 f'Orbit angle (true anomaly from solar vector): '
                                 f'{t_to_theta(408e3, timestep * self.dt) * 180 / np.pi:.1f} degrees')

                    unit = np.eye(3)
                    unit[0] = rot_all.apply(unit[0])
                    unit[1] = rot_all.apply(unit[1])
                    unit[2] = rot_all.apply(unit[2])
                    plot3dframe(unit, ax1, 'firebrick', 'g', 'k', lbl1='Zenith', lbl2='Orbit Normal',
                                lbl3='Velocity', style='dashed')

                    eigenvec = self.rotation_obj.as_rotvec()
                    eigenvec /= (np.sqrt(np.sum(eigenvec ** 2)) * 2)  # ensure length 0.5
                    plot3dline(np.zeros(3), rot_all.apply(eigenvec), ax1, colour='magenta', lbl='Eigenaxis',
                               style='dashdot')

                    colours = ['red', 'limegreen', 'royalblue', 'orange', 'turquoise', 'gold', 'darkviolet', 'deeppink',
                               'salmon', 'deepskyblue', 'grey', 'pink', 'mediumpurple']
                    for surf in range(len(self.surfaces)):
                        c = colours[surf % len(colours)]
                        plot3dline(np.zeros(3), rot_all.apply(self.attitude[timestep][surf]), ax1, colour=c,
                                   lbl=self.names[surf], ordr=0)

                    ax1.axes.set_xlim3d(left=-1, right=1)
                    ax1.axes.set_ylim3d(bottom=-1, top=1)
                    ax1.axes.set_zlim3d(bottom=-1, top=1)
                    ax1.legend(loc='upper left')

                    ax1.tick_params(colors='white')
                    ax1.tick_params(colors='white')
                    ax1.tick_params(colors='white')

                    ax1.set_aspect('equal')

                    if realtime:
                        plt.pause(self.dt)
                    else:
                        plt.pause(0.1)  # with a really fast computer, this can be set to dt for real-time
                timestep += 1
            plt.show(block=True)
        else:
            print("----------------ERROR----------------\n"
                  "OrbitalModel has not yet been computed with angular rates.\n"
                  "Assign a nonzero angular_rates argument to the OrbitalModel and run the .compute() command.\n"
                  "-------------------------------------\n")

    def compute(self, printing=True):
        """
        Run the orbital model (func heat_received) and store the results in self.q_pla, self.q_alb, and self.q_s. If the
        surfaces are of type Node, the heat fluxes will directly be assigned to the nodes. However, if those nodes are
        already in a NodalModel, the heat fluxes must also be assigned to the NodalModel with:
        NodalModel.modify_node(node, q_ext_new=OrbitalModel.get_heat(node)), or alternatively:
        NodalModel.modify_node(node, q_ext_new=tuple(node.q_pla, node.q_alb, node.q_s))

        :param printing: Boolean indicating whether the progress should be printed in the console.
        """
        if printing:
            print('Starting orbital analysis...')
        start_tauphi = np.zeros((len(self.surfaces), 2))
        start_cartesian = np.zeros((len(self.surfaces), 3))
        cnt = 0
        for i, surface in enumerate(self.surfaces):
            if isinstance(surface, str):
                tau, phi = tau_phi(surface)
            elif isinstance(surface, Node):
                direction = ''
                names_ = ['x+', 'x-', 'y+', 'y-', 'z+', 'z-']
                for name_ in names_:
                    if name_ in surface.name:  # Node name must include either of names_
                        direction = name_
                if direction == '':
                    surface.warnings = True
                tau, phi = tau_phi(direction)
            else:
                tau = surface[0]*np.pi/180.
                phi = surface[1]*np.pi/180.
            start_tauphi[i] = np.array([tau, phi])
            start_cartesian[i] = spherical_to_cartesian((tau, phi))
            self.attitude[0][i] = start_cartesian[i]

            if self.angular_rates is not None and not np.all(np.abs(self.angular_rates) < 1e-10):
                rot = Rotation.from_euler('xyz', np.array(self.angular_rates)*self.dt)
                self.rotation_obj = rot
                tau = np.ones(self.t.shape[0])*tau  # also already sets the first entry
                phi = np.ones(self.t.shape[0])*phi  # also already sets the first entry
                for timestep in range(1, self.t.shape[0]):  # first timestep is already known.
                    self.attitude[timestep][i] = rot.apply(self.attitude[timestep-1][i])
                    tau[timestep], phi[timestep] = cartesian_to_spherical(self.attitude[timestep][i])

            q_pla, q_alb, q_s = heat_received(self.day, self.beta, self.theta, self.h, tau, phi)
            self.q_pla[:, i] = q_pla
            self.q_alb[:, i] = q_alb
            self.q_s[:, i] = q_s
            if isinstance(surface, Node):
                surface.modify(q_ext_new=(q_pla, q_alb, q_s))

            if len(self.surfaces) > 1:
                progress = i / (len(self.surfaces) - 1) * 100
                cnt += 1 / (len(self.surfaces) - 1) * 100
                if cnt > 10:
                    if printing:
                        print(f'{progress:.0f}%')
                    cnt = 0
        if printing:
            print('Orbital analysis finished.\n')

    def get_heat(self, index):
        """
        Return the planet, albedo, and solar flux as a tuple, for the given surface.

        :param index: Surface given as an integer (order in the list self.surfaces), string, or Node.
        :return: Tuple (q_pla, q_alb, q_s) in [W/m^2].
        """
        if isinstance(index, int):
            return tuple((self.q_pla[:, index], self.q_alb[:, index], self.q_s[:, index]))
        elif isinstance(index, str):
            index = index.lower()
            if all(isinstance(surf, str) for surf in self.surfaces):
                idx = np.argwhere(np.array(self.surfaces) == index)[0, 0]
                return tuple((self.q_pla[:, idx], self.q_alb[:, idx], self.q_s[:, idx]))
            elif all(isinstance(surf, Node) for surf in self.surfaces):
                names = np.array([x.name for x in self.surfaces])
                idx = np.argwhere(names == index)[0, 0]
                return tuple((self.q_pla[:, idx], self.q_alb[:, idx], self.q_s[:, idx]))
            else:
                print("----------------ERROR----------------\n"
                      "Given index in get_heat function cannot be used,\n"
                      "since the surfaces in the object have no names assigned.\n"
                      "The heat values were returned as zero.\n"
                      "-------------------------------------\n")
                return tuple((np.zeros(self.q_pla[:, 0].shape), np.zeros(self.q_alb[:, 0].shape),
                              np.zeros(self.q_s[:, 0].shape)))
        elif isinstance(index, Node):
            if all(isinstance(surf, Node) for surf in self.surfaces):
                idx = np.argwhere(np.array(self.surfaces) == index)[0, 0]
                return tuple((self.q_pla[:, idx], self.q_alb[:, idx], self.q_s[:, idx]))
            elif all(isinstance(surf, str) for surf in self.surfaces):
                idx = np.argwhere(np.array(self.surfaces) == index.name)[0, 0]
                return tuple((self.q_pla[:, idx], self.q_alb[:, idx], self.q_s[:, idx]))
            else:
                print("----------------ERROR----------------\n"
                      "Given index in get_heat function cannot be used,\n"
                      "since the surfaces in the object are not of type Node.\n"
                      "The heat values were returned as zero.\n"
                      "-------------------------------------\n")
                return tuple((np.zeros(self.q_pla[:, 0].shape), np.zeros(self.q_alb[:, 0].shape),
                              np.zeros(self.q_s[:, 0].shape)))

