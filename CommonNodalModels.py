# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 16:19:54 2024
Author: Frank Meijering (Delft University of Technology)

CommonNodalModels.py is the beginning of a collection of NodalModels. The modularity of the code allows for models to
be created and copy-pasted into a new model. Hence, a file such as this one can contain commonly used nodal models (such
as a PCB), which can then be used in other models. Currently, there are functions to make a 5- or 9-node PCB model, but
alternatively, the code at the bottom of this file can be simply copy-pasted and adapted as desired.
"""


import numpy as np
from ThermalBudget import Node, NodalModel, ensure_list


# --------------------- FUNCTIONS TO GENERATE PRE-DEFINED NODALMODELS --------------------------
def assign_q_ext_to_pcb(nodal_models, orbit_obj):
    """
    Since a PCB has multiple nodes, it can be tedious to assign environmental heat fluxes to all of them.
    This function does this automatically.

    :param nodal_models: either one PCB (NodalModel) or a list of PCBs (list of NodalModels). The nodes that include
                         either of 'x+', 'y+', ... are the ones given the heat flux.
                         E.g., the middle PCB node is 'PCBx+A', then 'PCBx+B', etc.
    :param orbit_obj: OrbitalModel object that has been computed already (hence contains heat fluxes).
    """
    nodal_models = ensure_list(nodal_models)
    directions = ['x+', 'y+', 'z+', 'x-', 'y-', 'z-']
    for model in nodal_models:
        for node in model.nodes:
            idx = ''
            for direc in directions:
                if direc in node.name:
                    idx = direc
            if idx != '':
                model.modify_node(node, q_ext_new=orbit_obj.get_heat(idx))


def make_5node_pcb(title, mass, origin, geom=(0.1, 0.1, 0.), thickness=0.0016, t=None, power=None, material='PCB',
                   coating='PCB_green', outer=False):
    """
    Make a 5-node PCB (1 central node + 4 corner nodes).
    Ordering of the nodes is first the centre node (A), then counterclockwise starting on the top right (B, C, D, E).

    For outer nodes, the received environment heat is spread out over the nodes, proportional to their area.
    For inner nodes, applied power is concentrated in the centre node A. For internal radiation, node A is assumed to
    be the only radiating face, but representing the entire PCB area. This is ensured automatically.

    :param title: Title/name that the NodalModel has.
    :param mass: Mass in kg of the PCB.
    :param origin: Tuple (x, y, z) of origin coordinates of the node.
    :param geom: Tuple (x_width, y_width, z_width) of the rectangular shape of the node, in metres.
    :param thickness: Thickness [m] of the PCB. Default is 0.0016 m (1.6 mm).
    :param t: Time array [s] of the entire simulation. It is recommended to use OrbitalModel.t as the time array.
    :param power: Power (heat) dissipation in W of the PCB. Can be a single value or an array of values throughout time,
                  as long as it is compatible with the OrbitalModel time array.
    :param material: Material object or string containing the bulk properties (C_cap) of the node.
    :param coating: Coating object or string containing the absorptivity (alpha) and emissivity (epsilon) of the node.
    :param outer: Whether the node radiates freely to space or not. Default is False.
    :return: NodalModel of a 5-node PCB, and all individual Node objects as well.
    """
    hor_ver_pos = [[1, 1], [-1, 1], [-1, -1], [1, -1]]  # horizontal & vertical position of the nodes B-I
    axis = np.argwhere(np.abs(geom) < 1e-10)[0, 0]  # axis of zero thickness
    if axis == 0:  # x
        for i in hor_ver_pos:
            i.insert(0, 0)
        xyz = np.array(hor_ver_pos)  # xyz positions for nodes B-I
        idx = [1, 2]  # indices of the nonzero edges
        areatot = geom[idx[0]]*geom[idx[1]]  # area of the PCB (will be equally distributed over the nodes)
    elif axis == 1:  # y
        for i in hor_ver_pos:
            i.insert(1, 0)
        xyz = np.array(hor_ver_pos)  # xyz positions for nodes B-I
        idx = [0, 2]
        areatot = geom[idx[0]]*geom[idx[1]]
    else:  # axis = 2 (z)
        for i in hor_ver_pos:
            i.append(0)
        xyz = np.array(hor_ver_pos)  # xyz positions for nodes B-I
        idx = [0, 1]
        areatot = geom[idx[0]]*geom[idx[1]]

    PCB5 = NodalModel(t=t, title=title)
    PCB5A = Node(name=f'{title}A', material=material, coating=coating, mass=mass/2, origin=origin, geom=geom,
                 P_int=power, outer=outer, area=areatot/2*(2-outer))  # area represents the whole PCB area if outer is False. Overwrites the area computed from geom (this is intentional).
    PCB5B = Node(name=f'{title}B', material=material, coating=coating, mass=mass/8,
                 origin=tuple(np.array(origin)+np.array(geom)*0.4*xyz[0]), area=areatot/8*outer, outer=outer)  # origin is only assigned such that the node can be shown in the 3D plot.
    PCB5C = Node(name=f'{title}C', material=material, coating=coating, mass=mass/8,
                 origin=tuple(np.array(origin)+np.array(geom)*0.4*xyz[1]), area=areatot/8*outer, outer=outer)  # origin is only assigned such that the node can be shown in the 3D plot.
    PCB5D = Node(name=f'{title}D', material=material, coating=coating, mass=mass/8,
                 origin=tuple(np.array(origin)+np.array(geom)*0.4*xyz[2]), area=areatot/8*outer, outer=outer)  # origin is only assigned such that the node can be shown in the 3D plot.
    PCB5E = Node(name=f'{title}E', material=material, coating=coating, mass=mass/8,
                 origin=tuple(np.array(origin)+np.array(geom)*0.4*xyz[3]), area=areatot/8*outer, outer=outer)  # origin is only assigned such that the node can be shown in the 3D plot.
    PCB5.add_node([PCB5A, PCB5B, PCB5C, PCB5D, PCB5E])
    PCB5.connect(PCB5A, [PCB5B, PCB5C, PCB5D, PCB5E], L_through=3/4*np.sqrt((geom[idx[0]]/2)**2+(geom[idx[1]]/2)**2),
                 A=thickness*np.sqrt((geom[idx[0]]/2)**2+(geom[idx[1]]/2)**2))
    return PCB5, PCB5A, PCB5B, PCB5C, PCB5D, PCB5E


def make_9node_pcb(title, mass, origin, geom=(0.1, 0.1, 0.), thickness=0.0016, t=None, power=None, material='PCB',
                   coating='PCB_green', outer=False):
    """
    Make a 9-node PCB (3x3 grid).
    Ordering of the nodes is first the centre node, then counterclockwise starting at the centre right node.

    For outer nodes, the received environment heat is spread out over the nodes, proportional to their area.
    For inner nodes, applied power is concentrated in the centre node A. For internal radiation, node A is assumed to
    be the only radiating face, but representing the entire PCB area. This is ensured automatically.

    :param title: Title/name that the NodalModel has.
    :param mass: Mass in kg of the PCB.
    :param origin: Tuple (x, y, z) of origin coordinates of the node.
    :param geom: Tuple (x_width, y_width, z_width) of the rectangular shape of the node, in metres.
    :param thickness: Thickness [m] of the PCB. Default is 0.0016 m (1.6 mm).
    :param t: Time array [s] of the entire simulation. It is recommended to use OrbitalModel.t as the time array.
    :param power: Power (heat) dissipation in W of the PCB. Can be a single value or an array of values throughout time,
                  as long as it is compatible with the OrbitalModel time array.
    :param material: Material object or string containing the bulk properties (C_cap) of the node.
    :param coating: Coating object or string containing the absorptivity (alpha) and emissivity (epsilon) of the node.
    :param outer: Whether the node radiates freely to space or not. Default is False.
    :return: NodalModel of a 9-node PCB, and all individual Node objects as well.
    """
    hor_ver_pos = [[1, 0], [1, 1], [0, 1], [-1, 1], [-1, 0], [-1, -1], [0, -1], [1, -1]]  # horizontal & vertical position of the nodes B-I
    axis = np.argwhere(np.abs(geom) < 1e-10)[0, 0]  # axis of zero thickness
    if axis == 0:  # x
        for i in hor_ver_pos:
            i.insert(0, 0)
        xyz = np.array(hor_ver_pos)  # xyz positions for nodes B-I
        idx = [1, 2]  # indices of the nonzero edges
        areatot = geom[idx[0]]*geom[idx[1]]  # area of the PCB (will be equally distributed over the nodes)
    elif axis == 1:  # y
        for i in hor_ver_pos:
            i.insert(1, 0)
        xyz = np.array(hor_ver_pos)
        idx = [0, 2]
        areatot = geom[idx[0]]*geom[idx[1]]  # area of the PCB (will be equally distributed over the nodes)
    else:  # axis = 2 (z)
        for i in hor_ver_pos:
            i.append(0)
        xyz = np.array(hor_ver_pos)
        idx = [0, 1]
        areatot = geom[idx[0]]*geom[idx[1]]  # area of the PCB (will be equally distributed over the nodes)

    PCB9 = NodalModel(t, title=title)
    PCB9A = Node(name=f'{title}A', material=material, coating=coating, mass=mass/9, origin=origin, geom=geom,
                 P_int=power, outer=outer, area=areatot/9*(9-8*outer))  # area represents the whole PCB area if outer is False. Overwrites the area computed from geom (this is intentional).
    PCB9B = Node(name=f'{title}B', material=material, coating=coating, mass=mass/9,
                 origin=tuple(np.array(origin)+np.array(geom)/3*xyz[0]), area=areatot/9*outer, outer=outer)  # origin is only assigned such that the node can be shown in the 3D plot.
    PCB9C = Node(name=f'{title}C', material=material, coating=coating, mass=mass/9,
                 origin=tuple(np.array(origin)+np.array(geom)/3*xyz[1]), area=areatot/9*outer, outer=outer)  # origin is only assigned such that the node can be shown in the 3D plot.
    PCB9D = Node(name=f'{title}D', material=material, coating=coating, mass=mass/9,
                 origin=tuple(np.array(origin)+np.array(geom)/3*xyz[2]), area=areatot/9*outer, outer=outer)  # origin is only assigned such that the node can be shown in the 3D plot.
    PCB9E = Node(name=f'{title}E', material=material, coating=coating, mass=mass/9,
                 origin=tuple(np.array(origin)+np.array(geom)/3*xyz[3]), area=areatot/9*outer, outer=outer)  # origin is only assigned such that the node can be shown in the 3D plot.
    PCB9F = Node(name=f'{title}F', material=material, coating=coating, mass=mass/9,
                 origin=tuple(np.array(origin)+np.array(geom)/3*xyz[4]), area=areatot/9*outer, outer=outer)  # origin is only assigned such that the node can be shown in the 3D plot.
    PCB9G = Node(name=f'{title}G', material=material, coating=coating, mass=mass/9,
                 origin=tuple(np.array(origin)+np.array(geom)/3*xyz[5]), area=areatot/9*outer, outer=outer)  # origin is only assigned such that the node can be shown in the 3D plot.
    PCB9H = Node(name=f'{title}H', material=material, coating=coating, mass=mass/9,
                 origin=tuple(np.array(origin)+np.array(geom)/3*xyz[6]), area=areatot/9*outer, outer=outer)  # origin is only assigned such that the node can be shown in the 3D plot.
    PCB9I = Node(name=f'{title}I', material=material, coating=coating, mass=mass/9,
                 origin=tuple(np.array(origin)+np.array(geom)/3*xyz[7]), area=areatot/9*outer, outer=outer)  # origin is only assigned such that the node can be shown in the 3D plot.
    PCB9.add_node([PCB9A, PCB9B, PCB9C, PCB9D, PCB9E, PCB9F, PCB9G, PCB9H, PCB9I])
    PCB9.connect(PCB9A, [PCB9B, PCB9F], L_through=geom[idx[0]]/3, A=thickness*geom[idx[0]]/3)
    PCB9.connect(PCB9A, [PCB9D, PCB9H], L_through=geom[idx[1]]/3, A=thickness*geom[idx[1]]/3)
    PCB9.connect(PCB9B, [PCB9C, PCB9I], L_through=geom[idx[1]]/3, A=thickness*geom[idx[1]]/3)
    PCB9.connect(PCB9D, [PCB9C, PCB9E], L_through=geom[idx[0]]/3, A=thickness*geom[idx[0]]/3)
    PCB9.connect(PCB9F, [PCB9E, PCB9G], L_through=geom[idx[1]]/3, A=thickness*geom[idx[1]]/3)
    PCB9.connect(PCB9H, [PCB9G, PCB9I], L_through=geom[idx[0]]/3, A=thickness*geom[idx[0]]/3)
    return PCB9, PCB9A, PCB9B, PCB9C, PCB9D, PCB9E, PCB9F, PCB9G, PCB9H, PCB9I


# --------------------- NODALMODEL TEMPLATES TO COPY-PASTE DIRECTLY --------------------------
# 5-node PCB (this example has the surface normal in Z. Change geom and origin if the PCB has a different orientation.
# This example is 10x10 cm.
# This example is an outer node, so the incoming heat must be spread out over all nodes. Hence, the area is distributed.
# Must assign a time array later (PCB5.set_time({OrbitalModel object}.t)),
# and assign power later, if needed (PCB5.modify_node('PCB5A', P_int_new={some value in W})).
mass_PCB = 0.02  # [kg]
width = 0.1  # [m], assume symmetrical case here. Otherwise, use the function defined above.
thickness = 0.0016  # [m]
PCB5 = NodalModel(title='PCB5')
PCB5A = Node(name='PCB5A', material='PCB', coating='PCB_green', mass=mass_PCB/2, origin=(0., 0., 0.),
             geom=(width, width, 0), area=width**2/2, outer=True)
PCB5B = Node(name='PCB5B', material='PCB', coating='PCB_green', mass=mass_PCB/8, origin=(width*0.4, width*0.4, 0.), area=width**2/8, outer=True)
PCB5C = Node(name='PCB5C', material='PCB', coating='PCB_green', mass=mass_PCB/8, origin=(-width*0.4, width*0.4, 0.), area=width**2/8, outer=True)
PCB5D = Node(name='PCB5D', material='PCB', coating='PCB_green', mass=mass_PCB/8, origin=(-width*0.4, -width*0.4, 0.), area=width**2/8, outer=True)
PCB5E = Node(name='PCB5E', material='PCB', coating='PCB_green', mass=mass_PCB/8, origin=(width*0.4, -width*0.4, 0.), area=width**2/8, outer=True)
PCB5.add_node([PCB5A, PCB5B, PCB5C, PCB5D, PCB5E])
PCB5.connect(PCB5A, [PCB5B, PCB5C, PCB5D, PCB5E], L_through=3/4*np.sqrt(2*(width/2)**2),
             A=thickness*np.sqrt(2*(width/2)**2))

# 9-node PCB (3x3 grid)
# This example is 10x10 cm.
# This example is an inner node (outer=False), so only node A can take a power input, and node A represents the entire PCB area for internal radiation.
# Ordering of the nodes is first the centre node, then counterclockwise starting at the centre right node.
# For radiative heat transfer, the centre node (A) is assumed to be the only radiating face, using the entire PCB area.
mass_PCB = 0.02  # [kg]
width = 0.1  # [m], assume symmetrical case here. Otherwise, use the function defined above.
thickness = 0.0016  # [m]
PCB9 = NodalModel(title='PCB9')
PCB9A = Node(name='PCB9A', material='PCB', coating='PCB_green', mass=mass_PCB/9, origin=(0., 0., 0.),
             geom=(width, width, 0))
PCB9B = Node(name='PCB9B', material='PCB', coating='PCB_green', mass=mass_PCB/9, origin=(width/3, 0., 0.))
PCB9C = Node(name='PCB9C', material='PCB', coating='PCB_green', mass=mass_PCB/9, origin=(width/3, width/3, 0.))
PCB9D = Node(name='PCB9D', material='PCB', coating='PCB_green', mass=mass_PCB/9, origin=(0., width/3, 0.))
PCB9E = Node(name='PCB9E', material='PCB', coating='PCB_green', mass=mass_PCB/9, origin=(-width/3, width/3, 0.))
PCB9F = Node(name='PCB9F', material='PCB', coating='PCB_green', mass=mass_PCB/9, origin=(-width/3, 0., 0.))
PCB9G = Node(name='PCB9G', material='PCB', coating='PCB_green', mass=mass_PCB/9, origin=(-width/3, -width/3, 0.))
PCB9H = Node(name='PCB9H', material='PCB', coating='PCB_green', mass=mass_PCB/9, origin=(0., -width/3, 0.))
PCB9I = Node(name='PCB9I', material='PCB', coating='PCB_green', mass=mass_PCB/9, origin=(width/3, -width/3, 0.))
PCB9.add_node([PCB9A, PCB9B, PCB9C, PCB9D, PCB9E, PCB9F, PCB9G, PCB9H, PCB9I])
PCB9.connect('PCB9A', [PCB9B, PCB9D, PCB9F, PCB9H], L_through=width/3, A=thickness*width/3)
PCB9.connect('PCB9B', [PCB9C, PCB9I], L_through=width/3, A=thickness*width/3)
PCB9.connect('PCB9D', [PCB9C, PCB9E], L_through=width/3, A=thickness*width/3)
PCB9.connect('PCB9F', [PCB9E, PCB9G], L_through=width/3, A=thickness*width/3)
PCB9.connect('PCB9H', [PCB9G, PCB9I], L_through=width/3, A=thickness*width/3)
