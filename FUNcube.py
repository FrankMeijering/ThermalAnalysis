# -*- coding: utf-8 -*-
"""
Created on Thu Aug 1 12:37:27 2024
Author: Frank Meijering (Delft University of Technology)

FUNcube.py contains a reconstruction of the thermal model of the FUNcube-1 satellite. This file can also be considered
as an example on how to create orbital models and nodal models. However, there are generally more ways in which the
code can be used, so it is recommended to refer to the user's manual for more details.
"""


import numpy as np
from ThermalBudget import Node, NodalModel, OrbitalModel, show_available_materials
from CommonNodalModels import make_5node_pcb, assign_q_ext_to_pcb

show_available_materials()

P_tot = 1.2  # [W]
mass_PCB = 0.02  # [kg] approximate average mass of a PCB including the electronics
thickness_PCB = 0.0016  # [m]
# ---------------- OUTER PLATES & ENVIRONMENT ------------------
# FUNcube orbit
outer_nodes = ['x+', 'y+', 'z+', 'x-', 'y-', 'z-']
FUNcubeOrbit = OrbitalModel(h=(629e3+563e3)/2, surfaces=outer_nodes, beta=30., day=226, n_orbits=2., dt=5.,  # incl. is 97.7 deg, beta approx. 30 deg (from n2yo.com)
                            angular_rates=[0., 0., 0.])
FUNcubeOrbit.compute()
t = FUNcubeOrbit.t
print(f'Beta angle: {FUNcubeOrbit.beta*180/np.pi:.2f} deg\n')

# Start with the outer plates, 1U
xplus, xplusA, xplusB, xplusC, xplusD, xplusE = make_5node_pcb(title='x+', mass=mass_PCB, origin=(0.05, 0, 0), geom=(0, 0.1, 0.1), thickness=thickness_PCB,
                       t=t, coating='solar_cell_mix_black_paint', outer=True)
yplus, yplusA, yplusB, yplusC, yplusD, yplusE = make_5node_pcb(title='y+', mass=mass_PCB, origin=(0, 0.05, 0), geom=(0.1, 0, 0.1), thickness=thickness_PCB,
                       t=t, coating='solar_cell_mix_black_paint', outer=True)
zplus, zplusA, zplusB, zplusC, zplusD, zplusE = make_5node_pcb(title='z+', mass=mass_PCB, origin=(0, 0, 0.05), geom=(0.1, 0.1, 0), thickness=thickness_PCB,
                       t=t, coating='solar_cell_mix_black_paint', outer=True, power=0.040)
xmin, xminA, xminB, xminC, xminD, xminE = make_5node_pcb(title='x-', mass=mass_PCB, origin=(-0.05, 0, 0), geom=(0, 0.1, 0.1), thickness=thickness_PCB,
                      t=t, coating='solar_cell_mix_black_paint', outer=True)
ymin, yminA, yminB, yminC, yminD, yminE = make_5node_pcb(title='y-', mass=mass_PCB, origin=(0, -0.05, 0), geom=(0.1, 0, 0.1), thickness=thickness_PCB,
                      t=t, coating='solar_cell_mix_black_paint', outer=True)
zmin, zminA, zminB, zminC, zminD, zminE = make_5node_pcb(title='z-', mass=mass_PCB, origin=(0, 0, -0.05), geom=(0.1, 0.1, 0), thickness=thickness_PCB,
                      t=t, coating='solar_cell_mix_black_paint', outer=True)

# Need this function to assign the environmental heat to all nodes of the PCB
assign_q_ext_to_pcb([xplus, yplus, zplus, xmin, ymin, zmin], FUNcubeOrbit)

# ---------------------------- PCBs --------------------------
PCB1, PCB1A, PCB1B, PCB1C, PCB1D, PCB1E = make_5node_pcb(title='PCB1', mass=mass_PCB, origin=(0., 0., -0.04),
                                                         geom=(0.1, 0.1, 0.), thickness=thickness_PCB, t=t, power=0.462)
PCB2, PCB2A, PCB2B, PCB2C, PCB2D, PCB2E = make_5node_pcb(title='PCB2', mass=mass_PCB, origin=(0., 0., -0.02),
                                                         geom=(0.1, 0.1, 0.), thickness=thickness_PCB, t=t, power=0.160)
# Also give this one a battery
PCB2bat = Node(name='PCB2bat', material='copper', coating='black_paint', mass=0.05, origin=(0., 0., -0.015))
PCB2.add_node(PCB2bat)
PCB2.connect('PCB2A', PCB2bat, contact_obj='graphite', A=0.05**2)

PCB3, PCB3A, PCB3B, PCB3C, PCB3D, PCB3E = make_5node_pcb(title='PCB3', mass=mass_PCB, origin=(0., 0., 0.),
                                                         geom=(0.1, 0.1, 0.), thickness=thickness_PCB, t=t, power=0.303)
PCB4, PCB4A, PCB4B, PCB4C, PCB4D, PCB4E = make_5node_pcb(title='PCB4', mass=mass_PCB, origin=(0., 0., 0.02),
                                                         geom=(0.1, 0.1, 0.), thickness=thickness_PCB, t=t, power=0.190)
PCB5, PCB5A, PCB5B, PCB5C, PCB5D, PCB5E = make_5node_pcb(title='PCB5', mass=mass_PCB, origin=(0., 0., 0.04),
                                                         geom=(0.1, 0.1, 0.), thickness=thickness_PCB, t=t, power=0.015)

# ---------------------- SPACERS -----------------------------
spcr_mass = 0.02  # [kg] in reality there are also stainless steel rods, but they conduct almost nothing; their mass is included in the spacers.

spcr12 = NodalModel(t, title='spcr12')
spcr12B = Node(name='spcr12B', material='al7075', coating='al_unpolished', mass=spcr_mass, origin=(0.04, 0.04, -0.03))
spcr12C = Node(name='spcr12C', material='al7075', coating='al_unpolished', mass=spcr_mass, origin=(-0.04, 0.04, -0.03))
spcr12D = Node(name='spcr12D', material='al7075', coating='al_unpolished', mass=spcr_mass, origin=(-0.04, -0.04, -0.03))
spcr12E = Node(name='spcr12E', material='al7075', coating='al_unpolished', mass=spcr_mass, origin=(0.04, -0.04, -0.03))
spcr12.add_node([spcr12B, spcr12C, spcr12D, spcr12E])

spcr23 = NodalModel(t, title='spcr23')
spcr23B = Node(name='spcr23B', material='al7075', coating='al_unpolished', mass=spcr_mass, origin=(0.04, 0.04, -0.01))
spcr23C = Node(name='spcr23C', material='al7075', coating='al_unpolished', mass=spcr_mass, origin=(-0.04, 0.04, -0.01))
spcr23D = Node(name='spcr23D', material='al7075', coating='al_unpolished', mass=spcr_mass, origin=(-0.04, -0.04, -0.01))
spcr23E = Node(name='spcr23E', material='al7075', coating='al_unpolished', mass=spcr_mass, origin=(0.04, -0.04, -0.01))
spcr23.add_node([spcr23B, spcr23C, spcr23D, spcr23E])

spcr34 = NodalModel(t, title='spcr34')
spcr34B = Node(name='spcr34B', material='al7075', coating='al_unpolished', mass=spcr_mass, origin=(0.04, 0.04, 0.01))
spcr34C = Node(name='spcr34C', material='al7075', coating='al_unpolished', mass=spcr_mass, origin=(-0.04, 0.04, 0.01))
spcr34D = Node(name='spcr34D', material='al7075', coating='al_unpolished', mass=spcr_mass, origin=(-0.04, -0.04, 0.01))
spcr34E = Node(name='spcr34E', material='al7075', coating='al_unpolished', mass=spcr_mass, origin=(0.04, -0.04, 0.01))
spcr34.add_node([spcr34B, spcr34C, spcr34D, spcr34E])

spcr45 = NodalModel(t, title='spcr45')
spcr45B = Node(name='spcr45B', material='al7075', coating='al_unpolished', mass=spcr_mass, origin=(0.04, 0.04, 0.03))
spcr45C = Node(name='spcr45C', material='al7075', coating='al_unpolished', mass=spcr_mass, origin=(-0.04, 0.04, 0.03))
spcr45D = Node(name='spcr45D', material='al7075', coating='al_unpolished', mass=spcr_mass, origin=(-0.04, -0.04, 0.03))
spcr45E = Node(name='spcr45E', material='al7075', coating='al_unpolished', mass=spcr_mass, origin=(0.04, -0.04, 0.03))
spcr45.add_node([spcr45B, spcr45C, spcr45D, spcr45E])

# ------------------------ STRUCTURE -----------------------
rod_mass = 0.08
ring_zm = Node(name='ring_zm', material='al7075', coating='al_unpolished', mass=rod_mass, origin=(0., 0., -0.045))
ring_z = Node(name='ring_z', material='al7075', coating='al_unpolished', mass=rod_mass, origin=(0., 0., 0.045))
rodB = Node(name='rodB', material='al7075', coating='al_unpolished', mass=rod_mass, origin=(0.05, 0.05, 0.))
rodC = Node(name='rodC', material='al7075', coating='al_unpolished', mass=rod_mass, origin=(-0.05, 0.05, 0.))
rodD = Node(name='rodD', material='al7075', coating='al_unpolished', mass=rod_mass, origin=(-0.05, -0.05, 0.))
rodE = Node(name='rodE', material='al7075', coating='al_unpolished', mass=rod_mass, origin=(0.05, -0.05, 0.))

# # -------------------- COMBINE WHOLE MODEL ---------------------
FUNcubeModel = NodalModel(t, title='FUNcubeModel')
FUNcubeModel.add_node([PCB1, PCB2, PCB3, PCB4, PCB5, xplus, yplus, zplus, xmin, ymin, zmin, spcr12, spcr23, spcr34, spcr45,
                       ring_zm, ring_z, rodB, rodC, rodD, rodE])
spcr_area = 0.00001  # [m2] Approximately an open cylinder of 5 mm outside diameter and 3 mm inside diameter
FUNcubeModel.connect(PCB1A, ['z-A', PCB2A], rad=True)
FUNcubeModel.connect(PCB3A, [PCB2A, PCB4A], rad=True)
FUNcubeModel.connect(PCB5A, [PCB4A, 'z+A'], rad=True)
FUNcubeModel.connect(PCB1B, [spcr12B], contact_obj='al_al', A=spcr_area)  # contact point provides much more resistance than the spacer itself
FUNcubeModel.connect(PCB1C, [spcr12C], contact_obj='al_al', A=spcr_area)
FUNcubeModel.connect(PCB1D, [spcr12D], contact_obj='al_al', A=spcr_area)
FUNcubeModel.connect(PCB1E, [spcr12E], contact_obj='al_al', A=spcr_area)
FUNcubeModel.connect(PCB2B, [spcr12B, spcr23B], contact_obj='al_al', A=spcr_area)
FUNcubeModel.connect(PCB2C, [spcr12C, spcr23C], contact_obj='al_al', A=spcr_area)
FUNcubeModel.connect(PCB2D, [spcr12D, spcr23D], contact_obj='al_al', A=spcr_area)
FUNcubeModel.connect(PCB2E, [spcr12E, spcr23E], contact_obj='al_al', A=spcr_area)
FUNcubeModel.connect(PCB3B, [spcr23B, spcr34B], contact_obj='al_al', A=spcr_area)
FUNcubeModel.connect(PCB3C, [spcr23C, spcr34C], contact_obj='al_al', A=spcr_area)
FUNcubeModel.connect(PCB3D, [spcr23D, spcr34D], contact_obj='al_al', A=spcr_area)
FUNcubeModel.connect(PCB3E, [spcr23E, spcr34E], contact_obj='al_al', A=spcr_area)
FUNcubeModel.connect(PCB4B, [spcr34B, spcr45B], contact_obj='al_al', A=spcr_area)
FUNcubeModel.connect(PCB4C, [spcr34C, spcr45C], contact_obj='al_al', A=spcr_area)
FUNcubeModel.connect(PCB4D, [spcr34D, spcr45D], contact_obj='al_al', A=spcr_area)
FUNcubeModel.connect(PCB4E, [spcr34E, spcr45E], contact_obj='al_al', A=spcr_area)
FUNcubeModel.connect(PCB5B, [spcr45B], contact_obj='al_al', A=spcr_area)
FUNcubeModel.connect(PCB5C, [spcr45C], contact_obj='al_al', A=spcr_area)
FUNcubeModel.connect(PCB5D, [spcr45D], contact_obj='al_al', A=spcr_area)
FUNcubeModel.connect(PCB5E, [spcr45E], contact_obj='al_al', A=spcr_area)
FUNcubeModel.connect(ring_zm, ['z-B', 'z-C', 'z-D', 'z-E', 'x+D', 'x+E', 'y+D', 'y+E', 'x-D', 'x-E', 'y-D', 'y-E'], contact_obj='al_al', A=0.002)
FUNcubeModel.connect(ring_z, ['z+B', 'z+C', 'z+D', 'z+E', 'x+B', 'x+C', 'y+B', 'y+C', 'x-B', 'x-C', 'y-B', 'y-C'], contact_obj='al_al', A=0.002)
FUNcubeModel.connect(ring_zm, [rodB, rodC, rodD, rodE], L_through=0.05, A=0.05**2)  # lack of material assignment here means it uses the node's material (al7075)
FUNcubeModel.connect(ring_z, [rodB, rodC, rodD, rodE], L_through=0.05, A=0.05**2)
FUNcubeModel.connect(ring_zm, [PCB1B, PCB1C, PCB1D, PCB1E], contact_obj='al_al', A=spcr_area)
FUNcubeModel.connect(ring_z, [PCB5B, PCB5C, PCB5D, PCB5E], contact_obj='al_al', A=spcr_area)
FUNcubeModel.solve()

print(f'Total heat capacity: {np.sum(FUNcubeModel.C_cap)} J/K\n')

# FUNcubeModel.show_plots(whichnodes=['x+A', 'y+A', 'x-A', 'y-A'])
# FUNcubeOrbit.animate_attitude()
