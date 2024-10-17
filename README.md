# Thermal Analysis
Modular Thermal Analysis Software - MSc Thesis

Master's Thesis project in Aerospace Engineering at Delft University of Technology (TU Delft).
Main repository is available on the TU Delft GitLab page: https://gitlab.tudelft.nl/delfispace
A copy of this repository might be added on this GitHub page.

The MSc thesis report can be found on: https://resolver.tudelft.nl/uuid:74339e69-0fdb-4ce9-96d1-a909e729fb96
under the name: "Modular Thermal Analysis for CubeSats and PocketQubes"

## Overview of files
- ThermalBudget.py: contains the definition of the Node, NodalModel, and OrbitalModel classes. It is important to understand how to use them, and the documentation present in those class definitions helps a lot with input syntax and troubleshooting.
- CommonNodalModels.py: contains some functions to easily define PCBs with multiple nodes. It is recommended to expand this file by adding more and more NodalModels that are frequently reused.
- SensitivityAnalysis.py: contains the necessary functions to perform a sensitivity analysis such as presented in Chapter 5.
- Materials.csv: is the only CSV file, and this file contains the material database. All values can be viewed or adapted here, and new materials can be added as well.
- FUNcube.py: this file is not directly used, but it can serve as an example that the user can copy and adapt to fit their own model.
- Constants.py: Contains universal constants required for various computations. They do not cross the user’s path, but are indirectly used in the other files. They may be adapted by the user, but it is recommended to only do so if strictly needed. Each number has a source (url) from which it was taken.
- VerificationValidation.py: contains numerous thermal and orbital models, for the purpose of performing verification and validation. Results were shown in Chapter 4.
- EnvironmentRadiation.py: computes the orbit and environmental heat fluxes. However, the file is not directly accessed by the user, but indirectly via the OrbitalModel object defined in ThermalBudget.py.
- Materials.py: defines the Material, Coating, and Contact objects. However, these are likely never used by the user, but only used indirectly via the Node and NodalModel objects in ThermalBudget.py.
- TestFile.py: may be useful to test some things without messing with important other files.

## Overview of folders
- ESATAN: folder use to store thermal simulations made with ESATAN-TMS. Only used for verification purposes.
- FUNcubeData: folder with FUNcube-1 flight data. Only used for validation purposes.
- SensitivityAnalysis: folder which is used as a default to store results from any sensitivity analyses performed. The stored files are Python Pickle files, and contain NodalModel objects that have been solved, i.e., that contain simulation results. For the FUNcube-1 sensitivities, an example is given for the absorptivity 'ABS'. The other sensitivities can be created with SensitivityAnalysis.py, by running the examples at the bottom of the code.
