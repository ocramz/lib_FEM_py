# lib_FEM_py
2D Finite Element library, with extensions for micro-manufacturing simulation

I wrote this library during my PhD to simulate the propagation of optical waves in transparent dielectrics, such as fiber optics or silicon photonic waveguides. In particular, my research focused on how to include the compound effect of manufacturing defects in the device performance prediction and optimisation.

## Notes on the FE model
The library implements a 2D Galerkin FE method for solving the Helmholtz wave equation with PML boundary conditions [1] in the linear regime and on a uniform mesh.


## Notes on the micro-manufacturing model
The extensions mentioned above serve to model a spatial "diffusion" of the material coefficients, and subsequent transformations thereof that occur during manufacturing of silicon photonic devices ("integrated optics"). 
In particular, we include: 

1. a simplified etching photoresist development model ("RD" stands for "reaction-diffusion"). This is modeled as an initial value problem, via a pair of ODEs that capture the two reacting species (resist polymer and diffusing photo-acid catalyst)
2. an idealised etching step of the resulting material distribution map (a thresholding nonlinearity)


## References
[1] J. Berenger (1994). "A perfectly matched layer for the absorption of electromagnetic waves". Journal of Computational Physics. 114 (2): 185–200. 
