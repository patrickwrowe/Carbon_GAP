# Carbon_GAP

## Introduction
This machine learning potential uses Gaussian process regression to 'learn' the energies and forces acting on atoms directly from quantum mechanical calculations. Evaluating the model allows you to follow the dynamics of a system of carbon atoms atoms according to the quantum mechanical forces, without having to solve the Schrodinger equation - the resulting simulation is about thousands of times cheaper than direct quantum mechanical simulation. 

## Paper Abstract
We present an accurate machine learning (ML) model for atomistic simulations of carbon, constructed using the Gaussian approximation potential (GAP) methodology. The potential, named GAP-20, describes the properties of the bulk crystalline and amorphous phases, crystal surfaces, and defect structures with an accuracy approaching that of direct ab initio simulation, but at a significantly reduced cost. We combine structural databases for amorphous carbon and graphene, which we extend substantially by adding suitable configurations, for example, for defects in graphene and other nanostructures. The final potential is fitted to reference data computed using the optB88-vdW density functional theory (DFT) functional. Dispersion interactions, which are crucial to describe multilayer carbonaceous materials, are therefore implicitly included. We additionally account for long-range dispersion interactions using a semianalytical two-body term and show that an improved model can be obtained through an optimization of the many-body smooth overlap of atomic positions descriptor. We rigorously test the potential on lattice parameters, bond lengths, formation energies, and phonon dispersions of numerous carbon allotropes. We compare the formation energies of an extensive set of defect structures, surfaces, and surface reconstructions to DFT reference calculations. The present work demonstrates the ability to combine, in the same ML model, the previously attained flexibility required for amorphous carbon [V. L. Deringer and G. Csányi, Phys. Rev. B 95, 094203 (2017)] with the high numerical accuracy necessary for crystalline graphene [Rowe et al., Phys. Rev. B 97, 054303 (2018)], thereby providing an interatomic potential that will be applicable to a wide range of applications concerning diverse forms of bulk and nanostructured carbon.

# Repository for Carbon GAP-20 Model &amp; Training Data. 

For testing and validation data concerning the model, please see: P. Rowe, V. Deringer, P. Gasparotto, G. Csányi, A. Michaelides, “An accurate and transferable machine learning potential for carbon”, J. Chem. Phys., 153, 034702, (2020)

This paper has been made open source and is available to read for free at: https://discovery.ucl.ac.uk/id/eprint/10104458/


# Additional Information

For installation instructions of the supporting GAP and QUIP codes, please see: https://libatoms.github.io/GAP/installation.html

The GAP potential and training data here may be used under the Creative Commons (CC BY 4.0) license, for the license conditions of the supporting QUIP and GAP source code, however, please see: https://github.com/libAtoms/GAP/blob/main/LICENSE.md and http://www.libatoms.org/gap/gap_download.html

If making use of this potential, please cite the following papers:

1.	P. Rowe, V. Deringer, P. Gasparotto, G. Csányi, A. Michaelides, “An accurate and transferable machine learning potential for carbon”, J. Chem. Phys., 153, 034702, (2020)
2.	A. P. Bartók, M. C. Payne, R. Kondor, G. Csányi, "Gaussian Approximation Potentials: The Accuracy of Quantum Mechanics, without the Electrons", Phys. Rev. Lett., 104, 136403
