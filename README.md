# AthenaK

Block-based AMR framework with fluid, particle and numerical relativity solvers in Kokkos.

## Overview

AthenaK is a complete rewrite of the AMR framework and fluid solvers in the [Athena++](https://github.com/PrincetonUniversity/athena) astrophysical MHD code using the [Kokkos](https://kokkos.org/) programming model.

Using Kokkos enables *performance-portability*.  AthenaK will run on any hardware supported by Kokkos, including CPU, GPUs from various vendors, and ARM processors.

AthenaK is targeting challenging problems that require exascale resources, and as such it does not implement all of the features of Athena++.  Current code features are:
- Block-based AMR with dynamical execution via a task list
- Non-relativistic (Newtonian) hydrodynamics and MHD
- Special relativistic (SR) hydrodynamics and MHD
- General relativistic (GR) hydrodynamics and MHD in stationary spacetimes
- Relativistic radiation transport
- Lagrangian tracer particles, and charged test particles
- Numerical relativity solver using the Z4c formalism
- GR hydrodynamics and MHD in dynamical spacetimes

The numerical algorithms implemented in AthenaK are all based on higher-order finite volume methods with a variety of reconstruction algorithms, Riemann solvers, and time integration methods.

## Getting Started

The code is designed to be user-friendly with as few external dependencies as possible.

Documention is permanently under construction on the [wiki](https://github.com/IAS-Astrophysics/athenak/wiki) pages.

In particular, see the complete list of [requirements](https://github.com/IAS-Astrophysics/athenak/wikis/Requirements), or
instructions on how to [download](https://github.com/IAS-Astrophysics/athenak/wikis/Download) and [build](https://github.com/IAS-Astrophysics/athenak/wikis/Build) the code for various devices.
Other pages give instructions for running the code.

Since AthenaK is very similar to Athena++, the [Athena++ documention](https://github.com/PrincetonUniversity/athena/wiki) may also be helpful.

## Compiling and Running on Rusty

1. Load the following modules

`module purge`
`module load modules/2.3-20240529`
`module load openblas/single-0.3.26`
`module load gcc openmpi`
`module load cuda/12.3.2`

`export PATH=/usr/local/cuda-11.4.4/bin:$PATH`
`export LD_LIBRARY_PATH=/usr/local/cuda-11.4.4/lib64:$LD_LIBRARY_PATH`

2. Navigate to the athenak folder and run the following for v100

`cmake3 -D Kokkos_ENABLE_CUDA=On -D Kokkos_ARCH_AMPERE80=On -D CMAKE_CXX_COMPILER=~/Codes/athenak/kokkos/bin/nvcc_wrapper -D CMAKE_BUILD_TYPE=Debug -B {Build Folder}`

3. Navigate to the Build_Folder and run the following make command

`make -j 64`

4. run the following command for a particular problem

`srun src/athena -i ~/Codes/athenak/inputs/mhd/orszag_tang.athinput mesh/nx1=512 mesh/nx2=512 meshblock/nx1=64 meshblock/nx2=64 -d output/`



## Code papers

For more details on the features and algorithms implemented in AthenaK, see the code papers:
- [Stone et al (2024)](https://ui.adsabs.harvard.edu/abs/2024arXiv240916053S/abstract): basic framework
- [Zhu et al. (2024)](https://ui.adsabs.harvard.edu/abs/2024arXiv240910383Z/abstract): numerical relativity solver
- [Fields at al. (2024)](https://ui.adsabs.harvard.edu/abs/2024arXiv240910384F/abstract): GR hydro and MHD solver in dynamical spacetimes

Please reference these papers as appropriate for any publications that use AthenaK.
# athenak_updates

```#Rusty Job Script 
#!/bin/bash -l
#SBATCH --partition gpu
##SBATCH --constraint v100,ib
#SBATCH --gpus-per-node=1
#SBATCH -C a100

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --mem=200G

#SBATCH --mail-user=dp3327@columbia.edu
#SBATCH --job-name=test_athenak
#SBATCH --mail-type=ALL
#SBATCH --time=00-00:05:00
#SBATCH -o outfile # STDOUT 
#SBATCH -e errorfile # STDERR


srun src/athena -i ~/Codes/athenak/inputs/mhd/orszag_tang.athinput mesh/nx1=512 mesh/nx2=512 meshblock/nx1=64 meshblock/nx2=64 -d output/