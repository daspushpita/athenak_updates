//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file linear_alfven_wave.cpp
// ! \brief Testing Linear Alfven wave problem generator for 1D/2D/3D problems
// ! Driver::Finalize().

// C/C++ headers
#include <algorithm>  // min, max
#include <cmath>      // sqrt()
#include <cstdio>     // fopen(), fprintf(), freopen()
#include <iostream>   // endl
#include <sstream>    // stringstream
#include <string>     // c_str()
#include <limits>

// Athena++ headers
#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "coordinates/cell_locations.hpp"
#include "mesh/mesh.hpp"
#include "eos/eos.hpp"
#include "hydro/hydro.hpp"
#include "mhd/mhd.hpp"
#include "driver/driver.hpp"
#include "pgen/pgen.hpp"

// function to compute errors in solution at end of run
void LinearWaveErrors_alfven(ParameterInput *pin, Mesh *pm);

// function to compute eigenvectors of linear waves in mhd
void MHDEigensystem_rel(const Real d, const Real v1, const Real v2, const Real v3,
                    const Real p, const Real b1, const Real b2, const Real b3,
                    const Real x, const Real y, const EOS_Data &eos,
                    Real eigenvalues[1], Real right_eigenmatrix[9]);

namespace {
// global variable to control computation of initial conditions versus errors
bool set_initial_conditions = true;

//----------------------------------------------------------------------------------------
//! \struct LinWaveVariablesALfven
//! \brief container for variables shared with vector potential and error functions

struct LinWaveVariablesALfven {
  Real d0, p0, v1_0, v2_0, v3_0 b1_0, b2_0, b3_0, dby, dbz, k_par;
  Real cos_a2, cos_a3, sin_a2, sin_a3;
};

//----------------------------------------------------------------------------------------
//! \fn Real A1(const Real x1,const Real x2,const Real x3)
//! \brief A1: 1-component of vector potential, using a gauge such that Ax = 0, and Ay,
//! Az are functions of x and y alone.

KOKKOS_INLINE_FUNCTION
Real A1(const Real x1, const Real x2, const Real x3, const LinWaveVariablesALfven lw) {
  Real x =  x1*lw.cos_a2*lw.cos_a3 + x2*lw.cos_a2*lw.sin_a3 + x3*lw.sin_a2;
  Real y = -x1*lw.sin_a3           + x2*lw.cos_a3;
  Real Ay =  lw.b3_0*x - (lw.dbz/lw.k_par)*std::cos(lw.k_par*(x));
  Real Az = -lw.b2_0*x + (lw.dby/lw.k_par)*std::cos(lw.k_par*(x)) + lw.b1_0*y;

  return -Ay*lw.sin_a3 - Az*lw.sin_a2*lw.cos_a3;
}

//----------------------------------------------------------------------------------------
//! \fn Real A2(const Real x1,const Real x2,const Real x3)
//! \brief A2: 2-component of vector potential

KOKKOS_INLINE_FUNCTION
Real A2(const Real x1, const Real x2, const Real x3, const LinWaveVariablesALfven lw) {
  Real x =  x1*lw.cos_a2*lw.cos_a3 + x2*lw.cos_a2*lw.sin_a3 + x3*lw.sin_a2;
  Real y = -x1*lw.sin_a3            + x2*lw.cos_a3;
  Real Ay =  lw.b3_0*x - (lw.dbz/lw.k_par)*std::cos(lw.k_par*(x));
  Real Az = -lw.b2_0*x + (lw.dby/lw.k_par)*std::cos(lw.k_par*(x)) + lw.b1_0*y;

  return Ay*lw.cos_a3 - Az*lw.sin_a2*lw.sin_a3;
}

//----------------------------------------------------------------------------------------
//! \fn Real A3(const Real x1,const Real x2,const Real x3)
//! \brief A3: 3-component of vector potential

KOKKOS_INLINE_FUNCTION
Real A3(const Real x1, const Real x2, const Real x3, const LinWaveVariablesALfven lw) {
  Real x =  x1*lw.cos_a2*lw.cos_a3 + x2*lw.cos_a2*lw.sin_a3 + x3*lw.sin_a2;
  Real y = -x1*lw.sin_a3           + x2*lw.cos_a3;
  Real Az = -lw.b2_0*x + (lw.dby/lw.k_par)*std::cos(lw.k_par*(x)) + lw.b1_0*y;

  return Az*lw.cos_a2;
}
} // end anonymous namespace

//----------------------------------------------------------------------------------------
//! \fn void ProblemGenerator::LinearAlfvenWave_()
//! \brief Sets initial conditions for linear wave tests

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  // set linear wave errors function
  pgen_final_func = LinearWaveErrors_alfven;
  
  if (restart) return;

  // read global parameters
  int wave_flag = pin->GetInteger("problem", "wave_flag");
  Real amp = pin->GetReal("problem", "amp");
  Real vflow = pin->GetOrAddReal("problem", "vflow", 0.0);
  bool along_x1 = pin->GetOrAddBoolean("problem", "along_x1", false);
  bool along_x2 = pin->GetOrAddBoolean("problem", "along_x2", false);
  bool along_x3 = pin->GetOrAddBoolean("problem", "along_x3", false);
  // error check input flags
  if ((along_x1 && (along_x2 || along_x3)) || (along_x2 && along_x3)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "Can only specify one of along_x1/2/3 to be true" << std::endl;
    exit(EXIT_FAILURE);
  }
  if ((along_x2 || along_x3) && pmy_mesh_->one_d) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "Cannot specify waves along x2 or x3 axis in 1D" << std::endl;
    exit(EXIT_FAILURE);
  }
  if (along_x3 && pmy_mesh_->two_d) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "Cannot specify waves along x3 axis in 2D" << std::endl;
    exit(EXIT_FAILURE);
  }

  // Code below will automatically calculate wavevector along grid diagonal, imposing the
  // conditions of periodicity and exactly one wavelength along each grid direction
  Real x1size = pmy_mesh_->mesh_size.x1max - pmy_mesh_->mesh_size.x1min;
  Real x2size = pmy_mesh_->mesh_size.x2max - pmy_mesh_->mesh_size.x2min;
  Real x3size = pmy_mesh_->mesh_size.x3max - pmy_mesh_->mesh_size.x3min;

  // start with wavevector along x1 axis
  LinWaveVariablesALfven lwv;
  lwv.cos_a3 = 1.0;
  lwv.sin_a3 = 0.0;
  lwv.cos_a2 = 1.0;
  lwv.sin_a2 = 0.0;

  // choose the smallest projection of the wavelength in each direction that is > 0
  Real lambda = std::numeric_limits<float>::max();
  if (lwv.cos_a2*lwv.cos_a3 > 0.0) {
    lambda = std::min(lambda, x1size*lwv.cos_a2*lwv.cos_a3);
  }
  if (lwv.cos_a2*lwv.sin_a3 > 0.0) {
    lambda = std::min(lambda, x2size*lwv.cos_a2*lwv.sin_a3);
  }
  if (lwv.sin_a2 > 0.0) lambda = std::min(lambda, x3size*lwv.sin_a2);

  // Set background state: v1_0 is parallel to wavevector.
  // Similarly, for MHD:   b1_0 is parallel to wavevector, b2_0/b3_0 are perpendicular
  
  lwv.d0 = 1.0;       // This sets the density
  lwv.v1_0 = vflow;   // This sets the initial v^i (x)
  lwv.v2_0 = 0.;      // This sets the initial v^i (y)
  lwv.v3_0 = 0.;      // This sets the initial v^i (z)
  lwv.b1_0 = 1.0;     // This sets the initial B^i (x)
  lwv.b2_0 = 0.;      // This sets the initial B^i (y)
  lwv.b3_0 = 0.;      // This sets the initial B^i (z)
  Real xfact = 0.0;
  Real yfact = 1.0;

  // capture variables for kernel
  auto &indcs = pmy_mesh_->mb_indcs;
  int &is = indcs.is; int &ie = indcs.ie;
  int &js = indcs.js; int &je = indcs.je;
  int &ks = indcs.ks; int &ke = indcs.ke;
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  auto &size = pmbp->pmb->mb_size;

  // initialize MHD variables ------------------------------------------------------------
  if (pmbp->pmhd != nullptr) {
    EOS_Data &eos = pmbp->pmhd->peos->eos_data;
    int nmb = pmbp->nmb_thispack;
    int nmhd_ = pmbp->pmhd->nmhd;
    Real gm1 = eos.gamma - 1.0;
    Real p0 = 1.0/eos.gamma;

    // Initialize k_parallel
    lwv.k_par = 2.0*(M_PI)/lambda;

    // Compute eigenvectors in mhd
    Real rem[7];
    Real ev[1];

    par_for("lin_alfven_rel1", DevExeSpace(), 0,nmb-1,ks,ke,js,je,is,ie,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      Real &x1min = size.d_view(m).x1min;
      Real &x1max = size.d_view(m).x1max;
      int nx1 = indcs.nx1;
      Real x1v = CellCenterX(i-is, nx1, x1min, x1max);

      Real &x2min = size.d_view(m).x2min;
      Real &x2max = size.d_view(m).x2max;
      int nx2 = indcs.nx2;
      Real x2v = CellCenterX(j-js, nx2, x2min, x2max);

      Real &x3min = size.d_view(m).x3min;
      Real &x3max = size.d_view(m).x3max;
      int nx3 = indcs.nx3;
      Real x3v = CellCenterX(k-ks, nx3, x3min, x3max); 

      Real dx1 = size.d_view(m).dx1;
      Real dx2 = size.d_view(m).dx2;
      Real dx3 = size.d_view(m).dx3;

    // Extract metric and inverse
    Real glower[4][4], gupper[4][4];
    ComputeMetricAndInverse(x1v, x2v, x3v, coord.is_minkowski, coord.bh_spin,
                            glower, gupper);


    });



    MHDEigensystem_rel(lwv.d0, lwv.v1_0, 0.0, 0.0, p0, lwv.b1_0, lwv.b2_0, lwv.b3_0,
                   xfact, yfact, eos, ev, rem);

    lwv.dby = amp*rem[nmhd_  ][wave_flag];
    lwv.dbz = amp*rem[nmhd_+1][wave_flag];

    // set new time limit in ParameterInput (to be read by Driver constructor) based on
    // wave speed of selected mode.
    // input tlim should be interpreted as number of wave periods for evolution
    if (set_initial_conditions) {
      Real tlim = pin->GetReal("time", "tlim");
      pin->SetReal("time", "tlim", tlim*(std::abs(lambda/ev[wave_flag])));
    }

    // compute solution in u1/b1 registers. For initial conditions, set u1/b1 -> u0/b0.
    auto &u1 = (set_initial_conditions)? pmbp->pmhd->u0 : pmbp->pmhd->u1;
    auto &b1 = (set_initial_conditions)? pmbp->pmhd->b0 : pmbp->pmhd->b1;

    // compute vector potential over all faces
    int ncells1 = indcs.nx1 + 2*(indcs.ng);
    int ncells2 = (indcs.nx2 > 1)? (indcs.nx2 + 2*(indcs.ng)) : 2;
    int ncells3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*(indcs.ng)) : 2;
    DvceArray4D<Real> a1, a2, a3;
    Kokkos::realloc(a1, nmb,ncells3,ncells2,ncells1);
    Kokkos::realloc(a2, nmb,ncells3,ncells2,ncells1);
    Kokkos::realloc(a3, nmb,ncells3,ncells2,ncells1);

    auto &nghbr = pmbp->pmb->nghbr;
    auto &mblev = pmbp->pmb->mb_lev;

    par_for("lin_alfven_rel1", DevExeSpace(), 0,nmb-1,ks,ke+1,js,je+1,is,ie+1,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      Real &x1min = size.d_view(m).x1min;
      Real &x1max = size.d_view(m).x1max;
      int nx1 = indcs.nx1;
      Real x1v = CellCenterX(i-is, nx1, x1min, x1max);
      Real x1f   = LeftEdgeX(i  -is, nx1, x1min, x1max);

      Real &x2min = size.d_view(m).x2min;
      Real &x2max = size.d_view(m).x2max;
      int nx2 = indcs.nx2;
      Real x2v = CellCenterX(j-js, nx2, x2min, x2max);
      Real x2f   = LeftEdgeX(j  -js, nx2, x2min, x2max);

      Real &x3min = size.d_view(m).x3min;
      Real &x3max = size.d_view(m).x3max;
      int nx3 = indcs.nx3;
      Real x3v = CellCenterX(k-ks, nx3, x3min, x3max); // For 2D, set x3v = 0
      // Real x3v = (indcs.nx3 == 1) ? 0.0 : CellCenterX(k-ks, nx3, x3min, x3max); // For 2D, set x3v = 0
      Real x3f = LeftEdgeX(k  -ks, nx3, x3min, x3max);
      // printf("x3f %e\n", x3f);
      Real dx1 = size.d_view(m).dx1;
      Real dx2 = size.d_view(m).dx2;
      Real dx3 = size.d_view(m).dx3;

      a1(m,k,j,i) = A1(x1v, x2f, x3f, lwv);
      a2(m,k,j,i) = A2(x1f, x2v, x3f, lwv);
      a3(m,k,j,i) = A3(x1f, x2f, x3v, lwv);

      // When neighboring MeshBock is at finer level, compute vector potential as sum of
      // values at fine grid resolution.  This guarantees flux on shared fine/coarse
      // faces is identical.

      if (indcs.nx3 > 1){
      // Correct A1 at x2-faces, x3-faces, and x2x3-edges
        if ((nghbr.d_view(m,8 ).lev > mblev.d_view(m) && j==js) ||
            (nghbr.d_view(m,9 ).lev > mblev.d_view(m) && j==js) ||
            (nghbr.d_view(m,10).lev > mblev.d_view(m) && j==js) ||
            (nghbr.d_view(m,11).lev > mblev.d_view(m) && j==js) ||
            (nghbr.d_view(m,12).lev > mblev.d_view(m) && j==je+1) ||
            (nghbr.d_view(m,13).lev > mblev.d_view(m) && j==je+1) ||
            (nghbr.d_view(m,14).lev > mblev.d_view(m) && j==je+1) ||
            (nghbr.d_view(m,15).lev > mblev.d_view(m) && j==je+1) ||
            (nghbr.d_view(m,24).lev > mblev.d_view(m) && k==ks) ||
            (nghbr.d_view(m,25).lev > mblev.d_view(m) && k==ks) ||
            (nghbr.d_view(m,26).lev > mblev.d_view(m) && k==ks) ||
            (nghbr.d_view(m,27).lev > mblev.d_view(m) && k==ks) ||
            (nghbr.d_view(m,28).lev > mblev.d_view(m) && k==ke+1) ||
            (nghbr.d_view(m,29).lev > mblev.d_view(m) && k==ke+1) ||
            (nghbr.d_view(m,30).lev > mblev.d_view(m) && k==ke+1) ||
            (nghbr.d_view(m,31).lev > mblev.d_view(m) && k==ke+1) ||
            (nghbr.d_view(m,40).lev > mblev.d_view(m) && j==js && k==ks) ||
            (nghbr.d_view(m,41).lev > mblev.d_view(m) && j==js && k==ks) ||
            (nghbr.d_view(m,42).lev > mblev.d_view(m) && j==je+1 && k==ks) ||
            (nghbr.d_view(m,43).lev > mblev.d_view(m) && j==je+1 && k==ks) ||
            (nghbr.d_view(m,44).lev > mblev.d_view(m) && j==js && k==ke+1) ||
            (nghbr.d_view(m,45).lev > mblev.d_view(m) && j==js && k==ke+1) ||
            (nghbr.d_view(m,46).lev > mblev.d_view(m) && j==je+1 && k==ke+1) ||
            (nghbr.d_view(m,47).lev > mblev.d_view(m) && j==je+1 && k==ke+1)) {
          Real xl = x1v + 0.25*dx1;
          Real xr = x1v - 0.25*dx1;
          a1(m,k,j,i) = 0.5*(A1(xl,x2f,x3f,lwv) + A1(xr,x2f,x3f,lwv));
        }

        // Correct A2 at x1-faces, x3-faces, and x1x3-edges
        if ((nghbr.d_view(m,0 ).lev > mblev.d_view(m) && i==is) ||
            (nghbr.d_view(m,1 ).lev > mblev.d_view(m) && i==is) ||
            (nghbr.d_view(m,2 ).lev > mblev.d_view(m) && i==is) ||
            (nghbr.d_view(m,3 ).lev > mblev.d_view(m) && i==is) ||
            (nghbr.d_view(m,4 ).lev > mblev.d_view(m) && i==ie+1) ||
            (nghbr.d_view(m,5 ).lev > mblev.d_view(m) && i==ie+1) ||
            (nghbr.d_view(m,6 ).lev > mblev.d_view(m) && i==ie+1) ||
            (nghbr.d_view(m,7 ).lev > mblev.d_view(m) && i==ie+1) ||
            (nghbr.d_view(m,24).lev > mblev.d_view(m) && k==ks) ||
            (nghbr.d_view(m,25).lev > mblev.d_view(m) && k==ks) ||
            (nghbr.d_view(m,26).lev > mblev.d_view(m) && k==ks) ||
            (nghbr.d_view(m,27).lev > mblev.d_view(m) && k==ks) ||
            (nghbr.d_view(m,28).lev > mblev.d_view(m) && k==ke+1) ||
            (nghbr.d_view(m,29).lev > mblev.d_view(m) && k==ke+1) ||
            (nghbr.d_view(m,30).lev > mblev.d_view(m) && k==ke+1) ||
            (nghbr.d_view(m,31).lev > mblev.d_view(m) && k==ke+1) ||
            (nghbr.d_view(m,32).lev > mblev.d_view(m) && i==is && k==ks) ||
            (nghbr.d_view(m,33).lev > mblev.d_view(m) && i==is && k==ks) ||
            (nghbr.d_view(m,34).lev > mblev.d_view(m) && i==ie+1 && k==ks) ||
            (nghbr.d_view(m,35).lev > mblev.d_view(m) && i==ie+1 && k==ks) ||
            (nghbr.d_view(m,36).lev > mblev.d_view(m) && i==is && k==ke+1) ||
            (nghbr.d_view(m,37).lev > mblev.d_view(m) && i==is && k==ke+1) ||
            (nghbr.d_view(m,38).lev > mblev.d_view(m) && i==ie+1 && k==ke+1) ||
            (nghbr.d_view(m,39).lev > mblev.d_view(m) && i==ie+1 && k==ke+1)) {
          Real xl = x2v + 0.25*dx2;
          Real xr = x2v - 0.25*dx2;
          a2(m,k,j,i) = 0.5*(A2(x1f,xl,x3f,lwv) + A2(x1f,xr,x3f,lwv));
        }

        // Correct A3 at x1-faces, x2-faces, and x1x2-edges
        if ((nghbr.d_view(m,0 ).lev > mblev.d_view(m) && i==is) ||
            (nghbr.d_view(m,1 ).lev > mblev.d_view(m) && i==is) ||
            (nghbr.d_view(m,2 ).lev > mblev.d_view(m) && i==is) ||
            (nghbr.d_view(m,3 ).lev > mblev.d_view(m) && i==is) ||
            (nghbr.d_view(m,4 ).lev > mblev.d_view(m) && i==ie+1) ||
            (nghbr.d_view(m,5 ).lev > mblev.d_view(m) && i==ie+1) ||
            (nghbr.d_view(m,6 ).lev > mblev.d_view(m) && i==ie+1) ||
            (nghbr.d_view(m,7 ).lev > mblev.d_view(m) && i==ie+1) ||
            (nghbr.d_view(m,8 ).lev > mblev.d_view(m) && j==js) ||
            (nghbr.d_view(m,9 ).lev > mblev.d_view(m) && j==js) ||
            (nghbr.d_view(m,10).lev > mblev.d_view(m) && j==js) ||
            (nghbr.d_view(m,11).lev > mblev.d_view(m) && j==js) ||
            (nghbr.d_view(m,12).lev > mblev.d_view(m) && j==je+1) ||
            (nghbr.d_view(m,13).lev > mblev.d_view(m) && j==je+1) ||
            (nghbr.d_view(m,14).lev > mblev.d_view(m) && j==je+1) ||
            (nghbr.d_view(m,15).lev > mblev.d_view(m) && j==je+1) ||
            (nghbr.d_view(m,16).lev > mblev.d_view(m) && i==is && j==js) ||
            (nghbr.d_view(m,17).lev > mblev.d_view(m) && i==is && j==js) ||
            (nghbr.d_view(m,18).lev > mblev.d_view(m) && i==ie+1 && j==js) ||
            (nghbr.d_view(m,19).lev > mblev.d_view(m) && i==ie+1 && j==js) ||
            (nghbr.d_view(m,20).lev > mblev.d_view(m) && i==is && j==je+1) ||
            (nghbr.d_view(m,21).lev > mblev.d_view(m) && i==is && j==je+1) ||
            (nghbr.d_view(m,22).lev > mblev.d_view(m) && i==ie+1 && j==je+1) ||
            (nghbr.d_view(m,23).lev > mblev.d_view(m) && i==ie+1 && j==je+1)) {
          Real xl = x3v + 0.25*dx3;
          Real xr = x3v - 0.25*dx3;
          a3(m,k,j,i) = 0.5*(A3(x1f,x2f,xl,lwv) + A3(x1f,x2f,xr,lwv));
        }
      }
      else{

      // Correct A1 at x2-faces, x3-faces, and x2x3-edges
        if ((nghbr.d_view(m,8 ).lev > mblev.d_view(m) && j==js) ||
            (nghbr.d_view(m,9 ).lev > mblev.d_view(m) && j==js) ||
            (nghbr.d_view(m,10).lev > mblev.d_view(m) && j==js) ||
            (nghbr.d_view(m,11).lev > mblev.d_view(m) && j==js) ||
            (nghbr.d_view(m,12).lev > mblev.d_view(m) && j==je+1) ||
            (nghbr.d_view(m,13).lev > mblev.d_view(m) && j==je+1) ||
            (nghbr.d_view(m,14).lev > mblev.d_view(m) && j==je+1) ||
            (nghbr.d_view(m,15).lev > mblev.d_view(m) && j==je+1)) {
          Real xl = x1v + 0.25*dx1;
          Real xr = x1v - 0.25*dx1;
          a1(m,k,j,i) = 0.5*(A1(xl,x2f,x3f,lwv) + A1(xr,x2f,x3f,lwv));
        }

        // Correct A2 at x1-faces, x3-faces, and x1x3-edges
        if ((nghbr.d_view(m,0 ).lev > mblev.d_view(m) && i==is) ||
            (nghbr.d_view(m,1 ).lev > mblev.d_view(m) && i==is) ||
            (nghbr.d_view(m,2 ).lev > mblev.d_view(m) && i==is) ||
            (nghbr.d_view(m,3 ).lev > mblev.d_view(m) && i==is) ||
            (nghbr.d_view(m,4 ).lev > mblev.d_view(m) && i==ie+1) ||
            (nghbr.d_view(m,5 ).lev > mblev.d_view(m) && i==ie+1) ||
            (nghbr.d_view(m,6 ).lev > mblev.d_view(m) && i==ie+1) ||
            (nghbr.d_view(m,7 ).lev > mblev.d_view(m) && i==ie+1)) {
          Real xl = x2v + 0.25*dx2;
          Real xr = x2v - 0.25*dx2;
          a2(m,k,j,i) = 0.5*(A2(x1f,xl,x3f,lwv) + A2(x1f,xr,x3f,lwv));
        }

        // Correct A3 at x1-faces, x2-faces, and x1x2-edges
        if ((nghbr.d_view(m,0 ).lev > mblev.d_view(m) && i==is) ||
            (nghbr.d_view(m,1 ).lev > mblev.d_view(m) && i==is) ||
            (nghbr.d_view(m,2 ).lev > mblev.d_view(m) && i==is) ||
            (nghbr.d_view(m,3 ).lev > mblev.d_view(m) && i==is) ||
            (nghbr.d_view(m,4 ).lev > mblev.d_view(m) && i==ie+1) ||
            (nghbr.d_view(m,5 ).lev > mblev.d_view(m) && i==ie+1) ||
            (nghbr.d_view(m,6 ).lev > mblev.d_view(m) && i==ie+1) ||
            (nghbr.d_view(m,7 ).lev > mblev.d_view(m) && i==ie+1) ||
            (nghbr.d_view(m,8 ).lev > mblev.d_view(m) && j==js) ||
            (nghbr.d_view(m,9 ).lev > mblev.d_view(m) && j==js) ||
            (nghbr.d_view(m,10).lev > mblev.d_view(m) && j==js) ||
            (nghbr.d_view(m,11).lev > mblev.d_view(m) && j==js) ||
            (nghbr.d_view(m,12).lev > mblev.d_view(m) && j==je+1) ||
            (nghbr.d_view(m,13).lev > mblev.d_view(m) && j==je+1) ||
            (nghbr.d_view(m,14).lev > mblev.d_view(m) && j==je+1) ||
            (nghbr.d_view(m,15).lev > mblev.d_view(m) && j==je+1) ||
            (nghbr.d_view(m,16).lev > mblev.d_view(m) && i==is && j==js) ||
            (nghbr.d_view(m,17).lev > mblev.d_view(m) && i==is && j==js) ||
            (nghbr.d_view(m,18).lev > mblev.d_view(m) && i==ie+1 && j==js) ||
            (nghbr.d_view(m,19).lev > mblev.d_view(m) && i==ie+1 && j==js) ||
            (nghbr.d_view(m,20).lev > mblev.d_view(m) && i==is && j==je+1) ||
            (nghbr.d_view(m,21).lev > mblev.d_view(m) && i==is && j==je+1) ||
            (nghbr.d_view(m,22).lev > mblev.d_view(m) && i==ie+1 && j==je+1) ||
            (nghbr.d_view(m,23).lev > mblev.d_view(m) && i==ie+1 && j==je+1)) {
          Real xl = x3v + 0.25*dx3;
          Real xr = x3v - 0.25*dx3;
          a3(m,k,j,i) = 0.5*(A3(x1f,x2f,xl,lwv) + A3(x1f,x2f,xr,lwv));
        }
      }
    });

    // now compute conserved quantities, as well as face-centered fields
    par_for("lin_alfven_mine2", DevExeSpace(), 0,nmb-1,ks,ke,js,je,is,ie,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      Real &x1min = size.d_view(m).x1min;
      Real &x1max = size.d_view(m).x1max;
      int nx1 = indcs.nx1;
      Real x1v = CellCenterX(i-is, nx1, x1min, x1max);

      Real &x2min = size.d_view(m).x2min;
      Real &x2max = size.d_view(m).x2max;
      int nx2 = indcs.nx2;
      Real x2v = CellCenterX(j-js, nx2, x2min, x2max);

      Real &x3min = size.d_view(m).x3min;
      Real &x3max = size.d_view(m).x3max;
      int nx3 = indcs.nx3;
      // Real x3v = (indcs.nx3 == 1) ? 0.0 : CellCenterX(k-ks, nx3, x3min, x3max);
      Real x3v = CellCenterX(k-ks, nx3, x3min, x3max);

      Real x = lwv.cos_a2*(x1v*lwv.cos_a3 + x2v*lwv.sin_a3) + x3v*lwv.sin_a2;
      Real sn = std::sin(lwv.k_par*x);
      Real mx = lwv.d0*vflow + amp*sn*rem[1][wave_flag];
      Real my = amp*sn*rem[2][wave_flag];
      Real mz = amp*sn*rem[3][wave_flag];

      // compute cell-centered conserved variables
      u1(m,IDN,k,j,i)=lwv.d0 + amp*sn*rem[0][wave_flag];
      u1(m,IM1,k,j,i)=mx*lwv.cos_a2*lwv.cos_a3 -my*lwv.sin_a3 -mz*lwv.sin_a2*lwv.cos_a3;
      u1(m,IM2,k,j,i)=mx*lwv.cos_a2*lwv.sin_a3 +my*lwv.cos_a3 -mz*lwv.sin_a2*lwv.sin_a3;
      u1(m,IM3,k,j,i)=mx*lwv.sin_a2                           +mz*lwv.cos_a2;

      if (eos.is_ideal) {
        u1(m,IEN,k,j,i) = p0/gm1 + 0.5*lwv.d0*SQR(lwv.v1_0) +
         amp*sn*rem[4][wave_flag] + 0.5*(SQR(lwv.b1_0) + SQR(lwv.b2_0) + SQR(lwv.b3_0));
      }

        // Compute face-centered fields from curl(A).
        Real dx1 = size.d_view(m).dx1;
        Real dx2 = size.d_view(m).dx2;
        Real dx3 = size.d_view(m).dx3;

        b1.x1f(m,k,j,i) = (a3(m,k,j+1,i) - a3(m,k,j,i))/dx2 -
                          (a2(m,k+1,j,i) - a2(m,k,j,i))/dx3;
        b1.x2f(m,k,j,i) = (a1(m,k+1,j,i) - a1(m,k,j,i))/dx3 -
                          (a3(m,k,j,i+1) - a3(m,k,j,i))/dx1;
        b1.x3f(m,k,j,i) = (a2(m,k,j,i+1) - a2(m,k,j,i))/dx1 -
                          (a1(m,k,j+1,i) - a1(m,k,j,i))/dx2;

        // Include extra face-component at edge of block in each direction
        if (i==ie) {
          b1.x1f(m,k,j,i+1) = (a3(m,k,j+1,i+1) - a3(m,k,j,i+1))/dx2 -
                              (a2(m,k+1,j,i+1) - a2(m,k,j,i+1))/dx3;
        }
        if (j==je) {
          b1.x2f(m,k,j+1,i) = (a1(m,k+1,j+1,i) - a1(m,k,j+1,i))/dx3 -
                              (a3(m,k,j+1,i+1) - a3(m,k,j+1,i))/dx1;
        }
        if (k==ke) {
          b1.x3f(m,k+1,j,i) = (a2(m,k+1,j,i+1) - a2(m,k+1,j,i))/dx1 -
                              (a1(m,k+1,j+1,i) - a1(m,k+1,j,i))/dx2;
        }
    });

    // Compute cell-centered fields
    auto &bcc_ = pmbp->pmhd->bcc0;
    par_for("pgen_bcc", DevExeSpace(), 0,nmb-1,ks,ke,js,je,is,ie,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      // cell-centered fields are simple linear average of face-centered fields
      Real& w_bx = bcc_(m,IBX,k,j,i);
      Real& w_by = bcc_(m,IBY,k,j,i);
      Real& w_bz = bcc_(m,IBZ,k,j,i);
      w_bx = 0.5*(b0.x1f(m,k,j,i) + b0.x1f(m,k,j,i+1));
      w_by = 0.5*(b0.x2f(m,k,j,i) + b0.x2f(m,k,j+1,i));
      w_bz = 0.5*(b0.x3f(m,k,j,i) + b0.x3f(m,k+1,j,i));
    });


  }  // End initialization MHD variables

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void MHDEigensystem()
//! \brief computes eigenvectors of linear waves in ideal gas/isothermal mhd

void MHDEigensystem_rel(const Real d, const Real v1, const Real v2, const Real v3,
                    const Real p, const Real b1, const Real b2, const Real b3,
                    const Real x, const Real y, const EOS_Data &eos,
                    Real eigenvalues[1], Real right_eigenmatrix[9]) {

  Real fluid_b[4];
  Real alpha1[4], alpha2[4];
  Real g1, g2, f1, f2;

  //--- Ideal Gas MHD ---
  if (eos.is_ideal) {
    Real vsq = v1*v1 + v2*v2 + v3*v3;       // Initial vsq = v_iv^i for Minkowski
    Real gm1 = eos.gamma - 1.0;

    // Compute relativistic Alfven speed in Minkowski Metric
    Real rhoh = d + (eos.gamma / gm1) * p; //  Total Enthalpy
    Real B_sq = b1*b1 + b2*b2 + b3*b3;
    Real vaxsq = B_sq / (rhoh + bsq);     // For now simplifying for the initial B = (B0,0,0)

    Real vax  = std::sqrt(vaxsq);
    eigenvalues[0] = v1 - vax;

    Real lor = sqrt(1./(1. - vsq));

    fluid_b[0] = lor * (b1 * v1 + b2 * v2 + b3 * v3); // b^0 = Lorentz * (B dot v)
    fluid_b[1] = b1 / lor + fluid_b[0] * v1;
    fluid_b[2] = b2 / lor + fluid_b[0] * v2;
    fluid_b[3] = b3 / lor + fluid_b[0] * v3;

    Real bsq = B_sq / (lor * lor) + pow((b1 * v1 + b2 * v2 + b3 * v3),2);
    Real lamda_a = (fluid_b[1] + sqrt(rhoh + bsq) * v1)/(fluid_b[0] + sqrt(rhoh + bsq) * lor);

    alpha1[0] = lor * v3;
    alpha1[1] = lor * lamda_a * v3;
    alpha1[2] = 0;
    alpha1[3] = 1. - lamda_a * v1;

    alpha2[0] = -lor * v2;
    alpha2[1] = -lor * v2;
    alpha2[2] = -lor * (1. - lamda_a * v1);
    alpha2[3] = 0.;

    g1 = (1/lor) * (b2 + (lamda_a * v2/(1. - lamda_a * v1)) * b1);
    g2 = (1/lor) * (b3 + (lamda_a * v3/(1. - lamda_a * v1)) * b1);

    f1 = g1/sqrt(g1 * g1 + g2 * g2);
    f2 = g2/sqrt(g1 * g1 + g2 * g2);

    right_eigenmatrix[0] = f1 * alpha1[0] + f2 * alpha2[0];
    right_eigenmatrix[1] = f1 * alpha1[1] + f2 * alpha2[1];
    right_eigenmatrix[2] = f1 * alpha1[2] + f2 * alpha2[2];
    right_eigenmatrix[3] = f1 * alpha1[3] + f2 * alpha2[3];

    right_eigenmatrix[4] = - sqrt(rhoh + bsq) * (f1 * alpha1[0] + f2 * alpha2[0]);
    right_eigenmatrix[5] = - sqrt(rhoh + bsq) * (f1 * alpha1[1] + f2 * alpha2[1]);
    right_eigenmatrix[6] = - sqrt(rhoh + bsq) * (f1 * alpha1[2] + f2 * alpha2[2]);
    right_eigenmatrix[7] = - sqrt(rhoh + bsq) * (f1 * alpha1[3] + f2 * alpha2[3]);

    right_eigenmatrix[8] = 0.;
    right_eigenmatrix[9] = 0.;
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void LinearWaveErrors_()
//! \brief Computes errors in linear wave solution by calling initialization function
//! again to compute initial condictions, and subtracting current solution from ICs, and
//! outputs errors to file. Problem must be run for an integer number of wave periods.

void LinearWaveErrors_alfven(ParameterInput *pin, Mesh *pm) {
  // calculate reference solution by calling pgen again.  Solution stored in second
  // register u1/b1 when flag is false.
  set_initial_conditions = false;
  pm->pgen->UserProblem(pin, false);

  Real l1_err[8];
  Real linfty_err=0.0;
  int nvars=0;

  // capture class variables for kernel
  auto &indcs = pm->mb_indcs;
  int &nx1 = indcs.nx1;
  int &nx2 = indcs.nx2;
  int &nx3 = indcs.nx3;
  int &is = indcs.is;
  int &js = indcs.js;
  int &ks = indcs.ks;
  MeshBlockPack *pmbp = pm->pmb_pack;
  auto &size = pmbp->pmb->mb_size;

  // compute errors for Hydro  -----------------------------------------------------------
  if (pmbp->phydro != nullptr) {
    nvars = pmbp->phydro->nhydro;

    EOS_Data &eos = pmbp->phydro->peos->eos_data;
    auto &u0_ = pmbp->phydro->u0;
    auto &u1_ = pmbp->phydro->u1;

    const int nmkji = (pmbp->nmb_thispack)*nx3*nx2*nx1;
    const int nkji = nx3*nx2*nx1;
    const int nji  = nx2*nx1;
    array_sum::GlobalSum sum_this_mb;
    Kokkos::parallel_reduce("LW-err",Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
    KOKKOS_LAMBDA(const int &idx, array_sum::GlobalSum &mb_sum, Real &max_err) {
      // compute n,k,j,i indices of thread
      int m = (idx)/nkji;
      int k = (idx - m*nkji)/nji;
      int j = (idx - m*nkji - k*nji)/nx1;
      int i = (idx - m*nkji - k*nji - j*nx1) + is;
      k += ks;
      j += js;

      Real vol = size.d_view(m).dx1*size.d_view(m).dx2*size.d_view(m).dx3;

      // conserved variables:
      array_sum::GlobalSum evars;
      evars.the_array[IDN] = vol*fabs(u0_(m,IDN,k,j,i) - u1_(m,IDN,k,j,i));
      max_err = fmax(max_err, evars.the_array[IDN]);
      evars.the_array[IM1] = vol*fabs(u0_(m,IM1,k,j,i) - u1_(m,IM1,k,j,i));
      max_err = fmax(max_err, evars.the_array[IM1]);
      evars.the_array[IM2] = vol*fabs(u0_(m,IM2,k,j,i) - u1_(m,IM2,k,j,i));
      max_err = fmax(max_err, evars.the_array[IM2]);
      evars.the_array[IM3] = vol*fabs(u0_(m,IM3,k,j,i) - u1_(m,IM3,k,j,i));
      max_err = fmax(max_err, evars.the_array[IM3]);
      if (eos.is_ideal) {
        evars.the_array[IEN] = vol*fabs(u0_(m,IEN,k,j,i) - u1_(m,IEN,k,j,i));
        max_err = fmax(max_err, evars.the_array[IEN]);
      }

      // fill rest of the_array with zeros, if narray < NREDUCTION_VARIABLES
      for (int n=nvars; n<NREDUCTION_VARIABLES; ++n) {
        evars.the_array[n] = 0.0;
      }

      // sum into parallel reduce
      mb_sum += evars;
    }, Kokkos::Sum<array_sum::GlobalSum>(sum_this_mb), Kokkos::Max<Real>(linfty_err));

    // store data into l1_err array
    for (int n=0; n<nvars; ++n) {
      l1_err[n] = sum_this_mb.the_array[n];
    }
  }

  // compute errors for MHD  -------------------------------------------------------------
  if (pmbp->pmhd != nullptr) {
    nvars = pmbp->pmhd->nmhd + 3;  // include 3-compts of cell-centered B in errors

    EOS_Data &eos = pmbp->pmhd->peos->eos_data;
    auto &u0_ = pmbp->pmhd->u0;
    auto &u1_ = pmbp->pmhd->u1;
    auto &b0_ = pmbp->pmhd->b0;
    auto &b1_ = pmbp->pmhd->b1;

    const int nmkji = (pmbp->nmb_thispack)*nx3*nx2*nx1;
    const int nkji = nx3*nx2*nx1;
    const int nji  = nx2*nx1;
    array_sum::GlobalSum sum_this_mb;
    Kokkos::parallel_reduce("LW-err-Sums",Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
    KOKKOS_LAMBDA(const int &idx, array_sum::GlobalSum &mb_sum, Real &max_err) {
      // compute n,k,j,i indices of thread
      int m = (idx)/nkji;
      int k = (idx - m*nkji)/nji;
      int j = (idx - m*nkji - k*nji)/nx1;
      int i = (idx - m*nkji - k*nji - j*nx1) + is;
      k += ks;
      j += js;

      Real vol = size.d_view(m).dx1*size.d_view(m).dx2*size.d_view(m).dx3;

      // conserved variables:
      array_sum::GlobalSum evars;
      evars.the_array[IDN] = vol*fabs(u0_(m,IDN,k,j,i) - u1_(m,IDN,k,j,i));
      max_err = fmax(max_err, evars.the_array[IDN]);
      evars.the_array[IM1] = vol*fabs(u0_(m,IM1,k,j,i) - u1_(m,IM1,k,j,i));
      max_err = fmax(max_err, evars.the_array[IM1]);
      evars.the_array[IM2] = vol*fabs(u0_(m,IM2,k,j,i) - u1_(m,IM2,k,j,i));
      max_err = fmax(max_err, evars.the_array[IM2]);
      evars.the_array[IM3] = vol*fabs(u0_(m,IM3,k,j,i) - u1_(m,IM3,k,j,i));
      max_err = fmax(max_err, evars.the_array[IM3]);
      if (eos.is_ideal) {
        evars.the_array[IEN] = vol*fabs(u0_(m,IEN,k,j,i) - u1_(m,IEN,k,j,i));
        max_err = fmax(max_err, evars.the_array[IEN]);
      }

      // cell-centered B
      Real bcc0 = 0.5*(b0_.x1f(m,k,j,i) + b0_.x1f(m,k,j,i+1));
      Real bcc1 = 0.5*(b1_.x1f(m,k,j,i) + b1_.x1f(m,k,j,i+1));
      evars.the_array[IEN+1] = vol*fabs(bcc0 - bcc1);
      max_err = fmax(max_err, evars.the_array[IEN+1]);

      bcc0 = 0.5*(b0_.x2f(m,k,j,i) + b0_.x2f(m,k,j+1,i));
      bcc1 = 0.5*(b1_.x2f(m,k,j,i) + b1_.x2f(m,k,j+1,i));
      evars.the_array[IEN+2] = vol*fabs(bcc0 - bcc1);
      max_err = fmax(max_err, evars.the_array[IEN+2]);

      bcc0 = 0.5*(b0_.x3f(m,k,j,i) + b0_.x3f(m,k+1,j,i));
      bcc1 = 0.5*(b1_.x3f(m,k,j,i) + b1_.x3f(m,k+1,j,i));
      evars.the_array[IEN+3] = vol*fabs(bcc0 - bcc1);
      max_err = fmax(max_err, evars.the_array[IEN+3]);

      // fill rest of the_array with zeros, if narray < NREDUCTION_VARIABLES
      for (int n=nvars; n<NREDUCTION_VARIABLES; ++n) {
        evars.the_array[n] = 0.0;
      }

      // sum into parallel reduce
      mb_sum += evars;
    }, Kokkos::Sum<array_sum::GlobalSum>(sum_this_mb), Kokkos::Max<Real>(linfty_err));

    // store data into l1_err array
    for (int n=0; n<nvars; ++n) {
      l1_err[n] = sum_this_mb.the_array[n];
    }
  }

#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE, &l1_err, nvars, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &linfty_err, 1, MPI_ATHENA_REAL, MPI_MAX, MPI_COMM_WORLD);
#endif

  // normalize errors by number of cells
  Real vol=  (pmbp->pmesh->mesh_size.x1max - pmbp->pmesh->mesh_size.x1min)
            *(pmbp->pmesh->mesh_size.x2max - pmbp->pmesh->mesh_size.x2min)
            *(pmbp->pmesh->mesh_size.x3max - pmbp->pmesh->mesh_size.x3min);
  for (int i=0; i<nvars; ++i) l1_err[i] = l1_err[i]/vol;
  linfty_err /= vol;

  // compute rms error
  Real rms_err = 0.0;
  for (int i=0; i<nvars; ++i) {
    rms_err += SQR(l1_err[i]);
  }
  rms_err = std::sqrt(rms_err);

  // root process opens output file and writes out errors
  if (global_variable::my_rank == 0) {
    std::string fname;
    fname.assign(pin->GetString("job","basename"));
    fname.append("-errs.dat");
    FILE *pfile;

    // The file exists -- reopen the file in append mode
    if ((pfile = std::fopen(fname.c_str(), "r")) != nullptr) {
      if ((pfile = std::freopen(fname.c_str(), "a", pfile)) == nullptr) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl << "Error output file could not be opened" <<std::endl;
        std::exit(EXIT_FAILURE);
      }

    // The file does not exist -- open the file in write mode and add headers
    } else {
      if ((pfile = std::fopen(fname.c_str(), "w")) == nullptr) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl << "Error output file could not be opened" <<std::endl;
        std::exit(EXIT_FAILURE);
      }
      std::fprintf(pfile, "# Nx1  Nx2  Nx3   Ncycle  RMS-L1    L-infty       ");
      std::fprintf(pfile,"d_L1         M1_L1         M2_L1         M3_L1         E_L1");
      if (pmbp->pmhd != nullptr) {
        std::fprintf(pfile,"          B1_L1         B2_L1         B3_L1");
      }
      std::fprintf(pfile, "\n");
    }

    // write errors
    std::fprintf(pfile, "%04d", pmbp->pmesh->mesh_indcs.nx1);
    std::fprintf(pfile, "  %04d", pmbp->pmesh->mesh_indcs.nx2);
    std::fprintf(pfile, "  %04d", pmbp->pmesh->mesh_indcs.nx3);
    std::fprintf(pfile, "  %05d  %e %e", pmbp->pmesh->ncycle, rms_err, linfty_err);
    for (int i=0; i<nvars; ++i) {
      std::fprintf(pfile, "  %e", l1_err[i]);
    }
    std::fprintf(pfile, "\n");
    std::fclose(pfile);
  }

  return;
}
