//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file gr_alfven_wave.cpp
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
#include "coordinates/cartesian_ks.hpp"
#include "coordinates/cell_locations.hpp"
#include "eos/eos.hpp"
#include "geodesic-grid/geodesic_grid.hpp"
#include "hydro/hydro.hpp"
#include "mhd/mhd.hpp"
#include "driver/driver.hpp"
#include "mesh/mesh.hpp"
#include "radiation/radiation.hpp"
#include "radiation/radiation_tetrad.hpp"
#include "srcterms/srcterms.hpp"
#include "pgen/pgen.hpp"

// function to compute errors in solution at end of run
void LinearWaveErrors_alfven(ParameterInput *pin, Mesh *pm);

namespace {
// global variable to control computation of initial conditions versus errors
bool set_initial_conditions = true;

//----------------------------------------------------------------------------------------
//! \struct LinWaveVariablesALfven
//! \brief container for variables shared with vector potential and error functions

struct LinWaveVariablesALfven {
  Real d0, p0, v1_0, b1_0, b2_0, b3_0, dby, dbz, k_par;
  Real cos_a2, cos_a3, sin_a2, sin_a3;
  Real dux, duy, duz;
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
  
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;  

  if (!(pmbp->pcoord->is_general_relativistic)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "GR Alfven wave problem can only be run when GR defined in <coord> block"
              << std::endl;
    exit(EXIT_FAILURE);
  }
  if (pmbp->pmhd == nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "GR Alfven wave problem can only be run when MHD enabled via <mhd> block"
              << std::endl;
    exit(EXIT_FAILURE);
  }
  // read global parameters
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
  if (pmy_mesh_->multi_d && !(along_x1)) {
    Real ang_3 = std::atan(x1size/x2size);
    lwv.sin_a3 = std::sin(ang_3);
    lwv.cos_a3 = std::cos(ang_3);
  }
  if (pmy_mesh_->three_d && !(along_x1)) {
    Real ang_2 = std::atan(0.5*(x1size*lwv.cos_a3 + x2size*lwv.sin_a3)/x3size);
    lwv.sin_a2 = std::sin(ang_2);
    lwv.cos_a2 = std::cos(ang_2);
  }

  // hardcode wavevector along x2 axis, override ang_2, ang_3
  if (along_x2) {
    lwv.cos_a3 = 0.0;
    lwv.sin_a3 = 1.0;
    lwv.cos_a2 = 1.0;
    lwv.sin_a2 = 0.0;
  }

  // hardcode wavevector along x3 axis, override ang_2, ang_3
  if (along_x3) {
    lwv.cos_a3 = 0.0;
    lwv.sin_a3 = 1.0;
    lwv.cos_a2 = 0.0;
    lwv.sin_a2 = 1.0;
  }

  // choose the smallest projection of the wavelength in each direction that is > 0
  Real lambda = std::numeric_limits<float>::max();
  if (lwv.cos_a2*lwv.cos_a3 > 0.0) {
    lambda = std::min(lambda, x1size*lwv.cos_a2*lwv.cos_a3);
  }
  if (lwv.cos_a2*lwv.sin_a3 > 0.0) {
    lambda = std::min(lambda, x2size*lwv.cos_a2*lwv.sin_a3);
  }
  if (lwv.sin_a2 > 0.0) lambda = std::min(lambda, x3size*lwv.sin_a2);

  // Initialize k_parallel
  lwv.k_par = 2.0*(M_PI)/lambda;

  // Set background state: v1_0 is parallel to wavevector.
  // Similarly, for MHD:   b1_0 is parallel to wavevector, b2_0/b3_0 are perpendicular
  lwv.d0 = 1.0;
  lwv.v1_0 = vflow;
  lwv.b1_0 = 1.0;
  lwv.b2_0 = 0.;
  lwv.b3_0 = 0.;
  // Extract conserved and primitive arrays
  auto &u0_ = pmbp->pmhd->u0;
  auto &w0_ = pmbp->pmhd->w0;
  //Get ideal gas EOS
  EOS_Data &eos = pmbp->pmhd->peos->eos_data;
  Real gm1 = eos.gamma - 1.0;

  // capture variables for kernel
  auto &indcs = pmy_mesh_->mb_indcs;
  int &is = indcs.is; int &ie = indcs.ie;
  int &js = indcs.js; int &je = indcs.je;
  int &ks = indcs.ks; int &ke = indcs.ke;
  int &nmb = pmbp->nmb_thispack;
  auto &coord = pmbp->pcoord->coord_data;
  auto &size = pmbp->pmb->mb_size;

  int n1 = indcs.nx1 + 2*indcs.ng;
  int n2 = (indcs.nx2 > 1)? (indcs.nx2 + 2*indcs.ng) : 1;
  int n3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*indcs.ng) : 1;


  // initialize MHD variables ------------------------------------------------------------
  Real p0 = 1.0/eos.gamma;

  Real eigen_vec[7];
  Real lfac = 1./std::sqrt(1. - (vflow*vflow));    //Should work only for Minkowski
  Real rhoh = lwv.d0 + eos.gamma * p0/gm1;
  Real epsilon_mhd = rhoh + lwv.b1_0*lwv.b1_0;            //Static fluid, b^i = B^i 
  
  eigen_vec[0] = 0.0;
  eigen_vec[1] = - lfac / std::sqrt(2);
  eigen_vec[2] = lfac / std::sqrt(2);
  eigen_vec[3] = 0.0;
  eigen_vec[4] = lfac * std::sqrt(epsilon_mhd) / std::sqrt(2);
  eigen_vec[5] = -lfac * std::sqrt(epsilon_mhd)  / std::sqrt(2);
  eigen_vec[6] = 0.0;
  eigen_vec[7] = 0.0;

  Real alfven_v = lwv.b1_0/std::sqrt(epsilon_mhd);

  lwv.dux = amp*eigen_vec[0];
  lwv.duy = amp*eigen_vec[1];
  lwv.duz = amp*eigen_vec[2];
  lwv.dby = amp*eigen_vec[4];
  lwv.dbz = amp*eigen_vec[5];

  // set new time limit in ParameterInput (to be read by Driver constructor) based on
  // wave speed of selected mode.
  // input tlim should be interpreted as number of wave periods for evolution
  if (set_initial_conditions) {
    Real tlim = pin->GetReal("time", "tlim");
    pin->SetReal("time", "tlim", tlim*(std::abs(lambda/alfven_v)));
  }

  // initialize primitive variables
  par_for("pgen_fluid",DevExeSpace(), 0,nmb-1,ks,ke,js,je,js,je,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real &x1min = size.d_view(m).x1min;
    Real &x1max = size.d_view(m).x1max;
    Real x1v = CellCenterX(i-is, indcs.nx1, x1min, x1max);

    Real &x2min = size.d_view(m).x2min;
    Real &x2max = size.d_view(m).x2max;
    Real x2v = CellCenterX(j-js, indcs.nx2, x2min, x2max);

    Real &x3min = size.d_view(m).x3min;
    Real &x3max = size.d_view(m).x3max;
    Real x3v = CellCenterX(k-ks, indcs.nx3, x3min, x3max);

    Real x = lwv.cos_a2*(x1v*lwv.cos_a3 + x2v*lwv.sin_a3) + x3v*lwv.sin_a2;
    Real sn = std::sin(lwv.k_par*x);
    Real mx = lwv.d0*vflow + sn* lwv.dux;
    Real my = sn* lwv.duy;
    Real mz = sn* lwv.duz;

    // // compute cell-centered conserved variables
    // u1(m,IDN,k,j,i)=lwv.d0 + amp*sn*eigen_vec[6];
    // u1(m,IM1,k,j,i)=mx*lwv.cos_a2*lwv.cos_a3 -my*lwv.sin_a3 -mz*lwv.sin_a2*lwv.cos_a3;
    // u1(m,IM2,k,j,i)=mx*lwv.cos_a2*lwv.sin_a3 +my*lwv.cos_a3 -mz*lwv.sin_a2*lwv.sin_a3;
    // u1(m,IM3,k,j,i)=mx*lwv.sin_a2                           +mz*lwv.cos_a2;

    // u1(m,IEN,k,j,i) = p0/gm1 + 0.5*lwv.d0*SQR(lwv.v1_0) + 0.5*(SQR(lwv.b1_0) + SQR(lwv.b2_0) + SQR(lwv.b3_0));

    // compute cell-centered primitive variables

      w0_(m,IDN,k,j,i) = lwv.d0;
      w0_(m,IEN,k,j,i) = p0/gm1;

      w0_(m,IVX,k,j,i) = mx*lwv.cos_a2*lwv.cos_a3 -my*lwv.sin_a3 -mz*lwv.sin_a2*lwv.cos_a3;
      w0_(m,IVY,k,j,i) = mx*lwv.cos_a2*lwv.sin_a3 +my*lwv.cos_a3 -mz*lwv.sin_a2*lwv.sin_a3;
      w0_(m,IVZ,k,j,i) = mx*lwv.sin_a2                           +mz*lwv.cos_a2;
  });
  
  // initialize magnetic fields ----------------------------------------------------------

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

  par_for("pgen_aphi", DevExeSpace(), 0,nmb-1,ks,ke+1,js,je+1,is,ie+1,
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
    Real x3v = CellCenterX(k-ks, nx3, x3min, x3max);
    Real x3f   = LeftEdgeX(k  -ks, nx3, x3min, x3max);

    Real dx1 = size.d_view(m).dx1;
    Real dx2 = size.d_view(m).dx2;
    Real dx3 = size.d_view(m).dx3;

    a1(m,k,j,i) = A1(x1v, x2f, x3f, lwv);
    a2(m,k,j,i) = A2(x1f, x2v, x3f, lwv);
    a3(m,k,j,i) = A3(x1f, x2f, x3v, lwv);

    // When neighboring MeshBock is at finer level, compute vector potential as sum of
    // values at fine grid resolution.  This guarantees flux on shared fine/coarse
    // faces is identical.

    // Correct A1 at x2-faces, x3-faces, and x2x3-edges
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

  auto &b0 = pmbp->pmhd->b0;
  par_for("pgen_b0", DevExeSpace(), 0,nmb-1,ks,ke,js,je,is,ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    // Compute face-centered fields from curl(A).
    Real dx1 = size.d_view(m).dx1;
    Real dx2 = size.d_view(m).dx2;
    Real dx3 = size.d_view(m).dx3;

    b0.x1f(m,k,j,i) = ((a3(m,k,j+1,i) - a3(m,k,j,i))/dx2 -
                        (a2(m,k+1,j,i) - a2(m,k,j,i))/dx3);
    b0.x2f(m,k,j,i) = ((a1(m,k+1,j,i) - a1(m,k,j,i))/dx3 -
                        (a3(m,k,j,i+1) - a3(m,k,j,i))/dx1);
    b0.x3f(m,k,j,i) = ((a2(m,k,j,i+1) - a2(m,k,j,i))/dx1 -
                        (a1(m,k,j+1,i) - a1(m,k,j,i))/dx2);

    // Include extra face-component at edge of block in each direction
    if (i==ie) {
      b0.x1f(m,k,j,i+1) = ((a3(m,k,j+1,i+1) - a3(m,k,j,i+1))/dx2 -
                            (a2(m,k+1,j,i+1) - a2(m,k,j,i+1))/dx3);
    }
    if (j==je) {
      b0.x2f(m,k,j+1,i) = ((a1(m,k+1,j+1,i) - a1(m,k,j+1,i))/dx3 -
                            (a3(m,k,j+1,i+1) - a3(m,k,j+1,i))/dx1);
    }
    if (k==ke) {
      b0.x3f(m,k+1,j,i) = ((a2(m,k+1,j,i+1) - a2(m,k+1,j,i))/dx1 -
                            (a1(m,k+1,j+1,i) - a1(m,k+1,j,i))/dx2);
    }
  });

    // Compute cell-centered fields
  auto &bcc_ = pmbp->pmhd->bcc0;
  par_for("pgen_b0_cc", DevExeSpace(), 0,nmb-1,ks,ke,js,je,is,ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    // cell-centered fields are simple linear average of face-centered fields
    Real& w_bx = bcc_(m,IBX,k,j,i);
    Real& w_by = bcc_(m,IBY,k,j,i);
    Real& w_bz = bcc_(m,IBZ,k,j,i);
    w_bx = 0.5*(b0.x1f(m,k,j,i) + b0.x1f(m,k,j,i+1));
    w_by = 0.5*(b0.x2f(m,k,j,i) + b0.x2f(m,k,j+1,i));
    w_bz = 0.5*(b0.x3f(m,k,j,i) + b0.x3f(m,k+1,j,i));
  });

  // Convert primitives to conserved
  pmbp->pmhd->peos->PrimToCons(w0_, bcc_, u0_, is, ie, js, je, ks, ke);

  // End initialization MHD variables
  return;
}
//----------------------------------------------------------------------------------------

