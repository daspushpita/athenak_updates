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
  Real rho, p0, v1_0, v2_0, v3_0, b1_0, b2_0, b3_0, db0, dbx, dby, dbz, k_par;
  Real cos_a2, cos_a3, sin_a2, sin_a3;
  Real du0, dux, duy, duz;
};
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
  Real vx = pin->GetOrAddReal("problem", "vx", 0.0);
  Real vy = pin->GetOrAddReal("problem", "vy", 0.0);
  Real vz = pin->GetOrAddReal("problem", "vz", 0.0);

  Real rho = pin->GetOrAddReal("problem", "rho", 1.0);
  Real p0 = pin->GetOrAddReal("problem", "p0", 1.0);

  Real bx = pin->GetOrAddReal("problem", "bx", 1.0);
  Real by = pin->GetOrAddReal("problem", "by", 0.0);
  Real bz = pin->GetOrAddReal("problem", "bz", 0.0);

  bool along_x1 = pin->GetOrAddBoolean("problem", "along_x1", true);
  bool along_x2 = pin->GetOrAddBoolean("problem", "along_x2", false);
  bool along_x3 = pin->GetOrAddBoolean("problem", "along_x3", false);

  // conditions of periodicity and exactly one wavelength along each grid direction
  Real x1size = pmy_mesh_->mesh_size.x1max - pmy_mesh_->mesh_size.x1min;

  // start with wavevector along x1 axis
  LinWaveVariablesALfven lwv;

  // choose the smallest projection of the wavelength in each direction that is > 0
  Real lambda = x1size;
  // Extract primitive arrays
  auto &w0_ = pmbp->pmhd->w0;

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

  // Initialize k_parallel
  lwv.k_par = 2.0*(M_PI)/lambda;

  // initialize MHD variables ------------------------------------------------------------

  //Get ideal gas EOS
  EOS_Data &eos = pmbp->pmhd->peos->eos_data;
  Real gm1 = eos.gamma - 1.0;

  // Set background state: v1_0 is parallel to wavevector.
  // Similarly, for MHD:   b1_0 is parallel to wavevector, b2_0/b3_0 are perpendicular
  lwv.rho = rho;
  lwv.v1_0 = vx;
  lwv.v2_0 = vy;
  lwv.v3_0 = vz;

  lwv.b1_0 = bx;
  lwv.b2_0 = by;
  lwv.b3_0 = bz;
  
  
  lwv.p0 = p0;
  Real v_sq, W, rhoh;
  Real u4[4], b4[4], delta_u[4], delta_b[4];

  v_sq = SQR(lwv.v1_0) + SQR(lwv.v2_0) + SQR(lwv.v3_0);
  W= 1./std::sqrt(1. - (v_sq));           //Should work only for Minkowski
  rhoh = lwv.rho + eos.gamma * lwv.p0/gm1;

  u4[0] = W;
  u4[1] = u4[0] * lwv.v1_0;
  u4[2] = u4[0] * lwv.v2_0;
  u4[3] = u4[0] * lwv.v3_0;

  Real b0 = lwv.b1_0*u4[1] + lwv.b2_0*u4[2] + lwv.b3_0*u4[3];
  b4[0] = b0;
  b4[1] = (lwv.b1_0 + b4[0] * u4[1]) / u4[0];
  b4[2] = (lwv.b2_0 + b4[0] * u4[2]) / u4[0];
  b4[3] = (lwv.b3_0 + b4[0] * u4[3]) / u4[0];

  Real b_sq = -SQR(b4[0]) + SQR(b4[1]) + SQR(b4[2]) + SQR(b4[3]);
  Real h_tot = rhoh + b_sq;   //Static fluid, b^i = B^i 
  Real lambda_a = (b4[1] + std::sqrt(h_tot) * u4[1]) / (b4[0] + W * std::sqrt(h_tot));

  //Auxilliary variables
  Real alpha1_mu[4], alpha2_mu[4], g_1, g_2, f_1, f_2;

  alpha1_mu[0] = W * u4[3];
  alpha1_mu[1] = W * lambda_a * u4[3];
  alpha1_mu[2] = 0.0;
  alpha1_mu[3] = W * (1 - lambda_a * u4[1]);

  alpha2_mu[0] = - W * u4[2];
  alpha2_mu[1] = - W * lambda_a * u4[2];
  alpha2_mu[2] = - W * (1 - lambda_a * u4[1]);
  alpha2_mu[3] = 0.0;

  g_1 = (1./ W) * (b4[2] + (lambda_a * u4[2] / (1 - lambda_a * u4[1])) * b4[1]);
  g_2 = (1./ W) * (b4[3] + (lambda_a * u4[3] / (1 - lambda_a * u4[1])) * b4[1]);

  if (g_1 == 0.0 && g_2 == 0.0) {
    f_1 = f_2 = 1./std::sqrt(2.0);  // (A 67)
  } else {
    f_1 = g_1 / std::sqrt(SQR(g_1) + SQR(g_2));  // (A 66)
    f_2 = g_2 / std::sqrt(SQR(g_1) + SQR(g_2));  // (A 66)
  }

  //Define perturbations
  for (int mu = 0; mu < 4; ++mu) {
    delta_u[mu] = f_1 * alpha1_mu[mu] + f_2 * alpha2_mu[mu];
    delta_b[mu] = - std::sqrt(h_tot) * delta_u[mu];
  }

  lwv.du0 = delta_u[0];
  lwv.dux = delta_u[1];
  lwv.duy = delta_u[2];
  lwv.duz = delta_u[3];

  lwv.db0 = amp * delta_b[0];
  lwv.dbx = amp * delta_b[1];
  lwv.dby = amp * delta_b[2];
  lwv.dbz = amp * delta_b[3];

  if (set_initial_conditions) {
    Real tlim = pin->GetReal("time", "tlim");
    pin->SetReal("time", "tlim", tlim*(std::abs(lambda/lambda_a)));
  }

  auto &b1 = (set_initial_conditions)? pmbp->pmhd->b0 : pmbp->pmhd->b1;

  // initialize primitive variables
  par_for("pgen_fluid",DevExeSpace(), 0,nmb-1,ks,ke,js,je,is,ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real &x1min = size.d_view(m).x1min;
    Real &x1max = size.d_view(m).x1max;
    Real x1v = CellCenterX(i-is, indcs.nx1, x1min, x1max);
    Real x1f   = LeftEdgeX(i  -is, indcs.nx1, x1min, x1max);
    Real x1fp1   = LeftEdgeX(i+1  -is, indcs.nx1, x1min, x1max);

    Real local_amp = amp * std::sin(lwv.k_par * x1v);

    Real u_mink[4], b_mink[4];

    for (int mu = 0; mu < 4; ++mu) {
      u_mink[mu] = u4[mu] + local_amp * delta_u[mu];
      b_mink[mu] = b4[mu] + local_amp * delta_b[mu];
    }

    // // compute cell-centered conserved variables
    // u1(m,IDN,k,j,i)=lwv.d0 + amp*sn*eigen_vec[6];
    // u1(m,IM1,k,j,i)=mx*lwv.cos_a2*lwv.cos_a3 -my*lwv.sin_a3 -mz*lwv.sin_a2*lwv.cos_a3;
    // u1(m,IM2,k,j,i)=mx*lwv.cos_a2*lwv.sin_a3 +my*lwv.cos_a3 -mz*lwv.sin_a2*lwv.sin_a3;
    // u1(m,IM3,k,j,i)=mx*lwv.sin_a2                           +mz*lwv.cos_a2;

    // u1(m,IEN,k,j,i) = p0/gm1 + 0.5*lwv.d0*SQR(lwv.v1_0) + 0.5*(SQR(lwv.b1_0) + SQR(lwv.b2_0) + SQR(lwv.b3_0));

    // compute cell-centered primitive variables

    w0_(m,IDN,k,j,i) = lwv.rho;
    w0_(m,IEN,k,j,i) = lwv.p0/gm1;

    w0_(m,IVX,k,j,i) = u_mink[1];
    w0_(m,IVY,k,j,i) = u_mink[2];
    w0_(m,IVZ,k,j,i) = u_mink[3];
  });
    //Adding new
  par_for("pgen_fluid2",DevExeSpace(), 0,nmb-1,ks,ke,js,je,is,ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real &x1min = size.d_view(m).x1min;
    Real &x1max = size.d_view(m).x1max;
    Real x1v = CellCenterX(i-is, indcs.nx1, x1min, x1max);
    Real x1f   = LeftEdgeX(i  -is, indcs.nx1, x1min, x1max);
    Real x1fp1   = LeftEdgeX(i+1  -is, indcs.nx1, x1min, x1max);
    Real local_amp = amp * std::sin(lwv.k_par * x1v);
    Real local_ampf = amp * std::sin(lwv.k_par * x1f);
    Real local_ampp = amp * std::sin(lwv.k_par * x1fp1);

    Real u_mink[4], b_mink[4];
    Real u_minkf[4], b_minkf[4];
    Real u_minkp[4], b_minkp[4];

    for (int mu = 0; mu < 4; ++mu) {
      u_mink[mu] = u4[mu] + local_amp * delta_u[mu];
      b_mink[mu] = b4[mu] + local_amp * delta_b[mu];

      u_minkf[mu] = u4[mu] + local_ampf * delta_u[mu];
      b_minkf[mu] = b4[mu] + local_ampf * delta_b[mu];

      u_minkp[mu] = u4[mu] + local_ampp * delta_u[mu];
      b_minkp[mu] = b4[mu] + local_ampp * delta_b[mu];
    }
    b1.x1f(m,k,j,i) = b_minkf[1] * u_minkf[0] - b_minkf[0] * u_minkf[1];
    b1.x2f(m,k,j,i) = b_mink[2] * u_mink[0] - b_mink[0] * u_mink[2];
    b1.x3f(m,k,j,i) = b_mink[3] * u_mink[0] - b_mink[0] * u_mink[3];

    // }
    // // Include extra face-component at edge of block in each direction
    if (i==ie) {
      b1.x1f(m,k,j,i+1) = b_minkp[1] * u_minkp[0] - b_minkp[0] * u_minkp[1];

    }
    if (j==je) {
      b1.x2f(m,k,j+1,i) = b_mink[2] * u_mink[0] - b_mink[0] * u_mink[2];

    }
    if (k==ke) {
      b1.x3f(m,k+1,j,i) = b_mink[3] * u_mink[0] - b_mink[0] * u_mink[3];

    }
    });
  // End initialization MHD variables
  // Compute cell-centered fields
  auto &bcc_ = pmbp->pmhd->bcc0;
  par_for("pgen_b0_cc", DevExeSpace(), 0,nmb-1,ks,ke,js,je,is,ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    // cell-centered fields are simple linear average of face-centered fields
    bcc_(m,IBX,k,j,i) = 0.5*(b1.x1f(m,k,j,i) + b1.x1f(m,k,j,i+1));
    bcc_(m,IBY,k,j,i) = 0.5*(b1.x2f(m,k,j,i) + b1.x2f(m,k,j+1,i));
    bcc_(m,IBZ,k,j,i) = 0.5*(b1.x3f(m,k,j,i) + b1.x3f(m,k+1,j,i));
  });
  // Convert primitives to conserved
  auto &u0_ = pmbp->pmhd->u0;
  pmbp->pmhd->peos->PrimToCons(w0_, bcc_, u0_, is, ie, js, je, ks, ke);
  // End initialization MHD variables
  return;
}
//----------------------------------------------------------------------------------------

