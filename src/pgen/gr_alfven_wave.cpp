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
  
  // set linear wave errors function
  pgen_final_func = LinearWaveErrors_alfven;
  
  if (restart) return;

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
  int wave_flag = pin->GetOrAddInteger("problem", "wave_flag", 1);
  // conditions of periodicity and exactly one wavelength along each grid direction
  Real x1size = pmy_mesh_->mesh_size.x1max - pmy_mesh_->mesh_size.x1min;

  // start with wavevector along x1 axis
  LinWaveVariablesALfven lwv;

  // choose the smallest projection of the wavelength in each direction that is > 0
  Real wavelength = x1size;
  Real lambda; 
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
  lwv.k_par = 2.0*(M_PI)/wavelength;

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
  W= 1./std::sqrt(1. - v_sq);           //Should work only for Minkowski
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

  Real lambda_ap = (b4[1] + std::sqrt(h_tot) * u4[1]) / (b4[0] + W * std::sqrt(h_tot));            // (A 38)
  Real lambda_am = (b4[1] - std::sqrt(h_tot) * u4[1]) / (b4[0] - W * std::sqrt(h_tot));            // (A 38)
  
  Real sign = 1.0;
  
  if (lambda_ap > lambda_am) {  // \lambda_{a,\pm} = \lambda_a^\pm
    if (wave_flag == 1) {  // leftgoing
      sign = -1.0;
    }
  } else {  // lambda_{a,\pm} = \lambda_a^\mp
    if (wave_flag == 5) {  // rightgoing
    sign = -1.0;
    }
  }
  if (sign > 0) {  // want \lambda_{a,+}
    lambda = lambda_ap;
  } else {  // want \lambda_{a,-} instead
    lambda = lambda_am;
  }

  // Real lambda_a = (b4[1] + std::sqrt(h_tot) * u4[1]) / (b4[0] + W * std::sqrt(h_tot));

  //Auxilliary variables
  Real alpha1_mu[4], alpha2_mu[4], g_1, g_2, f_1, f_2;

  alpha1_mu[0] = u4[3];
  alpha1_mu[1] = lambda * u4[3];
  alpha1_mu[2] = 0.0;
  alpha1_mu[3] = u4[0] - lambda * u4[1];

  alpha2_mu[0] = -u4[2];
  alpha2_mu[1] = -lambda * u4[2];
  alpha2_mu[2] = lambda * u4[1] - u4[0];
  alpha2_mu[3] = 0.0;

  g_1 = (1./ W) * (b4[2] + (lambda * u4[2] / (1 - lambda * u4[1])) * b4[1]);
  g_2 = (1./ W) * (b4[3] + (lambda * u4[3] / (1 - lambda * u4[1])) * b4[1]);

  // Safe threshold check for near-zero g_1, g_2
  constexpr Real eps = std::numeric_limits<Real>::epsilon();
  constexpr Real tol = 100 * eps;  // Adjust the multiplier if needed

  // if (g_1 == 0.0 && g_2 == 0.0) {
  //   f_1 = f_2 = 1./std::sqrt(2.0);  // (A 67)
  // } else {
  //   f_1 = g_1 / std::sqrt(SQR(g_1) + SQR(g_2));  // (A 66)
  //   f_2 = g_2 / std::sqrt(SQR(g_1) + SQR(g_2));  // (A 66)
  // }

  if (std::abs(g_1) < tol && std::abs(g_2) < tol) {
    f_1 = f_2 = 1./std::sqrt(2.0);  // (A 67)
  } else {
    f_1 = g_1 / std::sqrt(SQR(g_1) + SQR(g_2));
    f_2 = g_2 / std::sqrt(SQR(g_1) + SQR(g_2));
  }

  //Define perturbations
  for (int mu = 0; mu < 4; ++mu) {
    delta_u[mu] = f_1 * alpha1_mu[mu] + f_2 * alpha2_mu[mu];
    delta_b[mu] = - std::sqrt(h_tot) * delta_u[mu];
  }

  // Normalize perturbation like Athena++
  Real pert_size = SQR(delta_u[0]) + SQR(delta_b[0]);
  for (int mu = 1; mu < 4; ++mu) {
    pert_size += SQR(delta_u[mu]) + SQR(delta_b[mu]);
  }
  pert_size = std::sqrt(pert_size);
  for (int mu = 0; mu < 4; ++mu) {
    delta_u[mu] /= pert_size;
    delta_b[mu] /= pert_size;
  }

  lwv.du0 = delta_u[0];
  lwv.dux = delta_u[1];
  lwv.duy = delta_u[2];
  lwv.duz = delta_u[3];

  lwv.db0 = amp * delta_b[0];
  lwv.dbx = amp * delta_b[1];
  lwv.dby = amp * delta_b[2];
  lwv.dbz = amp * delta_b[3];

  std::cout << "delta_u[2] = " << delta_u[2] << std::endl;
  
  if (set_initial_conditions) {
    Real tlim = pin->GetReal("time", "tlim");
    pin->SetReal("time", "tlim", tlim*(std::abs(wavelength/lambda)));
  }
  // Extract primitive arrays
  auto &w0 = pmbp->pmhd->w0;
  //Conserved and magnetic fields
  auto &b1 = (set_initial_conditions)? pmbp->pmhd->b0 : pmbp->pmhd->b1;
  auto &u1 = (set_initial_conditions)? pmbp->pmhd->u0 : pmbp->pmhd->u1;

  // initialize primitive variables
  par_for("pgen_fluid",DevExeSpace(), 0,nmb-1,ks,ke,js,je,is,ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real &x1min = size.d_view(m).x1min;
    Real &x1max = size.d_view(m).x1max;
    Real x1v = CellCenterX(i-is, indcs.nx1, x1min, x1max);

    Real local_amp = amp * std::sin(lwv.k_par * x1v);
    //Adding New
    Real rho_local = rho;
    Real pgas_local = lwv.p0;

    Real u_mink[4];
    Real b_mink[4] = {0.0};

    for (int mu = 0; mu < 4; ++mu) {
      u_mink[mu] = u4[mu] + local_amp * delta_u[mu];
      b_mink[mu] = b4[mu] + local_amp * delta_b[mu];
    }
     
    //New

    Real u_local[4], b_local[4], u_local_low[4], b_local_low[4];
    for (int mu = 0; mu < 4; ++mu) {
      u_local[mu] = u_mink[mu];
      b_local[mu] = b_mink[mu];
      u_local_low[mu] = (mu == 0 ? -1.0 : 1.0) * u_local[mu];
      b_local_low[mu] = (mu == 0 ? -1.0 : 1.0) * b_local[mu];
    }

    // Calculate useful local scalars
    Real b_sq_local = 0.0;
    for (int mu = 0; mu < 4; ++mu) {
      b_sq_local += b_local[mu] * b_local_low[mu];
    }
    Real wtot_local = rho_local + (eos.gamma/gm1) * pgas_local + b_sq_local;
    Real ptot_local = pgas_local + 0.5*b_sq_local;

    // compute cell-centered primitive variables

    w0(m,IDN,k,j,i) = rho_local;
    w0(m,IEN,k,j,i) = pgas_local/gm1;
    w0(m,IVX,k,j,i) = u_local[1] / u_local[0]; 
    w0(m,IVY,k,j,i) = u_local[2] / u_local[0]; 
    w0(m,IVZ,k,j,i) = u_local[3] / u_local[0]; 

    // // compute cell-centered conserved variables
    // u1(m,IDN,k,j,i) = u_local[0] * rho_local;
    // u1(m,IM1,k,j,i) = wtot_local*u_local[0]*u_local[1] - b_local[0]*b_local[1];
    // u1(m,IM2,k,j,i) = wtot_local*u_local[0]*u_local[2] - b_local[0]*b_local[2];
    // u1(m,IM3,k,j,i) = wtot_local*u_local[0]*u_local[3] - b_local[0]*b_local[3];
    // u1(m,IEN,k,j,i) = wtot_local*u_local[0]*u_local[0] - b_local[0]*b_local[0]
    //                   + ptot_local;
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
    b1.x1f(m,k,j,i) = b_mink[1] * u_mink[0] - b_mink[0] * u_mink[1];
    b1.x2f(m,k,j,i) = b_mink[2] * u_mink[0] - b_mink[0] * u_mink[2];
    b1.x3f(m,k,j,i) = b_mink[3] * u_mink[0] - b_mink[0] * u_mink[3];

    // }
    // // Include extra face-component at edge of block in each direction
    if (i==ie) {
      b1.x1f(m,k,j,i+1) = b_mink[1] * u_mink[0] - b_mink[0] * u_mink[1];

    }
    if (j==je) {
      b1.x2f(m,k,j+1,i) = b_mink[2] * u_mink[0] - b_mink[0] * u_mink[2];

    }
    if (k==ke) {
      b1.x3f(m,k+1,j,i) = b_mink[3] * u_mink[0] - b_mink[0] * u_mink[3];

    }
    });
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
  // auto &u0_ = pmbp->pmhd->u0;
  // pmbp->pmhd->peos->PrimToCons(w0_, bcc_, u0_, is, ie, js, je, ks, ke);
  pmbp->pmhd->peos->PrimToCons(w0, bcc_, u1, 0, (n1-1), 0, (n2-1), 0, (n3-1));
  // End initialization MHD variables
  return;
}

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

//----------------------------------------------------------------------------------------

