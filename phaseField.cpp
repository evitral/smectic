/*****************************************
*                                        *
*        NUMERICAL SCHEME FOR THE        *
*           SMECTIC PHASE FIELD:         *
*     FINITE-DIFFERENCE SEMI-IMPLICIT    *
*       COORDINATE SPLITTING SCHEME      *
*                                        *
*****************************************/

#include<iostream>
#include<fstream>  // read, write
#include<cassert>
#include<vector>
#include<cmath>   // pow, fabs
#include<cstdlib> // rand
#include<ctime>
#include<complex>
#include<Eigen/Dense>
#include<eigen3/unsupported/Eigen/FFT>


/* 0a. DEFINITIONS AND PROTOTYPES */

// 0a-i. Functions for the Phase Field

Eigen::MatrixXd f1 (Eigen::MatrixXd psi, Eigen::MatrixXd u,
		    Eigen::MatrixXd v);

Eigen::MatrixXd f2 (Eigen::MatrixXd psi, Eigen::MatrixXd psi2,
		    Eigen::MatrixXd u, Eigen::MatrixXd v);

Eigen::MatrixXd penLx (Eigen::VectorXd diag1, Eigen::VectorXd diag2,
		       Eigen::VectorXd diag3, Eigen::MatrixXd psi);

Eigen::MatrixXd penLy (Eigen::VectorXd diag1, Eigen::VectorXd diag2,
		       Eigen::VectorXd diag3, Eigen::MatrixXd psi);

Eigen::VectorXd initEV (Eigen::VectorXd vec, double val);

// 0a-ii. Functions for the momentum equation

Eigen::MatrixXcd fft2 (Eigen::MatrixXd in);

Eigen::MatrixXd ifft2(Eigen::MatrixXcd out);

std::pair<Eigen::MatrixXd,Eigen::MatrixXd> makeSr (Eigen::MatrixXd psi);

std::pair<Eigen::MatrixXcd,Eigen::MatrixXcd> makeVc (Eigen::MatrixXcd Su,
						     Eigen::MatrixXcd Sv);

// PRESSURE
Eigen::MatrixXcd PRESSURE (Eigen::MatrixXcd Su, Eigen::MatrixXcd Sv);


/*****************************************
*                                        *
*        GENERAL INFORMATION             *
*                                        *
*    Mesh type: Regular                  *
*    BC: Periodic                        *
*    Geometry: Square (dx = dy)          *
*                                        *
*  - Nonlinear terms and advection are   *
*    written explicitely in F            *
*                                        *
*****************************************/

/* 0b. PARAMETERS AND CONSTANTS */

int readPsi = 0;  // 0: new  != 0: read
  
const int np    = 256;      // number of points
const double L  = 32;       // length of the system
const double dx = L/np;
const double dy = dx;

const double dt = 0.01;

// OBS: Coexistence: e = -1.5, a = 1.0, b = 3.0, gam = 1.0

const double epsilon = -1.5;
const double alpha   =  1.0;
const double beta    =  3.0;
const double gam     =  1.0;
  
const double qo = 1.0;   // wavenumber

// 0b-i. Phase Field constants

const double cf1 = 0.5*epsilon;
const double cf2 = alpha*pow(qo,2)*pow(dx,-2);
const double cf3 = alpha*pow(dx,-4);
const double cf4 = 0.5/(2*dx);

const double beta1d5 = 1.5*beta;
const double beta0d5 = 0.5*beta;
const double gam1d5  = 1.5*gam;
const double gam2d5  = 2.5*gam;

const int npm  = np-1;
const int npmm = np-2;

// 0b-ii. Momentum equation constants

  const double dy4   =   1/pow(dx,4);
  const double dx4   =   1/pow(dx,4);
  const double dx2y2 = 1/(pow(dx,4));
  const double dy2   =   1/pow(dx,2);
  const double dx2   =   1/pow(dx,2);
  const double dy1   =      1/(2*dx);
  const double dx1   =      1/(2*dx);
  const double qo2   =     pow(qo,2);
  const double qo4   =     pow(qo,4);

  const double PI    = 3.141592653589793;
  const double eta   =               1.0;
  const double deta  =             1/eta;
  const double dqx   =      2*PI/(dx*np);

  double cSr;
  Eigen::MatrixXd SrU(np,np);
  Eigen::MatrixXd SrV(np,np);
  std::pair <Eigen::MatrixXd,Eigen::MatrixXd> Sr;

  double dpx2, dpy2, dpx4, dpy4, dpx2y2;

// 0b-iii. Fourier solver parameters

    std::pair<Eigen::MatrixXcd,Eigen::MatrixXcd> Vc;
    Eigen::VectorXd Vqx(np);
    Eigen::VectorXd Vqy(np);
    Eigen::MatrixXd Mdetaq2(np,np);
    Eigen::MatrixXd Mdetaq4(np,np);
    Eigen::MatrixXcd Vcu(np,np);
    Eigen::MatrixXcd Vcv(np,np);
    int npmi, npmj;
    int npd2 = np/2;


// ob-iv. Misc

    int ip, ipp, im, imm, jp, jpp, jm, jmm;


/*****************************************
*                                        *
*              MAIN TASKS                *
*                                        *
*    1.Create storage matrices           *
*    2.Set initial conditions            *
*    3.Mount splitting operators         *
*    4.Assembly matrices M1-4            *  
*    5.Time loop                         *
*    6.Internal iterations               *
*    7.Compute new psi field             *
*    8.Compute velocity field            *
*    9.Post data acquisition tasks       *
*                                        *
*****************************************/


int main(void)
{

  /* 1a. MATRICES: PSI FIELD */

  Eigen::MatrixXd psiOld (np,np);

  Eigen::MatrixXd psiK (np,np); // current psiK

  Eigen::MatrixXd psiKstore (np,np); // store previous psiK

  Eigen::MatrixXd psiKbig (np,np); // psiK + BC (delete?)

  Eigen::MatrixXd psiTil (np,np);

  Eigen::MatrixXd psiNew (np,np);

  Eigen::MatrixXd F1 (np,np);    // Explicit terms

  Eigen::MatrixXd F2 (np,np);    // Semi-implicit terms

  Eigen::MatrixXd Vu (np,np);    // Velocity X

  Eigen::MatrixXd Vv (np,np);    // Velocity Y


  /* 1b. VELOCITY EQUATION MATRICES */

  Eigen::MatrixXcd Su (np,np);   // Stress divergence (X)

  Eigen::MatrixXcd Sv (np,np);   // Stress divergence (Y)

  // PRESSURE

  Eigen::MatrixXd PR (np,np);
  Eigen::MatrixXcd PRc (np,np);


  // 1b-i. Fourier modes vectors and usefull matrices
  
  double mq2;

  for (int i = 0; i < npd2; i++) {
    Vqx(i) = i*dqx;
    Vqy(i) = i*dqx;
    Vqx(npd2+i) = -(npd2-i)*dqx;
    Vqy(npd2+i) = -(npd2-i)*dqx;
  }

  Mdetaq2(0,0) = 0;
  Mdetaq4(0,0) = 0;

  for (int j = 1; j < np; j++){
    mq2 = Vqy(j)*Vqy(j);
    Mdetaq2(0,j) = deta/mq2;
    Mdetaq4(0,j) = deta/(mq2*mq2);
  }

  for (int i = 1; i < np; i++) {
    for (int j = 0; j < np; j++) {
      mq2 = Vqx(i)*Vqx(i) + Vqy(j)*Vqy(j);
      Mdetaq2(i,j) = deta/mq2;
      Mdetaq4(i,j) = deta/(mq2*mq2);
    }
  }


  /* 2a. INITIAL CONDITION: PSI */

  srand (time(NULL));  // random seed
  double Amp = 1.0;    // amplitude
  double qi  = qo;     // wavenumber

  if (readPsi == 0)
  {

    // 2a-i. Initial structure - New

    for ( int j = 28; j < 224; j++){
      double cond  = Amp*sin(qi*(j)*dx);
      for ( int i = 0; i < np; i++) {
	//psiNew(i,j) = 0.0002*(rand() % 10001) - 1.0;
	psiNew(i,j) =  cond; // j-16
      }
    }
  } 
  else{

    // 2a-ii. Initial structure - Read profile data

    std::ifstream psidata("psiRead.dat");
    assert(psidata.is_open());

    std::cout << "Reading from the file" << std::endl;

    for ( int j = 0; j < np ; j++)
    {
      for ( int i = 0; i < np ; i++)
      {
	psidata >> psiNew(i,j);
      }
    }

    psidata.close();

  }

  /* 2a-iii. Small perturbation

  for (int j = 24; j < 104; j++)    // 60 ; 198
  {
    for ( int i = 0; i < np ; i++)
    {
        psiNew(i,j) = psiNew(i,j) + Amp*qi*0.1*sin(qi*j*dx)*cos(0.25*i*dx);
    }
  }

  */

  /* 2b. INITIAL CONDITION: VELOCITY */

  for ( int i = 0; i < np; i++) {
    for ( int j = 0; j < np; j++) {
      Vu(i,j) = 0;
    }
  }


  /* 3a. OPERATOR Lx */

  Eigen::VectorXd LxV1 (np);
  LxV1 = initEV(LxV1,-alpha/(2*pow(dx,4.0)));

  Eigen::VectorXd LxV2 (np);
  LxV2 = initEV(LxV2,2*alpha/pow(dx,4.0));

  Eigen::VectorXd LxV3 (np);
  LxV3 = initEV(LxV3,-3*alpha/pow(dx,4.0)-alpha*pow(qo,4.0)/4.0);


  /* 3b. OPERATOR Ly */

  Eigen::VectorXd LyV1 (np);
  LyV1 = initEV(LyV1,-alpha/(2*pow(dy,4.0)));

  Eigen::VectorXd LyV2 (np);
  LyV2 = initEV(LyV2,2*alpha/pow(dy,4.0));

  Eigen::VectorXd LyV3 (np);
  LyV3 = initEV(LyV3,-3*alpha/pow(dy,4.0)-alpha*pow(qo,4.0)/4.0);

  
  /* 4. MATRICES invM1, M2, M3, invM4 */

  Eigen::VectorXd M1V1 (LxV1.size());
  Eigen::VectorXd M1V2 (LxV2.size());
  Eigen::VectorXd M1V3 (LxV3.size());

  Eigen::VectorXd M2V1 (LxV1.size());
  Eigen::VectorXd M2V2 (LxV2.size());
  Eigen::VectorXd M2V3 (LxV3.size());

  Eigen::VectorXd M3V1 (LyV1.size());
  Eigen::VectorXd M3V2 (LyV2.size());
  Eigen::VectorXd M3V3 (LyV3.size());

  Eigen::VectorXd M4V1 (LyV1.size());
  Eigen::VectorXd M4V2 (LyV2.size());
  Eigen::VectorXd M4V3 (LyV3.size());

  Eigen::MatrixXd invM1 (np,np); // inv I - dt*Lx
  Eigen::MatrixXd M2 (np,np);    // I + dt*Lx
  Eigen::MatrixXd M3 (np,np);    // dt*Ly
  Eigen::MatrixXd invM4 (np,np); // inv I - dt*Ly


  for ( int i = 0; i < LxV1.size(); i++){
    M1V1(i) = -dt*LxV1(i);
    M2V1(i) =  dt*LxV1(i);
    M3V1(i) =  dt*LyV1(i);
    M4V1(i) = -dt*LyV1(i);
  }

  for ( int i = 0; i < LxV2.size(); i++){
    M1V2(i) = -dt*LxV2(i);
    M2V2(i) =  dt*LxV2(i);
    M3V2(i) =  dt*LyV2(i);
    M4V2(i) = -dt*LyV2(i);
  }

  for ( int i = 0; i < LxV3.size(); i++){
    M1V3(i) = 1 - dt*LxV3(i);
    M2V3(i) = 1 + dt*LxV3(i);
    M3V3(i) =     dt*LyV3(i);
    M4V3(i) = 1 - dt*LyV3(i);
  }

  invM1(0,npmm) = M1V1(0);
  invM1(1,npm) =  M1V1(1);
  invM1(npmm,0) = M1V1(npmm);
  invM1(npm,1) =  M1V1(npm);

  invM4(0,npmm) = M4V1(0);
  invM4(1,npm) =  M4V1(1);
  invM4(npmm,0) = M4V1(npmm);
  invM4(npm,1) =  M4V1(npm);

  for (int i = 0; i < npmm ; i++){
    invM1(i,i+2) = M1V1(i);
    invM1(i+2,i) = M1V1(i+2);
    invM4(i,i+2) = M4V1(i);
    invM4(i+2,i) = M4V1(i+2);
  }

  invM1(0,npm) = M1V2(0);
  invM1(npm,0) = M1V2(npm);

  invM4(0,npm) = M4V2(0);
  invM4(npm,0) = M4V2(npm);

  for (int i = 0; i < npm ; i++){
    invM1(i,i+1) = M1V2(i);
    invM1(i+1,i) = M1V2(i+1);
    invM4(i,i+1) = M4V2(i);
    invM4(i+1,i) = M4V2(i+1);
  }

  for (int i = 0; i < np ; i++){
    invM1(i,i) = M1V3(i);
    invM4(i,i) = M4V3(i);
  }

  invM1 = invM1.inverse();
  invM4 = invM4.inverse();


  /*****************************************
  *                                        *
  *    TIME LOOP AND INTERNAL ITERATIONS   *
  *                                        *
  *    Time criterion: L1 < 10**-6         *
  *    In.Iter. criterion: Linf < 10**-8   *
  *                                        *
  *****************************************/
  
  double Linf;
  double L1 = 1; 
  double L1up, L1down;

  double L1Criterion = pow(10,-6);
  double LinfCriterion = pow(10,-8);

  int pass;
  int count = 0;

  std::ofstream L1_output("L1.dat");
  assert(L1_output.is_open());
  L1_output.close();

  /* 5. TIME LOOP */

  while( L1 > L1Criterion){

    count++;

    psiOld = psiNew;

    psiK = psiOld;

    F1 = f1(psiOld,Vu,Vv); // explicit part of F

    M2 = penLx(M2V1,M2V2,M2V3,psiOld); // M2 * psiOld

    M3 = penLy(M3V1,M3V2,M3V3,psiOld); // M3 * psiOld

    pass = 0;

    //clock_t start;
    //clock_t end;

    int count2 = 0;

    /* 6. INTERNAL ITERATIONS */

    while ( pass < 1 ){

      psiKstore = psiK;  // keep previous psi

      F2 = dt*(F1 + f2(psiK,psiOld,Vu,Vv));  // explicit + semiimplicit F

      // 6a. Compute psi til

      for ( int i = 0; i < np; i++)
      {
	psiTil.col(i) = invM1*(M2.col(i)+2*M3.col(i)+F2.col(i));
      }

      // 6b. Compute psi for new internal iteration

      for ( int i = 0; i < np; i++)
      {
	psiK.row(i) = invM4*((psiTil.row(i) - M3.row(i)).transpose());
      }

      // 6c. Calculate the norm Linf

      Linf = (psiK - psiKstore).cwiseAbs().maxCoeff()/
             psiK.cwiseAbs().maxCoeff();

      // 6d-i. STATUS: Convergence attained

      if ( Linf < LinfCriterion )
      {
	pass = 1;
	psiNew = psiK;
      }

      count2 ++;

      // 6d-ii. STATUS: Convergence not attained

      if ( count2 > 100) {
	std::cout << "Convergence failed during internal iterations" 
		  << std::endl;  
	return 0;
      }

    } // INTERNAL ITERATIONS - END


    // 6e-i. Calculate and save L1 + psi, V

    if ( count == 10){

      L1up = 0;
      L1down = 0;

      for ( int i = 0; i < np; i++){
	for (int j = 0; j < np; j++){
	  L1up += fabs(psiNew(i,j)-psiOld(i,j));
	  L1down += fabs(psiNew(i,j));
	}
      }
  
      L1 = L1up/(dt*L1down);

      // double time = (double)(end-start)/CLOCKS_PER_SEC*1000.0;

      std::cout << L1 <<  std::endl;
  
      L1_output.open("L1.dat",std::ios_base::app); // append result
      L1_output << L1 << "\n";
      L1_output.close();

      count = 0;


      // 6e-ii. Optional: Save psi

      std::ofstream psi_output("psi.dat");
      assert(psi_output.is_open());

      for ( int j = 0; j < np ; j++){
	for ( int i = 0; i < np ; i++){
	  psi_output << psiNew(i,j) << "\n ";
	}
      }

      psi_output.close();


      // 6e-iii. Optional: Save velocity field (or Sr)

      std::ofstream velu_output("velu.dat");
      assert(velu_output.is_open());

      std::ofstream velv_output("velv.dat");
      assert(velv_output.is_open());

      /* EXTRA: Calculate the divergence

      Eigen::MatrixXd divV (np,np);

      for(int i=1;i<np-1;i++){for(int j=1;j<np-1;j++){
        divV(i,j) = dx1*(Vu(i+1,j)-Vu(i-1,j))+dy1*(Vu(i,j+1)-Vu(i,j-1));
      }}

      */

      Eigen::MatrixXd gPR (np,np);
      Eigen::MatrixXd lapV (np,np);

      std::ofstream lapV_out("lapV.dat");
      assert(lapV_out.is_open());

      for(int j=0;j<np;j++){
            
      if (j == 0) { jm = npm;}
      else { jm = j-1;}

      if (j == npm) { jp = 0;}
      else { jp = j+1;}

      for(int i=0;i<np;i++){

      gPR(i,j) = dy1*(PR(i,jp)-PR(i,jm));
      lapV(i,j) = eta*dy2*(Vv(i,jp)-2*Vv(i,j)+Vv(i,jm));
      }}

      for ( int j = 0; j < np ; j++){
	for ( int i = 0; i < np ; i++){
	  velu_output << gPR(i,j) << "\n ";
	  velv_output << Sr.second(i,j) << "\n ";
	  lapV_out << lapV(i,j) << "\n ";
	}
      }

      velu_output.close();
      velu_output.close();
      lapV_out.close();

    } // END SAVING

    
    /* 8. VELOCITY EQUATION (NP EVEN) */

    // 8a. Create Sr and move to Fourier Space

    Sr = makeSr(psiNew);
    Su = fft2(Sr.first);
    Sv = fft2(Sr.second);

    // 8b. Obtain velocity in Fourier Space

    Vc = makeVc(Su,Sv);

    PRc = PRESSURE(Su,Sv);

    // 8c. Move velocity to real space

    Vu = ifft2(Vc.first);
    Vv = ifft2(Vc.second);
    
    PR = ifft2(PRc);    

/*

    Vcu(0,0) = 0;  // V = 0 for q = 0, uniform translation
    Vcv(0,0) = 0;

    for (int i = 1; i < np/2; i++) {
      double qx = i*dqx;
      double qy = 0;
      double mq2 = qx*qx;
      double detaq2 = deta/mq2;
      double detaq4 = deta/(mq2*mq2);
      Vcu(i,0) = (detaq2-detaq4*mq2)*Su(i,0);
      Vcv(i,0) = detaq2*Sv(i,0);
      Vcu(np-i,0) = Vcu(i,0);
      Vcv(np-i,0) = Vcv(i,0);
      Vcu(0,i) = detaq2*Su(0,i);
      Vcv(0,i) = (detaq2-detaq4*mq2)*Sv(0,i);
      Vcu(0,np-i) = Vcu(0,i);
      Vcv(0,np-i) = Vcv(0,i);
  }


    for (int i = 1; i < np/2; i++) {
      double qx = i*dqx;
      for (int j = 1; j < np/2; j++) {
	double qy = j*dqx;
	double mq2 = qx*qx+qy*qy;
	double detaq2 = deta/mq2;
	double detaq4 = deta/(mq2*mq2);
	Vcu(i,j) = detaq2*Su(i,j)-detaq4*qx*(qx*Su(i,j)+qy*Sv(i,j));
	Vcv(i,j) = detaq2*Sv(i,j)-detaq4*qy*(qx*Su(i,j)+qy*Sv(i,j));
	Vcu(np-i,np-j) = Vcu(i,j);
	Vcv(np-i,np-j) = Vcv(i,j);

	Vcu(np-i,j) = detaq2*Su(np-i,j)
		      + detaq4*qx*(-qx*Su(np-i,j)+qy*Sv(np-i,j));
	Vcv(np-i,j) = detaq2*Sv(np-i,j)
		      - detaq4*qy*(-qx*Su(np-i,j)+qy*Sv(np-i,j));
	Vcu(i,np-j) = Vcu(np-i,j);
	Vcv(i,np-j) = Vcv(np-i,j);
      }
    }
	
    for (int  i = 0; i < np/2; i++) {
      double qx = i*dqx;
      double qy = -PI/dx;
      double mq2 = qx*qx+qy*qy;
      double detaq2 = deta/mq2;
      double detaq4 = deta/(mq2*mq2);
      Vcu(i,np/2) = detaq2*Su(i,np/2)-detaq4*qx*(qx*Su(i,np/2)+qy*Sv(i,np/2));
      Vcv(i,np/2) = detaq2*Sv(i,np/2)-detaq4*qy*(qx*Su(i,np/2)+qy*Sv(i,np/2));
    }


    for (int i = 1; i < np/2+1; i++) {
      double qx = -i*dqx;
      double qy = -PI/dx;
      double mq2 = qx*qx+qy*qy;
      double detaq2 = deta/mq2;
      double detaq4 = deta/(mq2*mq2);
      Vcu(np-i,np/2) = detaq2*Su(np-i,np/2)
		     - detaq4*qx*(qx*Su(np-i,np/2)+qy*Sv(np-i,np/2));
      Vcv(np-i,np/2) = detaq2*Sv(np-i,np/2)
		     - detaq4*qy*(qx*Su(np-i,np/2)+qy*Sv(np-i,np/2));
    }

   for (int  j = 0; j < np/2; j++) {
      double qy = j*dqx;
      double qx = -PI/dx;
      double mq2 = qx*qx+qy*qy;
      double detaq2 = deta/mq2;
      double detaq4 = deta/(mq2*mq2);
      Vcu(np/2,j) = detaq2*Su(np/2,j)-detaq4*qx*(qx*Su(np/2,j)+qy*Sv(np/2,j));
      Vcv(np/2,j) = detaq2*Sv(np/2,j)-detaq4*qy*(qx*Su(np/2,j)+qy*Sv(np/2,j));
    }

    for (int j = 1; j < np/2; j++) {
      double qy = -j*dqx;
      double qx = -PI/dx;
      double mq2 = qx*qx+qy*qy;
      double detaq2 = deta/mq2;
      double detaq4 = deta/(mq2*mq2);
      Vcu(np/2,np-j) = detaq2*Su(np/2,np-j)
		     - detaq4*qx*(qx*Su(np/2,np-j)+qy*Sv(np/2,np-j));
      Vcv(np/2,np-j) = detaq2*Sv(np/2,np-j)
		     - detaq4*qy*(qx*Su(np/2,np-j)+qy*Sv(np/2,np-j));
    }



    Vu = ifft2(Vcu);
    Vv = ifft2(Vcv);
 */

  } // TIME LOOP - END
  
  L1_output.close();


  /*****************************************
  *                                        *
  *     POST DATA ACQUISITION TASKS        *
  *                                        *
  *    1. Save last psiNew                 *
  *    2. End of routine msg               *
  *                                        *
  *****************************************/
  

  std::ofstream psi_output("psi.dat");
  assert(psi_output.is_open());

  for ( int j = 0; j < np ; j++){
    for ( int i = 0; i < np ; i++){
      psi_output << psiNew(i,j) << "\n ";
    }
  }

  psi_output.close();

  std::cout << "\n" << "END OF ROUTINE"  <<  std::endl;

}


/*****************************************
*                                        *
*              FUNCTIONS                 *
*                                        *
*    f1: Explicit part of f^n+1/2        *
*    f2: Semi-implicit part of f^n+1/2   *
*    penmult: pentadiag x vector         *
*    penLx: pentadiag x matrix (Lx)      *
*    penLy: pentadiag x matrix (Ly)      *
*    initEV: initializes vector with     *
*            constant values             *
*    fft2: 2D FFT                        *
*    ifft2: 2D inverse FFT               *
*    makeSr: divergence of the stress    *
*                                        *
*****************************************/


/* f1 : all nonlinearities are explicit */

Eigen::MatrixXd f1 (Eigen::MatrixXd psi, Eigen::MatrixXd u,
		    Eigen::MatrixXd v )
{
 
  Eigen::MatrixXd f1 (np,np);

  // Vertices

  f1(0,0) = cf1*psi(0,0)
	- cf2*(psi(1,0) + psi(0,1) - 4*psi(0,0) + psi(npm,0)+psi(0,npm))
        - cf3*(psi(1,1) - 2*psi(1,0) + psi(1,npm)
	- 2*(psi(0,1) - 2*psi(0,0) + psi(0,npm))
	+ psi(npm,1) - 2*psi(npm,0) + psi(npm,npm))
	- cf4*(u(0,0)*(psi(1,0) - psi(npm,0))+v(0,0)*(psi(0,1)-psi(0,npm)))
	- beta0d5*pow(psi(0,0),3) + gam1d5*pow(psi(0,0),5);

  f1(0,npm) = cf1*psi(0,npm)
	- cf2*(psi(1,npm) + psi(0,0) - 4*psi(0,npm) + psi(npm,npm)+psi(0,npmm))
        - cf3*(psi(1,0) - 2*psi(1,npm) + psi(1,npmm)
	- 2*(psi(0,0) - 2*psi(0,npm) + psi(0,npmm))
	+ psi(npm,0) - 2*psi(npm,npm) + psi(npm,npmm))
	- cf4*(u(0,npm)*(psi(1,npm) - psi(npm,npm))
	      +v(0,npm)*(psi(0,0)-psi(0,npmm)))
	- beta0d5*pow(psi(0,npm),3) + gam1d5*pow(psi(0,npm),5);

  f1(npm,0) = cf1*psi(npm,0)
	- cf2*(psi(0,0) + psi(npm,1) - 4*psi(npm,0) + psi(npmm,0)+psi(npm,npm))
        - cf3*(psi(0,1) - 2*psi(0,0) + psi(0,npm)
	- 2*(psi(npm,1) - 2*psi(npm,0) + psi(npm,npm))
	+ psi(npmm,1) - 2*psi(npmm,0) + psi(npmm,npm))
	- cf4*(u(npm,0)*(psi(0,0)-psi(npmm,0))
	      +v(npm,0)*(psi(npm,1)-psi(npm,npm)))
	- beta0d5*pow(psi(npm,0),3) + gam1d5*pow(psi(npm,0),5);

  f1(npm,npm) = cf1*psi(npm,npm)
	- cf2*(psi(0,npm) + psi(npm,0) - 4*psi(npm,npm) 
        + psi(npmm,npm)+psi(npm,npmm))
        - cf3*(psi(0,0) - 2*psi(0,npm) + psi(0,npmm)
	- 2*(psi(npm,0) - 2*psi(npm,npm) + psi(npm,npmm))
	+ psi(npmm,0) - 2*psi(npmm,npm) + psi(npmm,npmm))
	- cf4*(u(npm,npm)*(psi(0,npm) - psi(npmm,npm))
	      +v(npm,npm)*(psi(npm,0)-psi(npm,npmm)))
	- beta0d5*pow(psi(npm,npm),3) + gam1d5*pow(psi(npm,npm),5);
  
  for (int i=1; i < npm; i++){

    int ip = i+1;
    int im = i-1;

    // First Column

    f1(i,0) = cf1*psi(i,0)
	- cf2*(psi(ip,0) + psi(i,1) - 4*psi(i,0) + psi(im,0)+psi(i,npm))
        - cf3*(psi(ip,1) - 2*psi(ip,0) + psi(ip,npm)
	- 2*(psi(i,1) - 2*psi(i,0) + psi(i,npm))
	+ psi(im,1) - 2*psi(im,0) + psi(im,npm))
	- cf4*(u(i,0)*(psi(ip,0) - psi(im,0))+v(i,0)*(psi(i,1)-psi(i,npm)))
	- beta0d5*pow(psi(i,0),3) + gam1d5*pow(psi(i,0),5);

    // First Row

    f1(0,i) = cf1*psi(0,i)
	- cf2*(psi(1,i) + psi(0,ip) - 4*psi(0,i) + psi(npm,i)+psi(0,im))
        - cf3*(psi(1,ip) - 2*psi(1,i) + psi(1,im)
	- 2*(psi(0,ip) - 2*psi(0,i) + psi(0,im))
	+ psi(npm,ip) - 2*psi(npm,i) + psi(npm,im))
	- cf4*(u(0,i)*(psi(1,i) - psi(npm,i))+v(0,i)*(psi(0,ip)-psi(0,im)))
	- beta0d5*pow(psi(0,i),3) + gam1d5*pow(psi(0,i),5);
 
    for (int j=1; j < npm; j++){
      
      int jp = j+1;
      int jm = j-1;

      f1(i,j) = cf1*psi(i,j)
	- cf2*(psi(ip,j) + psi(i,jp) - 4*psi(i,j) + psi(im,j)+psi(i,jm))
        - cf3*(psi(ip,jp) - 2*psi(ip,j) + psi(ip,jm)
	- 2*(psi(i,jp) - 2*psi(i,j) + psi(i,jm))
	+ psi(im,jp) - 2*psi(im,j) + psi(im,jm))
	- cf4*(u(i,j)*(psi(ip,j) - psi(im,j))+v(i,j)*(psi(i,jp)-psi(i,jm)))
	- beta0d5*pow(psi(i,j),3) + gam1d5*pow(psi(i,j),5);
    }

    // Last Row

    f1(i,npm) = cf1*psi(i,npm)
	- cf2*(psi(ip,npm) + psi(i,0) - 4*psi(i,npm) + psi(im,npm)+psi(i,npmm))
        - cf3*(psi(ip,0) - 2*psi(ip,npm) + psi(ip,npmm)
	- 2*(psi(i,0) - 2*psi(i,npm) + psi(i,npmm))
	+ psi(im,0) - 2*psi(im,npm) + psi(im,npmm))
	- cf4*(u(i,npm)*(psi(ip,npm)-psi(im,npm))
	      +v(i,npm)*(psi(i,0)-psi(i,npmm)))
	- beta0d5*pow(psi(i,npm),3) + gam1d5*pow(psi(i,npm),5);

    // Last Column   

    f1(npm,i) = cf1*psi(npm,i)
	- cf2*(psi(0,i) + psi(npm,ip) - 4*psi(npm,i) + psi(npmm,i)+psi(npm,im))
        - cf3*(psi(0,ip) - 2*psi(0,i) + psi(0,im)
	- 2*(psi(npm,ip) - 2*psi(npm,i) + psi(npm,im))
	+ psi(npmm,ip) - 2*psi(npmm,i) + psi(npmm,im))
	- cf4*(u(npm,i)*(psi(0,i)-psi(npmm,i))
	      +v(npm,i)*(psi(npm,ip)-psi(npm,im)))
        - beta0d5*pow(psi(npm,i),3) + gam1d5*pow(psi(npm,i),5);
  }

  return f1;
}


/* f2 : all nonlinearities are explicit */

Eigen::MatrixXd f2 (Eigen::MatrixXd psi, Eigen::MatrixXd psi2,
		    Eigen::MatrixXd u, Eigen::MatrixXd v)
{
 
  Eigen::MatrixXd f2 (np,np);

  // Vertices

  f2(0,0) = cf1*psi(0,0)
	- cf2*(psi(1,0) + psi(0,1) - 4*psi(0,0) + psi(npm,0)+psi(0,npm))
        - cf3*(psi(1,1) - 2*psi(1,0) + psi(1,npm)
	- 2*(psi(0,1) - 2*psi(0,0) + psi(0,npm))
	+ psi(npm,1) - 2*psi(npm,0) + psi(npm,npm))
	- cf4*(u(0,0)*(psi(1,0)-psi(npm,0))+v(0,0)*(psi(0,1)-psi(0,npm)))
	+ beta1d5*pow(psi2(0,0),2)*psi(0,0)
	- gam2d5*pow(psi2(0,0),4)*psi(0,0);

  f2(0,npm) = cf1*psi(0,npm)
	- cf2*(psi(1,npm) + psi(0,0) - 4*psi(0,npm) + psi(npm,npm)+psi(0,npmm))
        - cf3*(psi(1,0) - 2*psi(1,npm) + psi(1,npmm)
	- 2*(psi(0,0) - 2*psi(0,npm) + psi(0,npmm))
	+ psi(npm,0) - 2*psi(npm,npm) + psi(npm,npmm))
	- cf4*(u(0,npm)*(psi(1,npm)-psi(npm,npm))
	      +v(0,npm)*(psi(0,0)-psi(0,npmm)))
	+ beta1d5*pow(psi2(0,npm),2)*psi(0,npm)
	- gam2d5*pow(psi2(0,npm),4)*psi(0,npm);

  f2(npm,0) = cf1*psi(npm,0)
	- cf2*(psi(0,0) + psi(npm,1) - 4*psi(npm,0) + psi(npmm,0)+psi(npm,npm))
        - cf3*(psi(0,1) - 2*psi(0,0) + psi(0,npm)
	- 2*(psi(npm,1) - 2*psi(npm,0) + psi(npm,npm))
	+ psi(npmm,1) - 2*psi(npmm,0) + psi(npmm,npm))
	- cf4*(u(npm,0)*(psi(0,0)-psi(npmm,0))
	      +v(npm,0)*(psi(npm,1)-psi(npm,npm)))
	+ beta1d5*pow(psi2(npm,0),2)*psi(npm,0)
	- gam2d5*pow(psi2(npm,0),4)*psi(npm,0);

  f2(npm,npm) = cf1*psi(npm,npm)
	- cf2*(psi(0,npm) + psi(npm,0) - 4*psi(npm,npm) 
        + psi(npmm,npm)+psi(npm,npmm))
        - cf3*(psi(0,0) - 2*psi(0,npm) + psi(0,npmm)
	- 2*(psi(npm,0) - 2*psi(npm,npm) + psi(npm,npmm))
	+ psi(npmm,0) - 2*psi(npmm,npm) + psi(npmm,npmm))
	- cf4*(u(npm,npm)*(psi(0,npm)-psi(npmm,npm))
	      +v(npm,npm)*(psi(npm,0)-psi(npm,npmm)))
	+ beta1d5*pow(psi2(npm,npm),2)*psi(npm,npm)
	- gam2d5*pow(psi2(npm,npm),4)*psi(npm,npm);

  for (int i=1; i < npm; i++){

    int ip = i+1;
    int im = i-1;

    // First Column

    f2(i,0) = cf1*psi(i,0)
	- cf2*(psi(ip,0) + psi(i,1) - 4*psi(i,0) + psi(im,0)+psi(i,npm))
        - cf3*(psi(ip,1) - 2*psi(ip,0) + psi(ip,npm)
	- 2*(psi(i,1) - 2*psi(i,0) + psi(i,npm))
	+ psi(im,1) - 2*psi(im,0) + psi(im,npm))
	- cf4*(u(i,0)*(psi(ip,0)-psi(im,0))+v(i,0)*(psi(i,1)-psi(i,npm)))
	+ beta1d5*pow(psi2(i,0),2)*psi(i,0)
	- gam2d5*pow(psi2(i,0),4)*psi(i,0);

    // First Row

    f2(0,i) = cf1*psi(0,i)
	- cf2*(psi(1,i) + psi(0,ip) - 4*psi(0,i) + psi(npm,i)+psi(0,im))
        - cf3*(psi(1,ip) - 2*psi(1,i) + psi(1,im)
	- 2*(psi(0,ip) - 2*psi(0,i) + psi(0,im))
	+ psi(npm,ip) - 2*psi(npm,i) + psi(npm,im))
	- cf4*(u(0,i)*(psi(1,i)-psi(npm,i))
	      +v(0,i)*(psi(0,ip)-psi(0,im)))
	+ beta1d5*pow(psi2(0,i),2)*psi(0,i)
	- gam2d5*pow(psi2(0,i),4)*psi(0,i);
 
    for (int j=1; j < npm; j++){
      
      int jp = j+1;
      int jm = j-1;

      f2(i,j) = cf1*psi(i,j)
	- cf2*(psi(ip,j) + psi(i,jp) - 4*psi(i,j) + psi(im,j)+psi(i,jm))
        - cf3*(psi(ip,jp) - 2*psi(ip,j) + psi(ip,jm)
	- 2*(psi(i,jp) - 2*psi(i,j) + psi(i,jm))
	+ psi(im,jp) - 2*psi(im,j) + psi(im,jm))
	- cf4*(u(i,j)*(psi(ip,j)-psi(im,j))+v(i,j)*(psi(i,jp)-psi(i,jm)))
	+ beta1d5*pow(psi2(i,j),2)*psi(i,j)
	- gam2d5*pow(psi2(i,j),4)*psi(i,j);
    }

    // Last Row

    f2(i,npm) = cf1*psi(i,npm)
	- cf2*(psi(ip,npm) + psi(i,0) - 4*psi(i,npm) + psi(im,npm)+psi(i,npmm))
        - cf3*(psi(ip,0) - 2*psi(ip,npm) + psi(ip,npmm)
	- 2*(psi(i,0) - 2*psi(i,npm) + psi(i,npmm))
	+ psi(im,0) - 2*psi(im,npm) + psi(im,npmm))
	- cf4*(u(i,npm)*(psi(ip,npm)-psi(im,npm))
	      +v(i,npm)*(psi(i,0)-psi(i,npmm)))
	+ beta1d5*pow(psi2(i,npm),2)*psi(i,npm)
	- gam2d5*pow(psi2(i,npm),4)*psi(i,npm);

    // Last Column   

    f2(npm,i) = cf1*psi(npm,i)
	- cf2*(psi(0,i) + psi(npm,ip) - 4*psi(npm,i) + psi(npmm,i)+psi(npm,im))
        - cf3*(psi(0,ip) - 2*psi(0,i) + psi(0,im)
	- 2*(psi(npm,ip) - 2*psi(npm,i) + psi(npm,im))
	+ psi(npmm,ip) - 2*psi(npmm,i) + psi(npmm,im))
	- cf4*(u(npm,i)*(psi(0,i)-psi(npmm,i))
	      +v(npm,i)*(psi(npm,ip)-psi(npm,im)))
        + beta1d5*pow(psi2(npm,i),2)*psi(npm,i)
	- gam2d5*pow(psi2(npm,i),4)*psi(npm,i);

  }

  return f2;
}


/* penLx : penta x matrix */

Eigen::MatrixXd penLx (Eigen::VectorXd diag1, Eigen::VectorXd diag2,
		       Eigen::VectorXd diag3, Eigen::MatrixXd psi)
{
  Eigen::MatrixXd penLx  (np,np);

  for ( int j = 0; j < np; j++)
    {

      penLx(0,j) = diag1(0)*psi(npmm,j) + diag2(0)*psi(npm,j) 
	         + diag3(0)*psi(0,j) + diag2(0)*psi(1,j) 
	         + diag1(0)*psi(2,j);

      penLx(1,j) = diag1(1)*psi(npm,j) + diag2(1)*psi(0,j) 
	         + diag3(1)*psi(1,j) + diag2(1)*psi(2,j) 
                 + diag1(1)*psi(3,j);

  for ( int i = 2; i < npmm; i++)
    {
      penLx(i,j) = diag1(i)*psi(i-2,j) + diag2(i)*psi(i-1,j) 
	         + diag3(i)*psi(i,j) + diag2(i)*psi(i+1,j) 
	         + diag1(i)*psi(i+2,j);
    }

  penLx(npmm,j) = diag1(npmm)*psi(npmm-2,j) + diag2(npmm)*psi(npmm-1,j) 
                  + diag3(npmm)*psi(npmm,j) + diag2(npmm)*psi(npm,j)
                  + diag1(npmm)*psi(0,j);

  penLx(npm,j) = diag1(npm)*psi(npmm-1,j) + diag2(npm)*psi(npmm,j) 
               + diag3(npm)*psi(npm,j) + diag2(npm)*psi(0,j)
               + diag1(npm)*psi(1,j);
    }

  return penLx;
}


/* penLy : penta x matrix */

Eigen::MatrixXd penLy (Eigen::VectorXd diag1, Eigen::VectorXd diag2,
		       Eigen::VectorXd diag3, Eigen::MatrixXd psi)
{
  Eigen::MatrixXd penLy  (np,np);

  for ( int i = 0; i < np; i++)
    {

      penLy(i,0) = diag1(0)*psi(i,npmm) + diag2(0)*psi(i,npm) 
	         + diag3(0)*psi(i,0) + diag2(0)*psi(i,1) 
	         + diag1(0)*psi(i,2);

      penLy(i,1) = diag1(1)*psi(i,npm) + diag2(1)*psi(i,0) 
	         + diag3(1)*psi(i,1) + diag2(1)*psi(i,2) 
                 + diag1(1)*psi(i,3);

  for ( int j = 2; j < npmm; j++)
    {
      penLy(i,j) = diag1(i)*psi(i,j-2) + diag2(i)*psi(i,j-1) 
	         + diag3(i)*psi(i,j) + diag2(i)*psi(i,j+1) 
	         + diag1(i)*psi(i,j+2);
    }

  penLy(i,npmm) = diag1(npmm)*psi(i,npmm-2) + diag2(npmm)*psi(i,npmm-1) 
                + diag3(npmm)*psi(i,npmm) + diag2(npmm)*psi(i,npm)
                + diag1(npmm)*psi(i,0);

  penLy(i,npm) = diag1(npm)*psi(i,npmm-1) + diag2(npm)*psi(i,npmm) 
               + diag3(npm)*psi(i,npm) + diag2(npm)*psi(i,0)
               + diag1(npm)*psi(i,1);
    }

  return penLy;

}

/* initEv : initializes vector with constant values */

Eigen::VectorXd initEV (Eigen::VectorXd vec, double val)
{
  for ( int i = 0; i < vec.size(); i++){
    vec(i) = val;
  }
  return vec;
}


/* MOMENTUM EQUATION FUNCTIONS */


/* fft2: 2D Fast Fourier Transform */

Eigen::MatrixXcd fft2 (Eigen::MatrixXd in){
    
    Eigen::MatrixXcd out (np,np);
    Eigen::FFT<double> fft;

    // Start iterating on the rows of the input matrix (real)

    for (int k = 0; k < in.rows(); k++) {
        Eigen::VectorXcd tmpOut(np);
        fft.fwd(tmpOut, in.row(k));
        out.row(k) = tmpOut;
    }

    // Iterate over the columns of the output matrix (complex)

    for (int k = 0; k < in.cols(); k++) {
        Eigen::VectorXcd tmpOut(np);
        fft.fwd(tmpOut, out.col(k));
        out.col(k) = tmpOut;
    }

    return out;
}


/* ifft2: 2D Inverse Fourier Transform */

Eigen::MatrixXd ifft2(Eigen::MatrixXcd out){

  Eigen::MatrixXcd iOut(np,np);
  Eigen::FFT<double> fft;

  // Start iterating on the rows of the FFT matrix

    for (int k = 0; k < out.rows(); k++) {
        Eigen::VectorXcd tmpOut(np);
        fft.inv(tmpOut, out.row(k));
        iOut.row(k) = tmpOut;
    }

  // Iterate over the columns of the previous output matrix

    for (int k = 0; k < out.cols(); k++) {
        Eigen::VectorXcd tmpOut(np);
        fft.inv(tmpOut, iOut.col(k));
        iOut.col(k) = tmpOut;
    }

  return iOut.real();

}


/* makeSr: 2D divergence of the stress */

std::pair<Eigen::MatrixXd,Eigen::MatrixXd> makeSr (Eigen::MatrixXd psi){
  
  for (int i = 0; i < np; i++)
  {

    if (i == 0) { im = npm; imm = npmm;}
    else if (i == 1) { im = 0; imm = npm;}
    else { im = i-1; imm = i-2;} 

    if (i == npm) { ip = 0; ipp = 1;}
    else if (i == npmm) { ip = np-1; ipp = 0;}
    else { ip = i+1; ipp = i+2;}

    for (int j = 0; j < np; j++)
    {

      if (j == 0) { jm = npm; jmm = npmm;}
      else if (j == 1) { jm = 0; jmm = npm;}
      else { jm = j-1; jmm = j-2;}

      if (j == npm) { jp = 0; jpp = 1;}
      else if (j == npmm) { jp = npm; jpp = 0;}
      else { jp = j+1; jpp = j+2;}

      dpx2 = dx2*(psi(ip,j)-2*psi(i,j)+psi(im,j));
      dpy2 = dy2*(psi(i,jp)-2*psi(i,j)+psi(i,jm));

      dpx4 = dx4*(psi(ipp,j)+psi(imm,j)-4*(psi(ip,j)+psi(im,j))+6*psi(i,j));
      dpy4 = dy4*(psi(i,jpp)+psi(i,jmm)-4*(psi(i,jp)+psi(i,jm))+6*psi(i,j));

      dpx2y2 = dx2y2*(psi(ip,jp)+psi(ip,jm)+psi(im,jp)+psi(im,jm)
	       -2*(psi(ip,j)+psi(im,j)+psi(i,jp)+psi(i,jm)-2*psi(i,j)));

      cSr = alpha*(qo4*psi(i,j)+2*qo2*(dpx2+dpy2)+(dpx4+2*dpx2y2+dpy4));
      SrU(i,j) = dx1*cSr*(psi(ip,j)-psi(im,j));
      SrV(i,j) = dy1*cSr*(psi(i,jp)-psi(i,jm));
    }
  }

  Sr = std::make_pair (SrU,SrV);

  return Sr;
}



std::pair<Eigen::MatrixXcd,Eigen::MatrixXcd> makeVc (Eigen::MatrixXcd Su,
					             Eigen::MatrixXcd Sv){

    Vcu(0,0) = 0;
    Vcv(0,0) = 0;

    for (int i = 1; i < npd2; i++) {

      npmi = np-i;

      Vcu(i,0) = 0;
      Vcv(i,0) = Mdetaq2(i,0)*Sv(i,0);
      Vcu(npmi,0) = Vcu(i,0);
      Vcv(npmi,0) = Vcv(i,0);
      Vcu(0,i) = Mdetaq2(0,i)*Su(0,i);
      Vcv(0,i) = 0;
      Vcu(0,npmi) = Vcu(0,i);
      Vcv(0,npmi) = Vcv(0,i);

      for (int j = 1; j < npd2; j++) {

	npmj = np-j;

	Vcu(i,j) = Mdetaq2(i,j)*Su(i,j)
		 - Mdetaq4(i,j)*Vqx(i)*(Vqx(i)*Su(i,j)+Vqy(j)*Sv(i,j));
	Vcv(i,j) = Mdetaq2(i,j)*Sv(i,j)
		 - Mdetaq4(i,j)*Vqy(j)*(Vqx(i)*Su(i,j)+Vqy(j)*Sv(i,j));
	Vcu(npmi,npmj) = Vcu(i,j);
	Vcv(npmi,npmj) = Vcv(i,j);

	Vcu(npmi,j) = Mdetaq2(npmi,j)*Su(npmi,j)
		    - Mdetaq4(npmi,j)*Vqx(npmi)*(Vqx(npmi)*Su(npmi,j)
		    + Vqy(j)*Sv(npmi,j));
	Vcv(npmi,j) = Mdetaq2(npmi,j)*Sv(npmi,j)
		    - Mdetaq4(npmi,j)*Vqy(j)*(Vqx(npmi)*Su(npmi,j)
		    + Vqy(j)*Sv(npmi,j));
	Vcu(i,npmj) = Vcu(npmi,j);
	Vcv(i,npmj) = Vcv(npmi,j);
      }
    }
	
    for (int  i = 0; i < npd2; i++) {
      Vcu(i,npd2) = Mdetaq2(i,npd2)*Su(i,npd2)
		  - Mdetaq4(i,npd2)*Vqx(i)*(Vqx(i)*Su(i,npd2)
		  + Vqy(npd2)*Sv(i,npd2));
      Vcv(i,npd2) = Mdetaq2(i,npd2)*Sv(i,npd2)
		  - Mdetaq4(i,npd2)*Vqy(npd2)*(Vqx(i)*Su(i,npd2)
	          + Vqy(npd2)*Sv(i,npd2));

      Vcu(npd2,i) = Mdetaq2(npd2,i)*Su(npd2,i)
		  - Mdetaq4(npd2,i)*Vqx(npd2)*(Vqx(npd2)*Su(npd2,i)
		  + Vqy(i)*Sv(npd2,i));
      Vcv(npd2,i) = Mdetaq2(npd2,i)*Sv(npd2,i)
		  - Mdetaq4(npd2,i)*Vqy(i)*(Vqx(npd2)*Su(npd2,i)
		  + Vqy(i)*Sv(npd2,i));
    }

    for (int i = npd2+1; i < np; i++) {
      Vcu(i,npd2) = Mdetaq2(i,npd2)*Su(i,npd2)
		  - Mdetaq4(i,npd2)*Vqx(i)*(Vqx(i)*Su(i,npd2)
		  + Vqy(npd2)*Sv(i,npd2));
      Vcv(i,npd2) = Mdetaq2(i,npd2)*Sv(i,npd2)
		  - Mdetaq4(i,npd2)*Vqy(npd2)*(Vqx(i)*Su(i,npd2)
		  + Vqy(npd2)*Sv(i,npd2));

      Vcu(npd2,i) = Mdetaq2(npd2,i)*Su(npd2,i)
		  - Mdetaq4(npd2,i)*Vqx(npd2)*(Vqx(npd2)*Su(npd2,i)
		  + Vqy(i)*Sv(npd2,i));
      Vcv(npd2,i) = Mdetaq2(npd2,i)*Sv(npd2,i)
		  - Mdetaq4(npd2,i)*Vqy(i)*(Vqx(npd2)*Su(npd2,i)
		  + Vqy(i)*Sv(npd2,i));
    }

    Vcu(npd2,npd2) = Mdetaq2(npd2,npd2)*Su(npd2,npd2)
		   - Mdetaq4(npd2,npd2)*Vqx(npd2)*(Vqx(npd2)*Su(npd2,npd2)
		   + Vqy(npd2)*Sv(npd2,npd2));
    Vcv(npd2,npd2) = Mdetaq2(npd2,npd2)*Sv(npd2,npd2)
		   - Mdetaq4(npd2,npd2)*Vqy(npd2)*(Vqx(npd2)*Su(npd2,npd2)
		   + Vqy(npd2)*Sv(npd2,npd2));

  Vc = std::make_pair (Vcu,Vcv);

  return Vc;
}

// PRESSURE

Eigen::MatrixXcd PRESSURE (Eigen::MatrixXcd Su, Eigen::MatrixXcd Sv){

  std::complex<double> iU (0.0,1.0);
  Eigen::MatrixXcd PRc (np,np);
  double rho = 1.0;
  double mq2;

  PRc(0,0) = 0;
  PRc(0,0) = 0;

  for (int j = 1; j < np; j++){
    mq2 = Vqy(j)*Vqy(j);
    PRc(0,j) = -rho*iU*(Vqy(j)*Sv(0,j))/mq2;
  }

  for (int i = 1; i < np; i++) {
    for (int j = 0; j < np; j++) {
      mq2 = Vqx(i)*Vqx(i) + Vqy(j)*Vqy(j);
      PRc(i,j) = -rho*iU*(Vqx(i)*Su(i,j)+Vqy(j)*Sv(i,j))/mq2;
    }
  }

  return PRc;
}
