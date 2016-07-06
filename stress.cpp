#include<iostream>
#include<fstream>  // read
#include<vector>
#include<cassert> // assert
#include<cmath>
#include<Eigen/Dense>

int main(void)
{

  const int np      = 256;
  const double qo   = 1;
  const double size = 32;
  const double dx   = size/np;

  Eigen::MatrixXd     psi(np,np);
  Eigen::MatrixXd stress (np,np);

  /* READ PROFILE DATA */

  std::ifstream psidata("psi.dat");
  assert(psidata.is_open());

  std::cout << "Reading from the file" << std::endl;

  for ( int j = 0; j < np ; j++)
  {
    for ( int i = 0; i < np ; i++)
    {
      psidata >> psi(i,j);
    }
  }

  psidata.close();

  /*  SIGMAxx

  Eigen::MatrixXd psi2(np,np);

  for ( int i = 0; i < np ; i++){for ( int j = 0; j < np ; j++){
    psi2(i,j) = psi(j,i);
  }}
  psi = psi2;

  */

  /* FINITE DIFFERENCE SCHEME: CALCULATE STRESS */

  const double dx2y = 1/(2*pow(dx,3));
  const double dy3  = 1/pow(dx,3);
  const double dy2  = 1/pow(dx,2);
  const double dx2  = 1/pow(dx,2);
  const double dy1  =    1/(2*dx);
  const double qo2  =   pow(qo,2);


  double  dpy3;
  double dpx2y;
  double  dpy2;
  double  dpx2;
  double   dpy;
  
  int ip, ipp, im, imm, jp, jpp, jm, jmm;


  for (int i = 0; i < np; i++)
  {
    for (int j = 0; j < np; j++)
    {
      if (i == 0) { im = np-1; imm = np-2;}
      else if (i == 1) { im = 0; imm = np-1;}
      else { im = i-1; imm = i-2;} 

      if (j == 0) { jm = np-1; jmm = np-2;}
      else if (j == 1) { jm = 0; jmm = np-1;}
      else { jm = j-1; jmm = j-2;}

      if (i == np-1) { ip = 0; ipp = 1;}
      else if (i == np-2) { ip = np-1; ipp = 0;}
      else { ip = i+1; ipp = i+2;}

      if (j == np-1) { jp = 0; jpp = 1;}
      else if (j == np-2) { jp = np-1; jpp = 0;}
      else { jp = j+1; jpp = j+2;}


      dpx2y = dx2y*(psi(ip,jp)-psi(ip,jm)+2*(psi(i,jm)-psi(i,jp))
	    + psi(im,jp) - psi(im,jm));
      dpy3  = dy3*(0.5*(-psi(i,jmm)+psi(i,jpp))+psi(i,jm)-psi(i,jp));
      dpy2  = dy2*(psi(i,jm)-2*psi(i,j)+psi(i,jp));
      dpx2  = dx2*(psi(im,j)-2*psi(i,j)+psi(ip,j));
      dpy   = dy1*(-psi(i,jm)+psi(i,jp));

      stress(i,j) = 1*qo2*((dpx2y + dpy3 + qo2*dpy)*dpy
    		  - (dpx2 + dpy2 + qo2*psi(i,j))*dpy2);
 
    }
  }

  std::ofstream stress_output("stress.dat");
  assert(stress_output.is_open());

  for ( int j = 0; j < np ; j++){
    for ( int i = 0; i < np ; i++){
      stress_output << stress(i,j) << "\n ";
    }
  }

  stress_output.close();

  std::cout << "\n" << "END OF ROUTINE"  <<  std::endl;


}
