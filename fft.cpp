/*************************
*                        *
*      FFT examples      *
*   for Eigen library    *
*                        *
*************************/

#include <Eigen/Core>
#include <eigen3/unsupported/Eigen/FFT>
#include <fstream>
#include <cassert>
#include <cmath>
#include <complex>  // std::complex, std::abs
#include <iostream>

const double PI = 3.141592653589793;
const int np = 128;

Eigen::MatrixXcd fft2 (Eigen::MatrixXd in);
Eigen::MatrixXd ifft2(Eigen::MatrixXcd out);
std::pair<Eigen::MatrixXd,Eigen::MatrixXd> makeSr (Eigen::MatrixXd psi);
Eigen::MatrixXcd fft2shift (Eigen::MatrixXcd in);

/* FINITE DIFFERENCE SCHEME: CALCULATE S(r) */

  const double qo = 1;
  const double alpha = 1;
  const double dx = 0.5;
  const double dy = dx;

  const double dy4   =   1/pow(dy,4);
  const double dx4   =   1/pow(dx,4);
  const double dx2y2 = 1/(pow(dx,4));
  const double dy2   =   1/pow(dx,2);
  const double dx2   =   1/pow(dx,2);
  const double dy1   =      1/(2*dx);
  const double dx1   =      1/(2*dx);
  const double qo2   =     pow(qo,2);
  const double qo4   =     pow(qo,4);

/////////////////////////////

int main ()
{
    size_t dim_x = 128, dim_y = 128;
    Eigen::FFT<double> fft;
    Eigen::MatrixXd in(dim_x,dim_y);// = Eigen::MatrixXf::Random(dim_x, dim_y);


    /* FUNCTION */

    for (int i = 0; i < dim_x; i++){
      for(int j = 0; j < dim_y; j++){
	in(i,j) = sin(2*PI*10*i/dim_x);
	std::cout << "doc ok" << std::endl;
      }
    }


/* TEST TEST */

Eigen::MatrixXd psi = in;

std::pair<Eigen::MatrixXd,Eigen::MatrixXd> Sr = makeSr(psi);

    /* FFT  */

    Eigen::MatrixXcd out = fft2(in);
    out = fft2shift(out);  // shift values for display

   // Save FFT into fft.dat

   std::ofstream fft_output("fft.dat");
   assert(fft_output.is_open());

   for ( int j = 0; j < dim_y ; j++){
     for ( int i = 0; i < dim_x ; i++){
       fft_output << std::abs(out(i,j))/(dim_x*0.5) << "\n";
     }
   }

  fft_output.close();

  out = fft2shift(out); // disshift values for inverse

  /* Inverse FFT */
  
  Eigen::MatrixXd iOut = ifft2(out);
  
  // Save iFFt into ifft.dat

  std::ofstream ifft_output("ifft.dat");
  assert(ifft_output.is_open());

  for ( int j = 0; j < dim_y ; j++){
    for ( int i = 0; i < dim_x ; i++){
      ifft_output << iOut(i,j) << "\n";
    }
  }

  ifft_output.close();

  return 0;

}


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


std::pair<Eigen::MatrixXd,Eigen::MatrixXd> makeSr (Eigen::MatrixXd psi){
  
  Eigen::MatrixXd cSr(np,np);
  Eigen::MatrixXd SrU(np,np);
  Eigen::MatrixXd SrV(np,np);

  double   dpy4, dpx4, dpx2y2, dpy2, dpx2;
  
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

      dpx2 = dx2*(psi(ip,j)-2*psi(i,j)+psi(im,j));
      dpy2 = dy2*(psi(i,jp)-2*psi(i,j)+psi(i,jm));

      dpx4 = dx4*(psi(ipp,j)+psi(imm,j)-4*(psi(ip,j)+psi(im,j))+6*psi(i,j));
      dpy4 = dy4*(psi(i,jpp)+psi(i,jmm)-4*(psi(i,jp)+psi(i,jm))+6*psi(i,j));

      dpx2y2 = dx2y2*(psi(ip,jp)+psi(ip,jm)+psi(im,jp)+psi(im,jm)
	       -2*(psi(ip,j)+psi(im,j)+psi(i,jp)+psi(i,jm)-2*psi(i,j)));

      cSr(i,j) = alpha*(qo4*psi(i,j)+2*qo2*(dpx2+dpy2)+(dpx4+2*dpx2y2+dpy4));
      SrU(i,j) = dx1*cSr(i,j)*(psi(ip,j)-psi(im,j));
      SrV(i,j) = dy1*cSr(i,j)*(psi(i,jp)-psi(i,jm));
    }
  }

  std::pair <Eigen::MatrixXd,Eigen::MatrixXd> makeSr;
  makeSr = std::make_pair (SrU,SrV);

  return makeSr;
}

Eigen::MatrixXcd fft2shift (Eigen::MatrixXcd in){
    Eigen::MatrixXcd out (np,np);

    for (int i = 0; i < (np/2); i++) {
      for (int j = 0; j < (np/2); j++) {
	out(i,j) = in(np/2+i,np/2+j);
	out(np/2+i,j) = in(i,np/2+j);
	out(i,np/2+j) = in(np/2+i,j);
	out(np/2+i,np/2+j) = in(i,j);
      }
    }

  return out;
}

  
