#include <cmath>
#include <complex>
#include "ft.hpp"

void ft_cubic(std::complex<double>* h, int32_t M, double a, double delta, double* R, int32_t N, std::complex<double> *I){
    for(int n=0;n<N;n++){
        I[n] = ft_cubic(h, M, a, delta, R[n]);
    }
}
        
std::complex<double> ft_cubic(std::complex<double>* h, int32_t M, double a, double delta, double r){
    using namespace std;
    using namespace std::complex_literals;
    double W;
    std::complex<double> a0,a1,a2,a3;
    
    //s = variable to be integrated (t for the Fourier transform)
    //r = external variable (omega for the Fourier transform)
    
    double th = r*delta;
    double th2 = pow(th,2);
    double th3 = pow(th,3);
    double th4 = pow(th,4);
    
    double b = a + (M-1)*delta;
    
    // If th is small, use the expansions for W and a_j, so that you don't
    // run in numerical errors because of the limits.
    if(abs(th)<5e-2){
        double th6 = pow(th,6);
        double th5 = pow(th,5);
        double th7 = pow(th,7);
        
        W = 1.0 - 11.0/720.0*th4 + 23.0/15120.0*th6;
        
        a0 = -2.0/3.0 + 1.0/45.0*th2 + 103.0/15120.*th4 -169.0/226800.0*th6 +
             1i*(2.0/45.0*th + 2.0/105.0*th3 - 8.0/2835.0*th5 + 86.0/467775.*th7);
        
        a1 = 7.0/24.0 - 7.0/180.0*th2 + 5.0/3456.0*th4 - 7.0/259200.0*th6 +
             1i*(7.0/72.0*th - 1.0/168.0*th3 + 11.0/72576.0*th5 - 13.0/5987520.*th7);
        
        a2 = -1.0/6.0 + 1.0/45.0*th2 - 5.0/6048.0*th4 + 1.0/64800.0*th6 +
             1i*(-7.0/90.0*th +1.0/210.0*th3 - 11.0/90720.*th5 + 13.0/7484400.*th7);
        
        a3 = 1.0/24.0 - 1.0/180.0*th2 + 5.0/24192.0*th4 - 1.0/259200.0*th6 +
             1i*(7.0/360.0*th - 1.0/840.0*th3 + 11.0/362880.*th5 - 13.0/29937600.*th7);
    } else {
        W = (6.0+th2) / (3.0*th4) * (3.0-4.0*cos(th)+cos(2.0*th));
        
        a0 = (-42.0 + 5.0*th2 + (6.0+th2) * (8.0*cos(th)-cos(2.0*th))) / (6.0*th4) +
             1i * (-12.0*th + 6.0*th3 + (6.0+th2)*sin(2.0*th)) / (6.0*th4);

        a1 = (14.0*(3.0-th2) - 7.0*(6.0+th2)*cos(th)) / (6.0*th4) +
             1i*(30.0*th - 5.0*(6.0+th2)*sin(th)) / (6.0*th4);

        a2 = (-4.0*(3.0-th2) + 2.0*(6.0+th2)*cos(th)) / (3.0*th4) +
             1i*(-12.0*th + 2.0*(6.0+th2)*sin(th)) / (3.0*th4);

        a3 = (2.0*(3.0-th2) - (6.0+th2)*cos(th)) / (6.0*th4) +
             1i*(6.0*th - (6.0+th2)*sin(th)) / (6.0*th4);
    }
    
    std::complex<double> I=0.0+0.0*1i, arg=0.0+0.0*1i;
    for(int j=0;j<M;j++){
        arg = 0.0+1i*j*th;
        I += h[j]*exp(arg);
    }
    
    I *= W;
    
    I += h[0]*a0 + h[1]*a1 + h[2]*a2 + h[3]*a3;
    arg = 1i*r*(b-a);
    I += exp(arg)*(h[M-1]*conj(a0) + h[M-2]*conj(a1) + h[M-3]*conj(a2) + h[M-4]*conj(a3));
    
    arg = 1i*r*a;
    I *= delta*exp(arg);
    
    I /= sqrt(2.*M_PI);
    
    return I;
}
