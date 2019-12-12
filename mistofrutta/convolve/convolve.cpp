#include <cmath>
#include <iostream>
#include <stdint.h>
#include "convolve.hpp"

double _convolve1(double *A, double *B, int N, double delta) {
    double y = 0.0;
    for(int j=0;j<=N;j++){y += A[N-j]*B[j];}
    //Trapezoidal-rule correction
    if(N>2){y-=0.5*(A[N]*B[0]+A[0]*B[N]);}else{y *= 0.5;}
    y *= delta;
    return y;
}

void convolve(double *A, double *B, int M, double delta, double *out) {
    for(int m=0;m<M;m++){
        if(m==0){
            out[m] = 0.0;
        }else{
            out[m] = _convolve1(A,B,m,delta);
        }
        /**for(int j=0;j<=m;j++){
            out[m] += A[m-j]*B[j];
        }
        out[m] *= delta;**/
    }
}


void nonlinconv(double *A, double *B, int M, double coeff, double dt){
    double *C;
    C = new double[M];
    double *D;
    D = new double[M];
    for(int m=0;m<M;m++){
        for(int j=0;j<m;j++){
            C[j] = B[m-j]*A[j];
        }
        convolve(A,C,m,dt,D);
        B[m] += coeff*_convolve1(D,B,m,dt);
    }
}
