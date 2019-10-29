#include <cmath>
#include <iostream>
#include "convolve.hpp"

void convolve(double *A, double *B, int M, double delta, double *out) {
    for(int m=0;m<M;m++){
        out[m] = 0.0;
        for(int j=0;j<=m;j++){
            out[m] += A[m-j]*B[j];
        }
        //if(m>6){
        //    out[m] -= (5./8.*(A[m]*B[0]) + 1./6.*(A[m-1]*B[1]) + 1./24.*(A[m-2]*B[2]));
        //}
        out[m] *= delta;
    }
}
