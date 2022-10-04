#ifndef UTILS
#define UTILS
#include <iostream>
#include <string>
#include <cmath>
#include <iomanip>
#include <map>
#include <cstdlib>
#include <vector>
#include <fstream>
#include <cstring>
#include <sstream>
#include <getopt.h>
#include <set>
#include <unistd.h>
#include "cuda_helper.h"


#define EMPTY_SET "#"
#define MAKE_STRING( msg )  ( ((std::ostringstream&)((std::ostringstream() << '\x0') << msg)).str().substr(1) )
//__device__ __host__ 
//ROUND TO 6 digit
__device__ __host__  double round1(double n){
	int k = 0;
	while(n<100000){
		k++;
		n *= 10.0;
	}
	n = round(n);
	n /= pow(10, k);
	return n;
}

__device__ __host__  double invErf(double f) {
// inverts the gaussian distribution N(0,1) using the bisection method
    double l = 0;
    double r = 1;
    double vl, vr, h, vh;

    // first iteration: find a v, such that N(v) > f
    while ((erf(r / M_SQRT2) + 1.0) / 2.0 < f) {
        r *= 2;
    }

    // solves the equation iteratively
    vl = (erf(l / M_SQRT2) + 1.0) / 2.0 - f;
    vr = (erf(r / M_SQRT2) + 1.0) / 2.0 - f;
    h = (l + r) / 2;
    while (fabs(r - l) / h > INV_ERF_ACCURACY) {
        vh = (erf(h / M_SQRT2) + 1.0) / 2.0 - f;
        if (vh * vl < 0.0) {
            r = h;
            vr = vh;
        } else {
            l = h;
            vl = vh;
        }
        h = (l + r) / 2;
    }

    return h;
}

#endif //UTILS