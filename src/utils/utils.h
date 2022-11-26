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
#include <float.h>
#include "../simulation-utils/simulation_config.h"

#define EMPTY_SET "#"
#define MAKE_STRING( msg )  ( ((std::ostringstream&)((std::ostringstream() << '\x0') << msg)).str().substr(1) )


double invErf(double f) {
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

void saveOnFile(configuration_description* config,std::map<double, double> results,char* outputfilename){
    double curr_alives = config->num_of_tests;
    double prec_time = 0;

    double mttf_int1 = 0;
    if (outputfilename) {
        std::ofstream outfile(outputfilename);
        if (results.count(0) == 0) {
            results[0] = 0;
        }
        //TODO understand if it is important
        for (std::map<double, double>::iterator mapIt = results.begin(); mapIt != results.end(); mapIt++) {
            curr_alives = curr_alives - mapIt->second;
            mttf_int1 = mttf_int1 + curr_alives / config->num_of_tests * (mapIt->first - prec_time);
            prec_time = mapIt->first;
            outfile << mapIt->first << " " << (curr_alives / config->num_of_tests) << std::endl;
        }
        outfile.close();
    }
}

#endif //UTILS