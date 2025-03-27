#pragma once
#ifndef FITNESS_FUNCTIONS_H
#define FITNESS_FUNCTIONS_H
#include "chromosome.h"

namespace FitnessFunctions {
    void rastrigin(chromosome* c, int num_dims, int num_bits_per_dimension);
    void griewangk(chromosome* c, int num_dims, int num_bits_per_dimension);
    void rosenbrock(chromosome* c, int num_dims, int num_bits_per_dimension);
    void michalewicz(chromosome* c, int num_dims, int num_bits_per_dimension);
}
void get_minmax(int function_option, double* min_x, double* max_x);
#endif
