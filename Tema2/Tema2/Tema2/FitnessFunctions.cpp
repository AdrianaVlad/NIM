#include "FitnessFunctions.h"
#include "utils.h"
#include <cmath>
#include <cstdlib>

#define M_PI 3.14159265358979323846 



void FitnessFunctions::rastrigin(chromosome* c, int num_dims, int num_bits_per_dimension) {
    double* x_real = (double*)malloc(num_dims * sizeof(double));
    for (int i = 0; i < num_dims; i++)
        x_real[i] = binary_to_real(c->x + i * num_bits_per_dimension, num_bits_per_dimension, -5.12, 5.12);

    c->fitness = 10 * num_dims;
    for (int i = 0; i < num_dims; i++)
        c->fitness += x_real[i] * x_real[i] - 10 * cos(2 * M_PI * x_real[i]);
    c->f = c->fitness;
    c->fitness = 1 / c->fitness;
    free(x_real);
}

void FitnessFunctions::griewangk(chromosome* c, int num_dims, int num_bits_per_dimension) {
    double* x_real = (double*)malloc(num_dims * sizeof(double));
    for (int i = 0; i < num_dims; i++)
        x_real[i] = binary_to_real(c->x + i * num_bits_per_dimension, num_bits_per_dimension, -600, 600);

    c->fitness = 1;
    double sum_squares = 0.0;
    double product_cos = 1.0;

    for (int i = 0; i < num_dims; i++) {
        sum_squares += x_real[i] * x_real[i];
        product_cos *= cos(x_real[i] / sqrt(i + 1));
    }
    c->fitness += (sum_squares / 4000.0) - product_cos;
    c->f = c->fitness;
    c->fitness = 1 / c->fitness;
    free(x_real);
}

void FitnessFunctions::rosenbrock(chromosome* c, int num_dims, int num_bits_per_dimension) {
    double* x_real = (double*)malloc(num_dims * sizeof(double));
    for (int i = 0; i < num_dims; i++)
        x_real[i] = binary_to_real(c->x + i * num_bits_per_dimension, num_bits_per_dimension, -2.048, 2.048);

    c->fitness = 0.0;
    for (int i = 0; i < num_dims - 1; i++) {
        c->fitness += 100 * pow(x_real[i + 1] - x_real[i] * x_real[i], 2) + pow(1 - x_real[i], 2);
    }
    c->f = c->fitness;
    c->fitness = 1 / c->fitness;
    free(x_real);
}

void FitnessFunctions::michalewicz(chromosome* c, int num_dims, int num_bits_per_dimension) {
    double* x_real = (double*)malloc(num_dims * sizeof(double));
    for (int i = 0; i < num_dims; i++)
        x_real[i] = binary_to_real(c->x + i * num_bits_per_dimension, num_bits_per_dimension, 0, M_PI);

    c->fitness = 0.0;
    for (int i = 0; i < num_dims; i++) {
        c->fitness -= sin(x_real[i]) * pow(sin((i + 1) * x_real[i] * x_real[i] / M_PI), 20);
    }
    c->f = c->fitness;
    c->fitness += 200;
    c->fitness = 1 / c->fitness;
    free(x_real);
}

void get_minmax(int function_option, double* min_x, double* max_x) {
    switch (function_option) {
    case 1:
        *min_x = -5.12;
        *max_x = 5.12;
        break;
    case 2:
        *min_x = -600;
        *max_x = 600;
        break;
    case 3:
        *min_x = -2.048;
        *max_x = 2.048;
        break;
    case 4:
        *min_x = 0;
        *max_x = M_PI;
        break;
    }
}