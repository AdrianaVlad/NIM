#include "chromosome.h"
#include "FitnessFunctions.h"
#include <sstream>
#include <iomanip>
#include <vector>
#include <string>
#include <cstdlib>
#include <cstring> 

const int PRECISION = 5; 


template <typename T>
std::string to_string(const T& value) {
    std::ostringstream oss;
    oss << value;
    return oss.str();
}


std::string vector_to_string(const std::vector<double>& values) {
    std::ostringstream oss;
    for (size_t i = 0; i < values.size(); ++i) {
        oss << std::fixed << std::setprecision(PRECISION) << values[i];
        if (i < values.size() - 1) {
            oss << "; ";
        }
    }
    return oss.str();
}

void rand_x(chromosome* c, int num_dims, int num_bits_per_dimension)
{
    int _length = num_dims * num_bits_per_dimension;
    for (int i = 0; i < _length; i++)
        c->x[i] = rand() % 2;
}

double binary_to_real(char* b_string, unsigned long long num_bits_per_dimension, double min_x, double max_x)
{
	double x_real = 0;
	for (int j = 0; j < num_bits_per_dimension; j++)
		x_real = x_real * 2 + (int)b_string[j];
	x_real /= ((1ULL << num_bits_per_dimension) - 1);
	x_real *= (max_x - min_x);
	x_real += min_x;
	x_real = round(x_real * 1e5) / 1e5;

	return x_real;
}

double calculate_standard_deviation(chromosome* population, int pop_size, int function_option) {
	double sum = 0.0;
	if (function_option == 4)
		sum = pop_size * 200;
	for (int i = 0; i < pop_size; i++) {
		sum += population[i].f;
	}

	double mean = sum / pop_size;
	sum = 0.0;
	if (function_option == 4)
		for (int i = 0; i < pop_size; i++) {
			sum += double(std::pow(population[i].f + 200 - mean, 2) / pop_size);
		}
	else
		for (int i = 0; i < pop_size; i++) {
			sum += double(std::pow(population[i].f - mean, 2) / pop_size);
		}
	double dev = std::sqrt(sum);
	dev = std::round(dev * 1e5) / 1e5;
	return dev;
}


double calculate_mean(chromosome* population, int pop_size) {
	double sum_fitness = 0.0;

	for (int i = 0; i < pop_size; i++) {
		sum_fitness += population[i].f;
	}
	double mean = sum_fitness / pop_size;
	return mean;
}

int calculate_num_bits_per_dimension(int function_option, double precision) {
    double min_x, max_x;
    get_minmax(function_option, &min_x, &max_x);
    return static_cast<int>(ceil(log2((max_x - min_x) * pow(10, precision))));
}

