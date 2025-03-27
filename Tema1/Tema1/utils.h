#pragma once
#ifndef UTILS_H
#define UTILS_H

#include <vector>
#include <string>

template <typename T>
std::string to_string(const T& value);

std::string vector_to_string(const std::vector<double>& values);

void rand_x(chromosome *c, int num_dims, int num_bits_per_dimension);
double binary_to_real(char* b_string, unsigned long long num_bits_per_dimension, double min_x, double max_x);
double calculate_standard_deviation(chromosome* population, int pop_size, int function_option);
double calculate_mean(chromosome* population, int pop_size);
int calculate_num_bits_per_dimension(int function_option, double precision);
#endif 

