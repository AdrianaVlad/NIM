#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <ctime>
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <iostream>
#include <fstream>
#include <sstream> 
#include <iomanip>
#include <cmath> 
#include <csignal>
#include <cstdlib>
#include <functional>
#include <thread>
#include "chromosome.h"
#include "FitnessFunctions.h"
#include "utils.h"
#include "./ThreadPool.h"
#define M_PI 3.14159265358979323846 
#define PRECISION 5

std::vector<double> worst_values;
double worst_fitness, worst_f;
std::vector<double> best_values;
std::vector<double> fitness_evolution;
double best_fitness, best_f;
double stddev;
double mean_global;
double final_pm;
int pop_size;
int num_gens;
int temporary_num_gens;
double pcross;
int num_dims;
clock_t start_time;
clock_t end_time;
double time_measurement;
int function_option;
int hc_option;
int tournament_size;
double pm;
int num_bits_per_dimension;


void adjust_mutation_rate(double& pm, double incline, double inc) {
	if (incline <= 0.0)
		pm = std::min(pm + 5 * inc, 0.01);
	else
		pm = std::max(pm - 5 * inc, 0.00001);
	pm = std::max(pm - 0.00005, 0.00001);
}

void copy_chromosome(chromosome* dest, chromosome source, int num_dims, int num_bits_per_dimension)
{
	int _length = num_dims * num_bits_per_dimension;
	for (int i = 0; i < _length; i++)
		dest->x[i] = source.x[i];
	dest->fitness = source.fitness;
	dest->f = source.f;
}

void compute_fitness(chromosome* c, int num_dims, int num_bits_per_dimension, int function_option) {
	switch (function_option) {
	case 1: FitnessFunctions::rastrigin(c, num_dims, num_bits_per_dimension); break;
	case 2: FitnessFunctions::griewangk(c, num_dims, num_bits_per_dimension); break;
	case 3: FitnessFunctions::rosenbrock(c, num_dims, num_bits_per_dimension); break;
	case 4: FitnessFunctions::michalewicz(c, num_dims, num_bits_per_dimension); break;
	}
}

void mutation(chromosome* c, int num_dims, int num_bits_per_dimension, double pm, std::mt19937& gen)
{
	int _length = num_dims * num_bits_per_dimension; 
	std::uniform_real_distribution<> dis(0.0, 1.0);
	double p;
	for (int i = 0; i < _length; i++) {
		p = dis(gen);
		if (p < pm)
			c->x[i] = 1 - c->x[i];
	}
}
void three_cut_point_crossover(chromosome parent1, chromosome parent2, chromosome* offspring1, chromosome* offspring2, int num_dims, int num_bits_per_dimension, std::mt19937& gen)
{
	std::uniform_int_distribution<> dis(1, num_dims * num_bits_per_dimension - 1); 
	int pct1 = dis(gen);
	int pct2 = dis(gen);
	int pct3 = dis(gen);
	if (pct1 > pct2) {
		std::swap(pct1, pct2);
	}
	if (pct2 > pct3) {
		std::swap(pct2, pct3);
	}
	if (pct1 > pct3) {
		std::swap(pct1, pct3);
	}
	for (int i = 0; i < pct1; i++) {
		offspring1->x[i] = parent1.x[i];
		offspring2->x[i] = parent2.x[i];
	}
	for (int i = pct1; i < pct2; i++) {
		offspring1->x[i] = parent2.x[i];
		offspring2->x[i] = parent1.x[i];
	}
	for (int i = pct2; i < pct3; i++) {
		offspring1->x[i] = parent1.x[i];
		offspring2->x[i] = parent2.x[i];
	}
	for (int i = pct3; i < num_dims * num_bits_per_dimension; i++) {
		offspring1->x[i] = parent2.x[i];
		offspring2->x[i] = parent1.x[i];
	}
}
int sort_function(const void* a, const void* b)
{
	if (((chromosome*)a)->fitness > ((chromosome*)b)->fitness)
		return -1;
	else
		if (((chromosome*)a)->fitness < ((chromosome*)b)->fitness)
			return 1;
		else
			return 0;
}

void print_chromosome(chromosome* population, int pop_size, int num_dims, int num_bits_per_dimension, int function_option, double std_dev, double mean, double pm)
{
	double min_x, max_x;
	get_minmax(function_option, &min_x, &max_x);

	chromosome* best_chromosome = &population[0];
	chromosome* worst_chromosome = &population[pop_size - 1];

	printf("Best: x = (");
	best_values.clear();
	for (int i = 0; i < num_dims; i++) {
		double x_real = binary_to_real(best_chromosome->x + i * num_bits_per_dimension, num_bits_per_dimension, min_x, max_x);
		best_values.push_back(x_real);
		printf("%lf ", x_real);
	}
	printf(") ");
	printf("fitness = %lf ", best_chromosome->fitness);
	printf("f(x) = %lf", best_chromosome->f);
	best_f = best_chromosome->f;
	best_fitness = best_chromosome->fitness;
	fitness_evolution.push_back(best_fitness);

	printf("\n");

	printf("Worst: x = (");
	worst_values.clear();
	for (int i = 0; i < num_dims; i++) {
		double x_real = binary_to_real(worst_chromosome->x + i * num_bits_per_dimension, num_bits_per_dimension, min_x, max_x);
		worst_values.push_back(x_real);
		printf("%lf ", x_real);
	}
	printf(") ");
	printf("fitness = %lf ", worst_chromosome->fitness);
	printf("f(x) = %lf", worst_chromosome->f);
	worst_f = worst_chromosome->f;
	worst_fitness = worst_chromosome->fitness;
	printf("\n");

	stddev = std_dev;
	final_pm = pm;

	printf("Standard deviation = %f, Mean = %f, Mutation probability = %f\n\n", std_dev, mean, pm);
}
void tournament_selection(int* k1, int* k2, int tournament_size, int pop_size)
{
	int i;
	*k1 = pop_size;
	*k2 = pop_size;
	for (int j = 0; j < tournament_size; j++) {
		i = rand() % pop_size;
		if (i < *k1) {
			*k2 = *k1;
			*k1 = i;
		}
		else if (i < *k2)
			*k2 = i;
	}
}

void hill_climbing(chromosome* dest, int num_dims, int num_bits_per_dimension, int function_option, int steps)
{
	int t = 0, current_modified_index = -1;
	chromosome neighbour;
	neighbour.x = (char*)malloc(num_dims * num_bits_per_dimension);
	bool local = false;
	int _length = num_dims * num_bits_per_dimension;
	copy_chromosome(&neighbour, *dest, num_dims, num_bits_per_dimension);
	while (!local && steps) {
		steps--;
		local = true;
		for (int i = 0; i < _length; i++) {
			neighbour.x[i] = 1 - neighbour.x[i];
			compute_fitness(&neighbour, num_dims, num_bits_per_dimension, function_option);
			if (neighbour.fitness > dest->fitness) {
				current_modified_index = i;
				dest->x[i] = 1 - dest->x[i];
				dest->fitness = neighbour.fitness;
				dest->f = neighbour.f;
				local = false;
			}
			neighbour.x[i] = 1 - neighbour.x[i];
		}
		if (!local) {
			neighbour.x[current_modified_index] = 1 - neighbour.x[current_modified_index];
			neighbour.fitness = dest->fitness;
			neighbour.f = dest->f;
		}
	}
	free(neighbour.x);
	printf(".");
}

void wheel_of_fortune_selection(chromosome* population, chromosome* new_population, int pop_size, int num_dims, int num_bits_per_dimension) {
	double* eval = (double*)malloc(pop_size * sizeof(double));
	double total_fitness = 0.0;
	double* selection_prob = (double*)malloc(pop_size * sizeof(double));
	double* accumulated_prob = (double*)malloc((pop_size + 1) * sizeof(double));

	for (int i = pop_size / 20; i < pop_size; i++) {
		eval[i] = population[i].fitness;
		total_fitness += eval[i];
	}

	for (int i = pop_size / 20; i < pop_size; i++) {
		selection_prob[i] = eval[i] / total_fitness;
	}

	accumulated_prob[0] = 0.0;
	for (int i = pop_size / 20; i < pop_size; i++) {
		accumulated_prob[i + 1] = accumulated_prob[i] + selection_prob[i];
	}

	for (int i = pop_size / 20; i < pop_size; i++) {
		double r = (double)rand() / RAND_MAX;

		for (int j = pop_size / 20; j < pop_size; j++) {
			if (accumulated_prob[j] < r && r <= accumulated_prob[j + 1]) {
				copy_chromosome(&new_population[i], population[j], num_dims, num_bits_per_dimension);
				break;
			}
		}
	}
}

void process_population_chunk(chromosome& new_chromo1, chromosome& new_chromo2, chromosome& temp_chromo1, chromosome& temp_chromo2, double pcross, double pm, int function_option, int num_dims, int num_bits_per_dimension, std::mt19937& gen) {
	std::uniform_real_distribution<> dis(0.0, 1.0);
	double p = dis(gen);
	copy_chromosome(&temp_chromo1, new_chromo1, num_dims, num_bits_per_dimension);
	copy_chromosome(&temp_chromo2, new_chromo2, num_dims, num_bits_per_dimension);
	if (p < pcross)
		three_cut_point_crossover(temp_chromo1, temp_chromo2, &new_chromo1, &new_chromo2, num_dims, num_bits_per_dimension, gen);
	mutation(&new_chromo1, num_dims, num_bits_per_dimension, pm, gen);
	compute_fitness(&new_chromo1, num_dims, num_bits_per_dimension, function_option);
	mutation(&new_chromo2, num_dims, num_bits_per_dimension, pm, gen);
	compute_fitness(&new_chromo2, num_dims, num_bits_per_dimension, function_option);

}


void log_to_csv(int function_option, int hc_option, int pop_size, int num_gens,
	double pcross, double pm, int num_dims, int tournament_size,
	int num_bits_per_dimension, double time_measurement,
	double best_fitness, const std::vector<double>& best_values,
	double worst_fitness, const std::vector<double>& worst_values,
	const std::vector<double>& fitness_evolution,
	double stddev, double mean_global, double best_f, double worst_f) {
	end_time = clock();
	time_measurement = static_cast<double>(end_time - start_time) / CLOCKS_PER_SEC;

	std::ostringstream filename;
	filename << "results_function_" << function_option
		<< "_hc_" << hc_option
		<< "_num_dims_" << num_dims << ".csv";
	std::string filename_str = filename.str();

	bool file_exists = false;
	std::ifstream file_check(filename_str);
	if (file_check.good()) {
		file_exists = true;
	}
	file_check.close();


	std::ofstream file(filename_str, std::ios::app);
	if (file.is_open()) {
		if (!file_exists) {
			file << "function_option,hc_option,pop_size,num_gens,pcross,pm,num_dims,"
				<< "tournament_size,num_bits_per_dimension,time_measurement,"
				<< "best_fitness,best_f,best_x,worst_fitness,worst_f,worst_x,fitness_evolution,std_deviation,mean\n";
		}
		file << std::fixed << std::setprecision(8);
		file << function_option << "," << hc_option << "," << pop_size << "," << temporary_num_gens << ","
			<< pcross << "," << pm << "," << num_dims << "," << tournament_size << ","
			<< num_bits_per_dimension << "," << time_measurement << ","
			<< best_fitness << "," << best_f << "," << "[" << vector_to_string(best_values) << "],"
			<< worst_fitness << "," << worst_f << "," << "[" << vector_to_string(worst_values) << "],"
			<< "[" << vector_to_string(fitness_evolution) << "]," << stddev << ", " << mean_global << "\n";
		file.flush();
		file.close();
		std::cout << "Results and hyperparameters saved to '" << filename_str << "'.\n";
	}
	else {
		std::cerr << "Unable to open file for writing.\n";
	}
}

void signalHandler(int signum) {
	std::cout << "Interrupt signal (" << signum << ") received.\n";
	log_to_csv(function_option, hc_option, pop_size, num_gens, pcross, pm, num_dims, tournament_size, num_bits_per_dimension, time_measurement, best_fitness, best_values, worst_fitness, worst_values, fitness_evolution, stddev, mean_global, best_f, worst_f);
	exit(signum);
}

void genetic_alg(int pop_size, int num_gens, int num_dims, int num_bits_per_dimension, double pcross, double pm, int function_option, int hc_option, int tournament_size)
{
	chromosome* population = (chromosome*)malloc(pop_size * sizeof(chromosome));
	chromosome* new_population = (chromosome*)malloc(pop_size * sizeof(chromosome));
	chromosome* temp_population = (chromosome*)malloc(pop_size * sizeof(chromosome));
	chromosome offspring1, offspring2;
	offspring1.x = (char*)malloc(num_dims * num_bits_per_dimension);
	offspring2.x = (char*)malloc(num_dims * num_bits_per_dimension);
	int gens_to_pm_adjust = num_gens / 50;
	double incline = 0.0;
	double inc = 0.00005;
	if (hc_option < 4)
		inc *= 2;
	for (int i = 0; i < pop_size; i++) {
		population[i].x = (char*)malloc(num_dims * num_bits_per_dimension);
		new_population[i].x = (char*)malloc(num_dims * num_bits_per_dimension);
		temp_population[i].x = (char*)malloc(num_dims * num_bits_per_dimension);
	}
	int num_threads = std::thread::hardware_concurrency();

	for (int i = 0; i < pop_size; i++) {
		rand_x(&population[i], num_dims, num_bits_per_dimension);
		compute_fitness(&population[i], num_dims, num_bits_per_dimension, function_option);
	}
	qsort(population, pop_size, sizeof(population[0]), sort_function);
	printf("generation 0\n");
	double std_dev_prev = calculate_standard_deviation(population, pop_size, function_option);
	double mean_prev = calculate_mean(population, pop_size);
	print_chromosome(population, pop_size, num_dims, num_bits_per_dimension, function_option, std_dev_prev, mean_prev, pm);
	std::vector<int> indices(pop_size);
	std::vector<std::mt19937> generators(pop_size / 2);
	for (int i = 0; i < pop_size / 2; i++) {
		std::random_device rd;
		std::mt19937 gen(rd());
		generators[i] = gen;
	}
	for (int i = 0; i < pop_size; i++) indices[i] = i;

	for (int g = 1; g <= num_gens; g++) {
		temporary_num_gens = g;
		if (hc_option / 2 == 1) {
			for (int i = 0; i < pop_size; i++)
				hill_climbing(&population[i], num_dims, num_bits_per_dimension, function_option,5+g/50);
		}
		for (int k = 0; k < pop_size / 20; k++) {
			copy_chromosome(&new_population[k], population[k], num_dims, num_bits_per_dimension);
		}
		for (int k = pop_size / 20; k < pop_size; k++) {
			int k1, k2;
			tournament_selection(&k1, &k2, tournament_size, pop_size);
			copy_chromosome(&new_population[k], population[k1], num_dims, num_bits_per_dimension);
		}
		std::random_shuffle(indices.begin(), indices.end());
		ThreadPool pool2(num_threads);
		for (int t = 0; t < pop_size; t += 2) {
			
			int k1 = indices[t];
			int k2 = indices[t + 1];
			pool2.enqueueTask([&, k1, k2]() {
				process_population_chunk(new_population[k1], new_population[k2], temp_population[k1], temp_population[k2], pcross, pm, function_option, num_dims, num_bits_per_dimension, generators[k1/2]);
				});
		}
		pool2.~ThreadPool();
		qsort((void*)new_population, pop_size, sizeof(new_population[0]), sort_function);
		int skipped = 0;
		std::vector<int> replaced;
		for (int k = 0; k < pop_size / 20; k++) {
			if (new_population[k - skipped].fitness > population[k].fitness) {
				copy_chromosome(&population[k], new_population[k - skipped], num_dims, num_bits_per_dimension);
				replaced.push_back(k);
			}
			else skipped++;
		}

		for (int k = pop_size / 20; k < pop_size; k++) {
			copy_chromosome(&population[k], new_population[k - skipped], num_dims, num_bits_per_dimension);
		}
		double std_dev = calculate_standard_deviation(population, pop_size, function_option);
		double mean = calculate_mean(population, pop_size);
		incline += (double)(std_dev - std_dev_prev);
		if (g % 10 == 0) {
			adjust_mutation_rate(pm, incline, inc);
			if (hc_option % 2 == 1 && incline > 0) {
				ThreadPool pool(num_threads);
				for (auto k = 0; k < pop_size / 20; k++)
					pool.enqueueTask([&, k]() {
					hill_climbing(&population[k], num_dims, num_bits_per_dimension, function_option, 1+g/50);
						});
				pool.~ThreadPool();
				std::random_shuffle(indices.begin(), indices.end());
				ThreadPool pool1(num_threads);
				for (auto k = 0; k < pop_size / 20; k++) {
					int k1 = indices[k];
					pool1.enqueueTask([&, k1]() {
						hill_climbing(&population[k1], num_dims, num_bits_per_dimension, function_option, 1+g/50);
						});
				}
				pool1.~ThreadPool();
				
				qsort((void*)population, pop_size, sizeof(population[0]), sort_function);
				adjust_mutation_rate(pm, incline, inc/5);
			}
			incline = 0.0;
		}
		if (std_dev <= 0.00001) {
			adjust_mutation_rate(pm, -1, inc * 100);
		}

		std_dev_prev = std_dev;
		mean_global = mean;
		printf("generation %d\n", g);
		print_chromosome(population, pop_size, num_dims, num_bits_per_dimension, function_option, std_dev, mean, pm);

	}

	for (int i = 0; i < pop_size; i++) {
		free(population[i].x);
		free(new_population[i].x);
		free(temp_population[i].x);
	}

	free(population);
	free(new_population);
	free(temp_population);
	log_to_csv(function_option, hc_option, pop_size, num_gens, pcross, pm, num_dims, tournament_size, num_bits_per_dimension, time_measurement, best_fitness, best_values, worst_fitness, worst_values, fitness_evolution, stddev, mean_global, best_f, worst_f);
}
void testing() {
	for (int i = 0; i < 30; i++) {
		start_time = clock();
		genetic_alg(pop_size, num_gens, num_dims, num_bits_per_dimension, pcross, pm, function_option, hc_option, tournament_size);
		fitness_evolution.clear();
	}
}

int main(void)
{
	signal(SIGINT, signalHandler);
	pcross = 0.8;
	num_dims = 100;
	std::cout << "Enter number of dimensions(2,30,100,others): ";
	std::cin >> num_dims;
	function_option = 3;
	hc_option = 1;
	tournament_size = 5;
	std::cout << "Enter function option (1-4): ";
	std::cin >> function_option;
	std::cout << "Enter hill climbing option (1: hc children, 2: hc all at start of generation, 3: both, 4: neither): ";
	std::cin >> hc_option;
	if (hc_option < 4) {
		pop_size = 150 * num_dims;
		num_gens = 500;
		pm = (double)30.0 / pop_size;
	}
	else {
		pop_size = 150 * num_dims;
		num_gens = 500;
		pm = (double)30.0 / pop_size;
	}
	num_bits_per_dimension = calculate_num_bits_per_dimension(function_option, PRECISION);
	printf("Number of bits per dimension %d \n", num_bits_per_dimension);
	srand(time(0));

	//testing();
	start_time = clock();
	genetic_alg(pop_size, num_gens, num_dims, num_bits_per_dimension, pcross, pm, function_option, hc_option, tournament_size);
	fitness_evolution.clear();

	return 0;
}