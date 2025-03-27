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
std::vector<double> f_evolution;
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
int num_particles, max_iterations, temp_iterations;


void adjust_mutation_rate(double& pm, double incline, double inc) {
	if (incline <= 0.0)
		pm = std::min(pm + 5 * inc, 0.1);
	else
		pm = std::max(pm - 5 * inc, 0.0);
	pm = std::max(pm - 0.00004, 0.0);
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

void process_population_chunk(chromosome& new_chromo1, chromosome& new_chromo2, chromosome& p_best1, chromosome p_best2, chromosome& temp_chromo1, chromosome& temp_chromo2, double pcross, double pm, int function_option, int num_dims, int num_bits_per_dimension, std::mt19937& gen) {
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
	if (p_best1.fitness < new_chromo1.fitness)
		copy_chromosome(&p_best1, new_chromo1, num_dims, num_bits_per_dimension);
	if (p_best2.fitness < new_chromo2.fitness)
		copy_chromosome(&p_best2, new_chromo2, num_dims, num_bits_per_dimension);
}


void log_to_csv(int function_option, int num_particles, int num_iterations, int num_dims,
	int num_bits_per_dimension, double time_measurement,
	double best_fitness, const std::vector<double>& best_values,
	const std::vector<double>& f_evolution,
	double best_f) {
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

	size_t mid_point = f_evolution.size() / 2;

	// Convert the first half to a string
	std::string first_half = vector_to_string(
		std::vector<double>(f_evolution.begin(), f_evolution.begin() + mid_point));

	// Convert the second half to a string
	std::string second_half = vector_to_string(
		std::vector<double>(f_evolution.begin() + mid_point, f_evolution.end()));


	std::ofstream file(filename_str, std::ios::app);
	if (file.is_open()) {
		if (!file_exists) {
			file << "function_option,num_particles,num_iterations,num_dims,"
				<< "num_bits_per_dimension,time_measurement,"
				<< "best_fitness,best_f,best_x,f_evolution\n";
		}
		file << std::fixed << std::setprecision(8);
		file << function_option << "," << num_particles << "," << num_iterations << ","
			<< num_dims << "," << num_bits_per_dimension << "," << time_measurement << ","
			<< best_fitness << "," << best_f << ","
			<< "[" << vector_to_string(best_values) << "],"
			<< "\"" << first_half << "\"," // First half in one cell
			<< "\"" << second_half << "\"" // Second half in another cell
			<< "\n";
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
	log_to_csv(function_option, num_particles, temp_iterations, num_dims, num_bits_per_dimension, time_measurement, best_fitness, best_values, f_evolution, best_f);
	exit(signum);
}

void process_particle(chromosome& particle, chromosome& p_best, chromosome g_best, double* velocity, int num_dims, int num_bits_per_dimension, double w, double c1, double c2, std::mt19937& gen, double min_x, double max_x, int function_option) {
	double p_best_x_real, g_best_x_real, particle_x_real, new_position_x_real;
	double r1, r2;
	for (int j = 0; j < num_dims; ++j) { //num_dims * num_bits_per_dimensions
		std::uniform_real_distribution<> dis(0.0, 1.0);
		r1 = dis(gen);
		r2 = dis(gen);
		
		p_best_x_real = binary_to_real(p_best.x + j * num_bits_per_dimension, num_bits_per_dimension, min_x, max_x);
		g_best_x_real = binary_to_real(g_best.x + j * num_bits_per_dimension, num_bits_per_dimension, min_x, max_x);
		particle_x_real = binary_to_real(particle.x + j * num_bits_per_dimension, num_bits_per_dimension, min_x, max_x);

		velocity[j] = w * velocity[j] +
			c1 * r1 * (p_best_x_real - particle_x_real) +
			c2 * r2 * (g_best_x_real - particle_x_real);

		new_position_x_real = particle_x_real + velocity[j];
		if (new_position_x_real < min_x) new_position_x_real = min_x;
		if (new_position_x_real > max_x) new_position_x_real = max_x;
		real_to_binary(new_position_x_real, particle.x + j * num_bits_per_dimension, num_bits_per_dimension, min_x, max_x);

		//velocity[i][j] = w * velocity[i][j] +
			//c1 * r1 * (p_best[i].x[j] - particles[i].x[j]) +
			//c2 * r2 * (g_best.x[j] - particles[i].x[j]);

		//double prob = 1.0 / (1.0 + exp(-velocity[i][j]));
		//double r = (double)rand() / RAND_MAX;
		//if (r < prob)
			//particles[i].x[j] = 1 - particles[i].x[j];
	}
	compute_fitness(&particle, num_dims, num_bits_per_dimension, function_option);

	if (particle.fitness > p_best.fitness)
		copy_chromosome(&p_best, particle, num_dims, num_bits_per_dimension);

}

void particle_swarm_optimization(int num_particles, int num_dims, int num_bits_per_dimension, int function_option, int max_iterations) {
	double min_x, max_x;
	double w, c1, c2;
	int num_threads = std::thread::hardware_concurrency();
	std::vector<std::mt19937> generators(num_particles);
	for (int i = 0; i < num_particles; i++) {
		std::random_device rd;
		std::mt19937 gen(rd());
		generators[i] = gen;
	}
	get_minmax(function_option, &min_x, &max_x);

	chromosome* particles = (chromosome*)malloc(num_particles * sizeof(chromosome));
	chromosome* temp_particles = (chromosome*)malloc(num_particles * sizeof(chromosome));
	chromosome* p_best = (chromosome*)malloc(num_particles * sizeof(chromosome));
	double** velocity = (double**)malloc(num_particles * sizeof(double*));
	for (int i = 0; i < num_particles; i++) {
		particles[i].x = (char*)malloc(num_dims * num_bits_per_dimension);
		temp_particles[i].x = (char*)malloc(num_dims * num_bits_per_dimension);
		p_best[i].x = (char*)malloc(num_dims * num_bits_per_dimension);
		velocity[i] = (double*)malloc(num_dims * sizeof(double));
	}
	chromosome g_best;
	g_best.x = (char*)malloc(num_dims * num_bits_per_dimension);
	rand_x(&g_best, num_dims, num_bits_per_dimension);
	compute_fitness(&g_best, num_dims, num_bits_per_dimension, function_option);

	for (int i = 0; i < num_particles; i++) {
		rand_x(&particles[i], num_dims, num_bits_per_dimension);
		compute_fitness(&particles[i], num_dims, num_bits_per_dimension, function_option);
		copy_chromosome(&p_best[i], particles[i], num_dims, num_bits_per_dimension);
		if (p_best[i].fitness > g_best.fitness) {
			copy_chromosome(&g_best, p_best[i], num_dims, num_bits_per_dimension);
		}
		for (int j = 0; j < num_dims; j++)
			velocity[i][j] = 0.0;
	}
	double std_dev_prev = calculate_standard_deviation(particles, num_particles, function_option);
	double incline = 0;
	double inc = 0.00001;
	double pm = 0.001;
	double pcross = 0.1;
	std::vector<int> indices(num_particles);
	for (int i = 0; i < num_particles; i++) indices[i] = i;

	for (int iter = 0; iter < max_iterations; iter++) {
		temp_iterations = iter;
		//double w_start = 0.9, w_end = 0.4;
		//double w = w_start - (iter / static_cast<double>(max_iterations)) * (w_start - w_end);
		//w = 0.4 * pow((1 - iter/max_iterations),2) + 0.01;
		//c1 = 2 * pow((1 - iter / max_iterations), 2) + 0.04;
		//c2 = 2 * pow((1 - iter / max_iterations), 2) + 0.04;
		/*w = 0.9 - iter * 0.0001;
		c1 = 1.75 - iter * 0.0002;
		c2 = 2.25 - iter * 0.0002; pt fct 3 dim 30*/
		w = std::max(0.8 - iter * 0.0001,0.15);
		c1 = std::max(2 - iter * 0.0001125,1.0);
		c2 = std::min(1 + iter * 0.0001125,2.0);
		if (iter > 10000)
			pcross -= 0.00005;
		else if (iter > 5000)
			pcross += 0.00005;
		if (iter > 15000) {
			w = std::max(0.15 - (iter-15000) * 0.00001, 0.1);
			c1 = std::min(1.0 + (iter - 15000) * 0.0002, 2.0);
			c2 = std::max(2.0 - (iter - 15000) * 0.0002, 1.0);
		}
		ThreadPool pool(num_threads);	
		for (int i = 0; i < num_particles; ++i) {
			pool.enqueueTask([&, i]() {
				process_particle(particles[i], p_best[i], g_best, velocity[i], num_dims, num_bits_per_dimension, w, c1, c2, generators[i], min_x, max_x, function_option);
				});
		}
		pool.~ThreadPool();

		double std_dev = calculate_standard_deviation(particles, num_particles, function_option);
		incline += (double)(std_dev - std_dev_prev);
		if (iter % 10 == 0) {
			std::random_shuffle(indices.begin(), indices.end());
			ThreadPool pool2(num_threads);
			for (int t = 0; t < num_particles; t += 2) {
				int r1 = indices[t];
				int r2 = indices[t + 1];
				pool2.enqueueTask([&, r1, r2]() {
					process_population_chunk(particles[r1], particles[r2], p_best[r1], p_best[r2], temp_particles[r1], temp_particles[r2], pcross, pm, function_option, num_dims, num_bits_per_dimension, generators[r1 / 2]);
					});
			}
			pool2.~ThreadPool();
		}
		
		if (iter % 100 == 0) {
			adjust_mutation_rate(pm, incline, inc);
			if (hc_option % 2 == 1 && incline > 0) {
				std::random_shuffle(indices.begin(), indices.end());
				ThreadPool pool1(num_threads);
				for (auto k = 0; k < num_particles / 20; k++) {
					int r1 = indices[k];
					pool1.enqueueTask([&, r1]() {
						hill_climbing(&particles[r1] , num_dims, num_bits_per_dimension, function_option, 5 + iter / 500);
						if (particles[r1].fitness > p_best[r1].fitness)
							copy_chromosome(&p_best[r1], particles[r1], num_dims, num_bits_per_dimension);
						});
				}
				hill_climbing(&g_best, num_dims, num_bits_per_dimension, function_option, 5 + iter / 500);
				pool1.~ThreadPool();
			}
			incline = 0.0;
		}
		for (int i = 0; i < num_particles; i++)
			if (p_best[i].fitness > g_best.fitness)
				copy_chromosome(&g_best, p_best[i], num_dims, num_bits_per_dimension);
		f_evolution.push_back(g_best.f);

		//if (g_best.f < 1e-6)
			//break;
		if (std_dev <= 0.00001) {
			adjust_mutation_rate(pm, -1, inc * 100);
		}

		std_dev_prev = std_dev;

		std::cout << "Iteration " << iter + 1 << ": Best fitness: " << g_best.fitness << " f(x) : " << g_best.f << " w,c1,c2: " << w << ", " << c1 << ", " << c2 << " std_dev: " << std_dev << " pm: " << pm << '\n';

	}

	std::cout << "Best solution: x = (";
	for (int i = 0; i < num_dims; ++i) {
		double x_real = binary_to_real(g_best.x + i * num_bits_per_dimension, num_bits_per_dimension, min_x, max_x);
		std::cout << x_real << (i < num_dims - 1 ? ", " : ")\n");
		best_values.push_back(x_real);
	}
	std::cout << " Fitness: " << g_best.fitness << " f(x): " << g_best.f;
	best_f = g_best.f;
	best_fitness = g_best.fitness;

	for (int i = 0; i < num_particles; i++) {
		free(particles[i].x);
		free(p_best[i].x);
		free(velocity[i]);
	}
	free(particles);
	free(p_best);
	free(velocity);
	free(g_best.x);
	log_to_csv(function_option, num_particles, max_iterations, num_dims, num_bits_per_dimension, time_measurement, best_fitness, best_values, f_evolution, best_f);
}

void testing() {
	for (int i = 0; i < 29; i++) {
		start_time = clock();
		particle_swarm_optimization(num_particles, num_dims, num_bits_per_dimension, function_option, max_iterations);
		f_evolution.clear();
		best_values.clear();
	}
}

int main(void)
{
	signal(SIGINT, signalHandler);
	pcross = 0.8;
	num_dims = 100;
	//std::cout << "Enter number of dimensions(2,30,100,others): ";
	//std::cin >> num_dims;
	//sugerate la ora, dar toate fct cu care lucram au min si max in definitie deci ????
	//double min_x = -1e10;
	//double max_x = 1e10;
	function_option = 4;
	hc_option = 1;
	tournament_size = 5;
	//std::cout << "Enter function option (1-4): ";
	//std::cin >> function_option;
	//std::cout << "Enter hill climbing option (1: hc children, 2: hc all at start of generation, 3: both, 4: neither): ";
	//std::cin >> hc_option;
	if (hc_option < 4) {
		pop_size = 100 * num_dims;
		num_gens = 1200;	
		pm = 0.002;
	}
	else {
		pop_size = 150 * num_dims;
		num_gens = 500;
		pm = 0.002;
	}
	num_bits_per_dimension = calculate_num_bits_per_dimension(function_option, PRECISION); //max is for fct3: needs at least 27. i rouded up
	//printf("Number of bits per dimension %d \n", num_bits_per_dimension);
	srand(time(0));

	start_time = clock();
	//genetic_alg(pop_size, num_gens, num_dims, num_bits_per_dimension, pcross, pm, function_option, hc_option, tournament_size);
	//fitness_evolution.clear();

	num_particles = 25 * num_dims;
	max_iterations = 20000;
	//particle_swarm_optimization(num_particles, num_dims, num_bits_per_dimension, function_option, max_iterations);
	testing();
	return 0;
}