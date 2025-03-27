#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <climits>
#include <cstdlib>
#include <ctime>
#include <limits.h>
#include <random>
#include <thread>
#include "./ThreadPool.h"
#include <csignal>
#include <sstream>
#include <fstream>
#include <iomanip>
#define V 76
#define M 7
#define START 0
#define POP_SIZE 5000
#define GENS 1000
#define PATIENCE 50

#define ALPHA 0.75
#define BETA 2.25
#define EVAPORATION_RATE 0.05
#define Q 100.0
#define MAX_ITER 200

double alpha = ALPHA;
double beta = BETA;

struct route {
    int* cities;
    int len;
};

struct chromosome {
    route* routes;
    double f;
    double fitness;
    double amplitude;
    double total_cost;
    int min_index = 0;
};
double starting_pm = 0.02;
double pcross = 0.8;
double pm;
clock_t start_time;
clock_t end_time;
std::vector<double> f_evolution;
int temp_gen;
chromosome best_chromosome;


std::string vector_to_string(const std::vector<double>& values) {
    std::ostringstream oss;
    for (size_t i = 0; i < values.size(); ++i) {
        oss << std::fixed << std::setprecision(6) << values[i];
        if (i < values.size() - 1) {
            oss << "; ";
        }
    }
    return oss.str();
}
std::string routes_to_string(route* routes) {
    std::ostringstream oss;
    for (int i = 0; i < M; ++i) {
        oss << "[";
        for (int j = 0; j < routes[i].len; j++) {
            oss << std::fixed << std::setprecision(6) << routes[i].cities[j];
            if (j < routes[i].len - 1) {
                oss << "; ";
            }
        }
        oss << "]";
    }
    return oss.str();
}
void log_to_csv(const char* dataset_name, int temporary_num_gens, chromosome best) {
    end_time = clock();
    double time_measurement = static_cast<double>(end_time - start_time) / CLOCKS_PER_SEC;

    std::ostringstream filename;
    filename << "results_" << dataset_name
        << "_num_salesmen_" << M << ".csv";
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
            file << "dataset_name,num_salesmen,pop_size,num_gens,pcross,pm,time_measurement,"
                << "best_fitness,best_f,best_total_cost,best_amplitude,best_x,f_evolution\n";
        }
        file << std::fixed << std::setprecision(6);
        file << dataset_name << "," << M << "," << POP_SIZE << "," << temporary_num_gens << ","
            << pcross << "," << pm << "," << time_measurement << ","
            << best.fitness << "," << best.f << "," << best.total_cost << "," << best.amplitude << ",[" << routes_to_string(best.routes) << "],"
            << "[" << vector_to_string(f_evolution) << "]" << "\n";
        file.flush();
        file.close();
        std::cout << "Results and hyperparameters saved to '" << filename_str << "'.\n";
    }
    else {
        std::cerr << "Unable to open file for writing.\n";
    }
    f_evolution.clear();
}


void adjust_mutation_rate(double& pm, double incline, double inc) {
    inc /= 10;
    if (incline <= 0.0)
        pm = std::min(pm + 5 * inc, 0.1);
    else
        pm = std::max(pm - 5 * inc, 0.00001);
    pm = std::max(pm - 3 * inc, 0.00001);
}

void copy_chromosome(chromosome& dest, chromosome source) {
    dest.f = source.f;
    dest.fitness = source.fitness;
    dest.amplitude = source.amplitude;
    dest.total_cost = source.total_cost;
    for (int i = 0; i < M; i++) {
        dest.routes[i].len = source.routes[i].len;
        for (int j = 0; j < source.routes[i].len; j++)
            dest.routes[i].cities[j] = source.routes[i].cities[j];
    }
}

double calculate_standard_deviation(chromosome* population) {
    double sum = 0.0;
    for (int i = 0; i < POP_SIZE; ++i) {
        sum += population[i].f;
    }
    double mean = sum / POP_SIZE;

    double variance_sum = 0.0;
    for (int i = 0; i < POP_SIZE; ++i) {
        double diff = population[i].f - mean;
        variance_sum += diff * diff;
    }
    double variance = variance_sum / POP_SIZE;
    return sqrt(variance);
}

int rand_num(int start, int end, std::mt19937& gen) {
    std::uniform_int_distribution<> dis(start, end);
    return dis(gen);
}

double euclidean_distance(int x1, int y1, int x2, int y2) {
    return sqrt(pow(x2 - x1, 2) + pow(y2 - y1, 2));
}

double calculate_route_length(route r, double* map) {
    double length = 0.0;
    for (int i = 1; i < r.len; ++i) {
        length += map[r.cities[i - 1] * V + r.cities[i]];
    }
    return length;
}


double calculate_amplitude(const chromosome& ind, double* map) {
    double max_length = 0.0;
    double min_length = DBL_MAX;
    for (int i = 0; i < M; i++) {
        double length = calculate_route_length(ind.routes[i], map);
        max_length = std::max(max_length, length);
        min_length = std::min(min_length, length);
    }
    return max_length - min_length;
}


double calculate_total_cost(const chromosome& ind, double* map) {
    double total_cost = 0.0;
    for (int i = 0; i < M; i++) {
        total_cost += calculate_route_length(ind.routes[i], map);
    }
    return total_cost;
}

double calc_f(chromosome& ind, double* map) {
    double max_length = 0.0;
    for (int i = 0; i < M; i++) {
        max_length = std::max(max_length, calculate_route_length(ind.routes[i], map));
    }
    return max_length;
}
void mutate(chromosome& ind, std::mt19937& gen) {
    std::uniform_real_distribution<> dis(0.0, 1.0);
    double random_value = dis(gen);

    if (random_value < 0.25) {
        int r = rand_num(0, M-1, gen);
        ind.routes[r].len--;
        if (ind.routes[r].len > 4) {
            int start_idx = rand_num(1, ind.routes[r].len - 3, gen);
            int end_idx = rand_num(start_idx + 1, ind.routes[r].len - 2, gen);
            while (start_idx < end_idx) {
                std::swap(ind.routes[r].cities[start_idx], ind.routes[r].cities[end_idx]);
                start_idx++;
                end_idx--;
            }
        }
        ind.routes[r].len++;
    }
  
    else if (random_value < 0.5) {
        int r = rand_num(0, M-1, gen);
        if (ind.routes[r].len > 4) {
            int idx1 = rand_num(1, ind.routes[r].len - 3, gen);
            int idx2 = rand_num(idx1+1, ind.routes[r].len - 2, gen);
            std::swap(ind.routes[r].cities[idx1], ind.routes[r].cities[idx2]);
        }
    }
    else if (random_value < 0.75) {
        int r = rand_num(0, M - 1, gen);
        if (ind.routes[r].len > 4) {
            int idx1 = rand_num(1, ind.routes[r].len - 3, gen);
            int idx2 = rand_num(idx1 + 1, ind.routes[r].len - 2, gen);
            int temp = ind.routes[r].cities[idx2];
            for (int i = idx2; i > idx1; i--) {
                ind.routes[r].cities[i] = ind.routes[r].cities[i - 1];
            }
            ind.routes[r].cities[idx1] = temp;
        }
    }
    else {
        int r1 = rand_num(0, M-1, gen);
        int r2 = rand_num(0, M-1, gen);
        if (r1 != r2 && ind.routes[r1].len > 4) {
            int idx1 = rand_num(1, ind.routes[r1].len - 2, gen);
            int city = ind.routes[r1].cities[idx1];
            for (int i = idx1; i < ind.routes[r1].len; i++) {
                ind.routes[r1].cities[i] = ind.routes[r1].cities[i + 1];
            }
            ind.routes[r1].len--;
            int insert_pos = rand_num(1, ind.routes[r2].len-1, gen);
            for (int i = ind.routes[r2].len; i > insert_pos; i--) {
                ind.routes[r2].cities[i] = ind.routes[r2].cities[i - 1];
            }
            ind.routes[r2].cities[insert_pos] = city;
            ind.routes[r2].len++;
            if (ind.routes[r2].cities[ind.routes[r2].len - 1] == ind.routes[r2].cities[ind.routes[r2].len - 2])
                idx1 = idx1;
        }
    }

}


void pmx_crossover(chromosome parent1, chromosome parent2, chromosome& child1, chromosome& child2, std::mt19937& gen) {
    int p1_cities[V], p2_cities[V], c1_cities[V] = { 0 }, c2_cities[V] = { 0 };
    bool found1[V] = { false }, found2[V] = { false };
    int l1 = 0, l2 = 0;
    for (int r = 0; r < M; ++r) {
        for (int c = 1; c < parent1.routes[r].len-1; ++c)
            p1_cities[l1++] = (parent1.routes[r].cities[c]);
        for (int c = 1; c < parent2.routes[r].len-1; ++c)
            p2_cities[l2++] = (parent2.routes[r].cities[c]);
    }

    int start = rand_num(1, V / 2, gen);
    int end = rand_num(start + 1, V - 1, gen);

    for (int i = start; i < end; ++i) {
        c1_cities[i] = p1_cities[i];
        c2_cities[i] = p2_cities[i];
        found1[c1_cities[i]] = true;
        found2[c2_cities[i]] = true;
    }

    for (int idx = 0, i = 0; i < V - 1; i++) {
        if (!found1[p2_cities[i]]) {
            if (c1_cities[idx] != 0) {
                idx = end;
            }
            c1_cities[idx++] = p2_cities[i];
        }
    }
    for (int idx = 0, i = 0; i < V - 1; i++) {
        if (!found2[p1_cities[i]]) {
            if (c2_cities[idx] != 0) {
                idx = end;
            }
            c2_cities[idx++] = p1_cities[i];
        }
    }
    for (int r = 0, idx1 = 0, idx2 = 0; r < M; ++r) {
        child1.routes[r].len = parent1.routes[r].len;
        child1.routes[r].cities[0] = 0;
        for (int c = 1; c < child1.routes[r].len-1; c++, idx1++) {
            child1.routes[r].cities[c] = c1_cities[idx1];
        }
        child1.routes[r].cities[child1.routes[r].len - 1] = 0;
        child2.routes[r].cities[0] = 0;
        child2.routes[r].len = parent2.routes[r].len;
        for (int c = 1; c < child2.routes[r].len-1; c++, idx2++) {
            child2.routes[r].cities[c] = c2_cities[idx2];
        }
        child2.routes[r].cities[child2.routes[r].len - 1] = 0;
    }
}
void stx_crossover(chromosome parent1, chromosome parent2, chromosome& child1, chromosome& child2, double* map, std::mt19937& gen) {
    bool city_used1[V] = { false }, city_used2[V] = { false };
    int start, end, p2_tour, max_shared, max_point;

    for (int tour_index = 0; tour_index < M; ++tour_index) {
        if (parent1.routes[tour_index].len < 5)
            p2_tour = 0;
        p2_tour = -1;
        max_shared = 0;
        for (int r = 0; r < M; ++r) {
            int shared_cities = 0;
            for (int i = 0; i < parent1.routes[tour_index].len; ++i) {
                for (int j = 0; j < parent2.routes[r].len; ++j) {
                    if (parent1.routes[tour_index].cities[i] == parent2.routes[r].cities[j]) {
                        shared_cities++;
                    }
                }
            }
            if (shared_cities > max_shared) {
                max_shared = shared_cities;
                p2_tour = r;
            }
        }
        if (parent1.routes[tour_index].len < 3 || parent2.routes[p2_tour].len < 3) {
            start = 1;
            end = 1;
        }
        else {
            if (parent1.routes[tour_index].len < parent2.routes[p2_tour].len) {
                max_point = parent1.routes[tour_index].len - 2;
            }
            else {
                max_point = parent2.routes[p2_tour].len - 2;
            }
            start = rand_num(1, max_point, gen);
            end = rand_num(start, max_point, gen);
        }
        child1.routes[tour_index].len = 0;
        child2.routes[tour_index].len = 0;
        child1.routes[tour_index].cities[child1.routes[tour_index].len++] = 0;
        child2.routes[tour_index].cities[child2.routes[tour_index].len++] = 0;

        for (int i = 1; i < start; ++i) {
            if(!city_used1[parent2.routes[p2_tour].cities[i]])
                child1.routes[tour_index].cities[child1.routes[tour_index].len++] = parent2.routes[p2_tour].cities[i];
            if(!city_used2[parent1.routes[tour_index].cities[i]])
                child2.routes[tour_index].cities[child2.routes[tour_index].len++] = parent1.routes[tour_index].cities[i];
            city_used1[parent2.routes[p2_tour].cities[i]] = true;
            city_used2[parent1.routes[tour_index].cities[i]] = true;
        }
        for (int i = start; i < end; ++i) {
            if (!city_used2[parent2.routes[p2_tour].cities[i]])
                child2.routes[tour_index].cities[child2.routes[tour_index].len++] = parent2.routes[p2_tour].cities[i];
            if (!city_used1[parent1.routes[tour_index].cities[i]])
                child1.routes[tour_index].cities[child1.routes[tour_index].len++] = parent1.routes[tour_index].cities[i];
            city_used2[parent2.routes[p2_tour].cities[i]] = true;
            city_used1[parent1.routes[tour_index].cities[i]] = true;
        }
        for (int i = end; i < parent2.routes[p2_tour].len - 1; ++i) {
            if (!city_used1[parent2.routes[p2_tour].cities[i]])
                child1.routes[tour_index].cities[child1.routes[tour_index].len++] = parent2.routes[p2_tour].cities[i];
            city_used1[parent2.routes[p2_tour].cities[i]] = true;
        }
        for (int i = end; i < parent1.routes[tour_index].len - 1; ++i) {
            if (!city_used2[parent1.routes[tour_index].cities[i]])
                child2.routes[tour_index].cities[child2.routes[tour_index].len++] = parent1.routes[tour_index].cities[i];
            city_used2[parent1.routes[tour_index].cities[i]] = true;
        }
        child1.routes[tour_index].cities[child1.routes[tour_index].len++] = 0;
        child2.routes[tour_index].cities[child2.routes[tour_index].len++] = 0;
    }
  
    int best_tour = -1, best_position = -1;
    double best_cost;
    for (int i = 1; i < V; ++i) {
        if (city_used1[i])
            continue;
        best_tour = -1;
        best_position = -1;
        best_cost = 1e9;

        for (int r = 0; r < M; ++r) {
            if (child1.routes[r].len < 3) {
                best_cost = -1;
                best_tour = r;
                best_position = 1;
            }
            for (int pos = 1; pos < child1.routes[r].len - 1; ++pos) {
                double cost = map[child1.routes[r].cities[pos - 1] * V + i] +
                    map[i * V + child1.routes[r].cities[pos]] -
                    map[child1.routes[r].cities[pos - 1] * V + child1.routes[r].cities[pos]];
                if (cost < best_cost) {
                    best_cost = cost;
                    best_tour = r;
                    best_position = pos;
                }
            }
        }
        child1.routes[best_tour].len++;
        for (int k = child1.routes[best_tour].len-1; k > best_position; --k) {
            child1.routes[best_tour].cities[k] = child1.routes[best_tour].cities[k - 1];
        }
        child1.routes[best_tour].cities[best_position] = i;
    }
    for (int i = 1; i < V; ++i) {
        if (city_used2[i])
            continue;
        best_tour = -1;
        best_position = -1;
        best_cost = 1e9;

        for (int r = 0; r < M; ++r) {
            if (child2.routes[r].len < 3) {
                best_cost = -1;
                best_tour = r;
                best_position = 1;
            }
            for (int pos = 1; pos < child2.routes[r].len - 1; ++pos) {
                double cost = map[child2.routes[r].cities[pos - 1] * V + i] +
                    map[i * V + child2.routes[r].cities[pos]] -
                    map[child2.routes[r].cities[pos - 1] * V + child2.routes[r].cities[pos]];
                if (cost < best_cost) {
                    best_cost = cost;
                    best_tour = r;
                    best_position = pos;
                }
            }
        }
        child2.routes[best_tour].len++;
        for (int k = child2.routes[best_tour].len - 1; k > best_position; --k) {
            child2.routes[best_tour].cities[k] = child2.routes[best_tour].cities[k - 1];
        }
        child2.routes[best_tour].cities[best_position] = i;
    }
}


bool lessthan(const chromosome& t1, const chromosome& t2) {
    return t1.fitness > t2.fitness;
}

void tournament_selection(int* k1, int* k2, int tournament_size) {
    int i;
    *k1 = POP_SIZE;
    *k2 = POP_SIZE;
    for (int j = 0; j < tournament_size; j++) {
        i = rand() % POP_SIZE;
        if (i < *k1) {
            *k2 = *k1;
            *k1 = i;
        }
        else if (i < *k2)
            *k2 = i;
    }
}

void hill_climbing(chromosome& ind, double* map, std::mt19937& gen) {
    int max_iterations = 10;
    bool improved = true;

    while (improved && max_iterations--) {
        chromosome best_neighbor = ind;
        double best_fitness = ind.f;
        for (int r = 0; r < M; ++r) {
            for (int i = 1; i < ind.routes[r].len - 2; ++i) {
                for (int j = i + 1; j < ind.routes[r].len - 1; ++j) {
                    int start = i, end = j;
                    while (start < end) {
                        std::swap(ind.routes[r].cities[start], ind.routes[r].cities[end]);
                        start++;
                        end--;
                    }

                    double new_f = calc_f(ind, map);
                    if (new_f < best_fitness) {
                        best_fitness = new_f;
                        copy_chromosome(best_neighbor, ind);
                        improved = true;
                    }

                    start = i, end = j;
                    while (start < end) {
                        std::swap(ind.routes[r].cities[start], ind.routes[r].cities[end]);
                        start++;
                        end--;
                    }
                }
            }
        }

        for (int r = 0; r < M; ++r) {
            for (int i = 1; i < ind.routes[r].len - 2; ++i) {
                for (int j = i + 1; j < ind.routes[r].len - 1; ++j) {
                    int temp = ind.routes[r].cities[j];
                    for (int k = j; k > i; --k) {
                        ind.routes[r].cities[k] = ind.routes[r].cities[k - 1];
                    }
                    ind.routes[r].cities[i] = temp;

                    double new_f = calc_f(ind, map);
                    if (new_f < best_fitness) {
                        best_fitness = new_f;
                        copy_chromosome(best_neighbor, ind);
                        improved = true;
                    }

                    temp = ind.routes[r].cities[i];
                    for (int k = i; k < j; ++k) {
                        ind.routes[r].cities[k] = ind.routes[r].cities[k + 1];
                    }
                    ind.routes[r].cities[j] = temp;
                }
            }
        }

        for (int r = 0; r < M; ++r) {
            for (int i = 1; i < ind.routes[r].len - 2; ++i) {
                for (int j = i + 1; j < ind.routes[r].len - 1; ++j) {
                    std::swap(ind.routes[r].cities[i], ind.routes[r].cities[j]);

                    double new_f = calc_f(ind, map);
                    if (new_f < best_fitness) {
                        best_fitness = new_f;
                        copy_chromosome(best_neighbor, ind);
                        improved = true;
                    }
                    std::swap(ind.routes[r].cities[i], ind.routes[r].cities[j]);
                }
            }
        }
        for (int r1 = 0; r1 < M; ++r1) {
            for (int r2 = 0; r2 < M; ++r2) {
                if (r1 == r2) continue;

                for (int i = 1; i < ind.routes[r1].len - 1; ++i) {
                    int city_to_move = ind.routes[r1].cities[i];

                    for (int k = i; k < ind.routes[r1].len - 1; ++k) {
                        ind.routes[r1].cities[k] = ind.routes[r1].cities[k + 1];
                    }
                    ind.routes[r1].len--;

                    for (int j = 1; j < ind.routes[r2].len; ++j) {
                        for (int k = ind.routes[r2].len; k > j; --k) {
                            ind.routes[r2].cities[k] = ind.routes[r2].cities[k - 1];
                        }
                        ind.routes[r2].cities[j] = city_to_move;
                        ind.routes[r2].len++;

                        double new_f = calc_f(ind, map);
                        if (new_f < best_fitness) {
                            best_fitness = new_f;
                            copy_chromosome(best_neighbor, ind);
                            improved = true;
                        }

                        for (int k = j; k < ind.routes[r2].len - 1; ++k) {
                            ind.routes[r2].cities[k] = ind.routes[r2].cities[k + 1];
                        }
                        ind.routes[r2].len--;
                    }
                    for (int k = ind.routes[r1].len; k > i; --k) {
                        ind.routes[r1].cities[k] = ind.routes[r1].cities[k - 1];
                    }
                    ind.routes[r1].cities[i] = city_to_move;
                    ind.routes[r1].len++;
                }
            }
        }

        if (improved) {
            copy_chromosome(ind, best_neighbor);
            ind.f = best_fitness;
            ind.fitness = 1 / best_fitness;
            ind.amplitude = calculate_amplitude(ind, map);
            ind.total_cost = calculate_total_cost(ind, map);
        }
    }
    printf(".");
}


void process_population_chunk(chromosome& new_population1, chromosome& new_population2, chromosome& temp_population1, chromosome& temp_population2, double pcross, double pm, std::mt19937& gen, double* map) {
    std::uniform_real_distribution<> dis(0.0, 1.0);
    double p = dis(gen);
    copy_chromosome(temp_population1, new_population1);
    copy_chromosome(temp_population2, new_population2);
    if (p < pcross)
        //pmx_crossover(new_population1, new_population2, temp_population1, temp_population2, gen);
        stx_crossover(new_population1, new_population2, temp_population1, temp_population2, map, gen);


    for (int i = 0; i < V / 2; i++) {
        p = dis(gen);
        if (p < pm)
            mutate(temp_population1, gen);
        p = dis(gen);
        if (p < pm)
            mutate(temp_population2, gen);
    }

    temp_population1.f = calc_f(temp_population1, map);
    temp_population1.fitness = 1 / temp_population1.f;
    temp_population1.amplitude = calculate_amplitude(temp_population1, map);
    temp_population1.total_cost = calculate_total_cost(temp_population1, map);

    temp_population2.f = calc_f(temp_population2, map);
    temp_population2.fitness = 1 / temp_population2.f;
    temp_population2.amplitude = calculate_amplitude(temp_population2, map);
    temp_population2.total_cost = calculate_total_cost(temp_population2, map);
    copy_chromosome(new_population1, temp_population1);
    copy_chromosome(new_population2, temp_population2);
}

void MinMaxTSP_GAHC(double* map, const char* dataset_name) {
    int city_index = 0;
    int cities_per_route = (V - 1) / M;
    int extra_cities = (V - 1) % M;
    int route_start = 0;
    int tournament_size = 5;
    double best_f = 1e9;
    int last_improvement = 0;
    double inc = 0.01;
    double std_dev_prev, incline = 0, std_dev;
    pm = starting_pm;
    std::vector<std::mt19937> generators(POP_SIZE / 2);
    for (int i = 0; i < POP_SIZE / 2; i++) {
        std::random_device rd;
        std::mt19937 gen(rd());
        generators[i] = gen;
    }
    int num_threads = std::thread::hardware_concurrency() - 1;

    chromosome* population = (chromosome*)malloc(POP_SIZE * sizeof(chromosome));
    chromosome* new_population = (chromosome*)malloc(POP_SIZE * sizeof(chromosome));
    chromosome* temp_population = (chromosome*)malloc(POP_SIZE * sizeof(chromosome));
    std::vector<int> cities;
    for (int i = 1; i < V; ++i) cities.push_back(i);

    for (int i = 0; i < POP_SIZE; ++i) {
        population[i].routes = (route*)malloc(M * sizeof(route));
        best_chromosome.routes = (route*)malloc(M * sizeof(route));
        new_population[i].routes = (route*)malloc(M * sizeof(route));
        temp_population[i].routes = (route*)malloc(M * sizeof(route));
        for (int j = 0; j < M; j++) {
            population[i].routes[j].cities = (int*)malloc((V+10) * sizeof(int));
            best_chromosome.routes[j].cities = (int*)malloc((V+10) * sizeof(int));
            new_population[i].routes[j].cities = (int*)malloc((V + 10) * sizeof(int));
            temp_population[i].routes[j].cities = (int*)malloc((V + 10) * sizeof(int));
        }

        random_shuffle(cities.begin(), cities.end());
        city_index = 0;

        for (int j = 0; j < M; ++j) {
            population[i].routes[j].cities[0] = 0;

            int route_size = cities_per_route + (j < extra_cities ? 1 : 0);
            population[i].routes[j].len = route_size + 2;
            for (int k = 0; k < route_size; ++k) {
                population[i].routes[j].cities[k + 1] = cities[city_index++];
            }
            population[i].routes[j].cities[route_size+1] = 0;
        }

        population[i].f = calc_f(population[i], map);
        population[i].fitness = 1 / population[i].f;
        population[i].amplitude = calculate_amplitude(population[i], map);
        population[i].total_cost = calculate_total_cost(population[i], map);
    }
    std::sort(population, population + POP_SIZE, lessthan);
    std_dev_prev = calculate_standard_deviation(population);
    printf("Initial population best:\n");
    printf("Routes: ");
    for (int i = 0; i < M; i++) {
        printf("[");
        for (int j = 0; j < population[0].routes[i].len; j++)
            printf("%d ", population[0].routes[i].cities[j]);
        printf("] ");
    }
    printf("Fitness: %f F: %f Amplitude: %f Total Cost: %f pm: %f stdiv: %f", population[0].fitness, population[0].f, population[0].amplitude, population[0].total_cost, pm, std_dev_prev);
    f_evolution.push_back(population[0].f);

    std::vector<int> indices(POP_SIZE);
    for (int i = 0; i < POP_SIZE; i++) indices[i] = i;

    for (int gen = 1; gen <= GENS; gen++) {
        // int base_size = 8;
        // int tournament_size = std::min(9, base_size + (gen % 3));
        for (int i = 0; i < POP_SIZE / 100; i++)
            copy_chromosome(new_population[i], population[i]);

        for (int i = POP_SIZE / 100; i < POP_SIZE; i++) {
            int k1, k2;
            tournament_selection(&k1, &k2, tournament_size);
            copy_chromosome(new_population[i], population[k1]);
        }

        std::random_shuffle(indices.begin(), indices.end());
        ThreadPool pool(num_threads);
        for (int t = 0; t < POP_SIZE; t += 2) {

            int k1 = indices[t];
            int k2 = indices[t + 1];
            pool.enqueueTask([&, k1, k2]() {
                process_population_chunk(new_population[k1], new_population[k2], temp_population[k1], temp_population[k2], pcross, pm, generators[k1 / 2], map);
                });
        }
        pool.~ThreadPool();
        for (int i = 0; i < POP_SIZE / 100; i++) {
            if (new_population[i].fitness > population[i].fitness)
                copy_chromosome(population[i], new_population[i]);
        }
        for (int i = POP_SIZE / 100; i < POP_SIZE; i++) {
            copy_chromosome(population[i], new_population[i]);
        }
        std::sort(population, population + POP_SIZE, lessthan);

        std_dev = calculate_standard_deviation(population);
        incline += (double)(std_dev - std_dev_prev);
        if (gen % 10 == 0) {
            adjust_mutation_rate(pm, incline, inc);
            if (incline > 0) {
                ThreadPool pool1(num_threads);
                for (int i = 0; i < POP_SIZE / 100; i++)
                    pool1.enqueueTask([&, i]() {
                    hill_climbing(population[i], map, generators[i]);
                        });
                pool1.~ThreadPool();
                std::random_shuffle(indices.begin(), indices.end());
                ThreadPool pool2(num_threads);
                for (int i = 0; i < POP_SIZE / 100; i++)
                    pool2.enqueueTask([&, i]() {
                    hill_climbing(population[indices[i]], map, generators[i]);
                        });
                pool2.~ThreadPool();
                std::sort(population, population + POP_SIZE, lessthan);
            }
            incline = 0;
        }
        std_dev_prev = std_dev;
        printf("\nGen %d:\n", gen);
        printf("Routes: ");
        for (int i = 0; i < M; i++) {
            printf("[");
            for (int j = 0; j < population[0].routes[i].len; j++)
                printf("%d ", population[0].routes[i].cities[j]);
            printf("] ");
        }
        printf("Fitness: %f F: %f Amplitude: %f Total Cost: %f pm: %f stdiv: %f", population[0].fitness, population[0].f, population[0].amplitude, population[0].total_cost, pm, std_dev);
        if (best_f > population[0].f) {
            best_f = population[0].f;
            last_improvement = gen;
        }
        copy_chromosome(best_chromosome, population[0]);
        temp_gen = gen;
        f_evolution.push_back(population[0].f);
        if (gen - last_improvement > PATIENCE) {
            break;
        }
    }
    for (int i = 0; i < POP_SIZE; i++) {
        for (int j = 0; j < M; j++) {
            free(population[i].routes[j].cities);
            free(new_population[i].routes[j].cities);
            free(temp_population[i].routes[j].cities);
        }
        free(population[i].routes);
        free(new_population[i].routes);
        free(temp_population[i].routes);
    }

    free(population);
    free(new_population);
    free(temp_population);
    log_to_csv(dataset_name, temp_gen, best_chromosome);
}


void update_pheromone(double** pheromone, chromosome* ants, int num_ants, int num_cities) {
    for (int i = 0; i < num_cities; ++i) {
        for (int j = 0; j < num_cities; ++j) {
            pheromone[i][j] *= (1.0 - EVAPORATION_RATE);
        }
    }

    for (int ant = 0; ant < num_ants; ++ant) {
        for (int route = 0; route < M; ++route) {
            for (int i = 0; i < ants[ant].routes[route].len - 1; ++i) {
                int city_i = ants[ant].routes[route].cities[i];
                int city_j = ants[ant].routes[route].cities[i + 1];
                pheromone[city_i][city_j] += Q / ants[ant].f;
                pheromone[city_j][city_i] += Q / ants[ant].f;
            }
        }
    }
}

void process_ant_population(chromosome& ants, double* map, double* probs, double** pheromone, std::mt19937& gen) {
    bool visited[V] = { false };
    int unvisited_count = V - 1;
    for (int route = 0; route < M; ++route) {
        ants.routes[route].len = 0;
        ants.routes[route].cities[ants.routes[route].len++] = 0;
    }
    while (unvisited_count) {
        int current_city = ants.routes[ants.min_index].cities[ants.routes[ants.min_index].len - 1];
        double sum = 0.0;
        for (int j = 1; j < V; ++j) {
            if (!visited[j]) {
                probs[j] = pow(pheromone[current_city][j], alpha) * pow(1.0 / map[current_city * V + j], beta);
                sum += probs[j];
            }
            else
                probs[j] = 0;
        }
        for (int j = 1; j < V; ++j) {
            if (sum > 0)
                probs[j] /= sum;
            else
                probs[j] = 0;
        }
        std::uniform_real_distribution<> dis(0.0, 1.0);
        double rand_val = dis(gen);
        double cumulative_prob = 0.0; int next_city = -1;
        for (int j = 1; j < V; ++j) {
            if (!visited[j]) {
                cumulative_prob += probs[j];
                if (rand_val <= cumulative_prob) {
                    next_city = j;
                    break;
                }
            }
        }
        if (next_city == -1) {
            for (int j = 1; j < V; ++j) {
                if (!visited[j]) {
                    next_city = j;
                    break;
                }
            }
        }
        ants.routes[ants.min_index].cities[ants.routes[ants.min_index].len++] = next_city;
        visited[next_city] = true;
        unvisited_count--;
        int min_index = ants.min_index;
        double min_cost = calculate_route_length(ants.routes[min_index], map);
        for (int route = 0; route < M; ++route) {
            double cost = calculate_route_length(ants.routes[route], map);
            if (cost < min_cost) {
                min_cost = cost; min_index = route;
            }
        }
        rand_val = dis(gen);
        if (rand_val < 0.99) {
            ants.min_index = min_index;
        }
        else {
            std::uniform_int_distribution<> rand_route(0, M - 1);
            ants.min_index = rand_route(gen);
        }
    }
    for (int route = 0; route < M; ++route) {
        ants.routes[route].cities[ants.routes[route].len++] = 0;
    }

    ants.f = calc_f(ants, map);
    ants.fitness = 1 / ants.f;
    ants.amplitude = calculate_amplitude(ants, map);
    ants.total_cost = calculate_total_cost(ants, map);
}
void two_opt(chromosome& ant, double* map) {
    bool improvement = true;
    int steps = 10;
    while (improvement && steps) {
        improvement = false;
        steps--;
        for (int route = 0; route < M; ++route) {
            for (int i = 1; i < ant.routes[route].len - 2; ++i) {
                for (int j = i + 1; j < ant.routes[route].len - 1; ++j) {
                    double delta = (map[ant.routes[route].cities[i - 1] * V + ant.routes[route].cities[j]]
                        + map[ant.routes[route].cities[i] * V + ant.routes[route].cities[j + 1]]
                        - map[ant.routes[route].cities[i - 1] * V + ant.routes[route].cities[i]]
                        - map[ant.routes[route].cities[j] * V + ant.routes[route].cities[j + 1]]);
                    if (delta < 0) {
                        std::reverse(ant.routes[route].cities + i, ant.routes[route].cities + j + 1);
                        improvement = true;
                    }
                }
            }
        }
    }
}

void MinMaxTSP_ACO(double* map, const char* dataset_name) {
    int num_threads = std::thread::hardware_concurrency();
    std::vector<std::mt19937> generators(POP_SIZE);
    for (int i = 0; i < POP_SIZE / 2; i++) {
        std::random_device rd;
        std::mt19937 gen(rd());
        generators[i] = gen;
    }
    
    double** pheromone = (double**)malloc(V * sizeof(double*));
    double** probs = (double**)malloc(POP_SIZE * sizeof(double*));
    for (int i = 0; i < V; ++i) {
        pheromone[i] = (double*)malloc(V * sizeof(double));
    }
    for (int i = 0; i < POP_SIZE; i++) {
        probs[i] = (double*)malloc(V * sizeof(double));
    }

    for (int i = 0; i < V; ++i) {
        for (int j = 0; j < V; ++j) {
            pheromone[i][j] = 1;
        }
    }

    chromosome* population = (chromosome*)malloc(POP_SIZE * sizeof(chromosome));
    chromosome* new_population = (chromosome*)malloc(POP_SIZE * sizeof(chromosome));
    chromosome* temp_population = (chromosome*)malloc(POP_SIZE * sizeof(chromosome));
    for (int i = 0; i < POP_SIZE; ++i) {
        population[i].routes = (route*)malloc(M * sizeof(route));
        new_population[i].routes = (route*)malloc(M * sizeof(route));
        temp_population[i].routes = (route*)malloc(M * sizeof(route));
        for (int j = 0; j < M; ++j) {
            population[i].routes[j].cities = (int*)malloc(V * sizeof(int));
            new_population[i].routes[j].cities = (int*)malloc(V * sizeof(int));
            temp_population[i].routes[j].cities = (int*)malloc(V * sizeof(int));
        }
        population[i].min_index = 0;
    }

    best_chromosome.fitness = 0;
    std::vector<int> indices(POP_SIZE);
    for (int i = 0; i < POP_SIZE; i++) indices[i] = i;

    for (int iter = 1; iter <= MAX_ITER; ++iter) { 
        temp_gen = iter;
        ThreadPool pool(num_threads);
        for (int i = 0; i < POP_SIZE; ++i) {
            pool.enqueueTask([&, i]() {
                process_ant_population(population[i],map, probs[i], pheromone, generators[i]);
                });
        }
        pool.~ThreadPool();
        int best_ind = 0;
        for (int i = 0; i < POP_SIZE; i++) {
            if (population[i].fitness > population[best_ind].fitness)
                best_ind = i;
        }
        if (iter % 5 == 0) {  
            for (int i = 0; i <= 50; i++) {
                for (int i = 0; i < POP_SIZE / 50; i++)
                    copy_chromosome(new_population[i], population[i]);
                for (int i = POP_SIZE / 50; i < POP_SIZE; i++) {
                    int k1, k2;
                    tournament_selection(&k1, &k2, 5);
                    copy_chromosome(new_population[i], population[k1]);
                }
                std::random_shuffle(indices.begin(), indices.end());
                ThreadPool pool2(num_threads);
                for (int t = 0; t < POP_SIZE; t += 2) {

                    int k1 = indices[t];
                    int k2 = indices[t + 1];
                    pool2.enqueueTask([&, k1, k2]() {
                        process_population_chunk(new_population[k1], new_population[k2], temp_population[k1], temp_population[k2], 0.8, 0.001, generators[k1 / 2], map);
                        });
                }
                pool2.~ThreadPool();
                for (int i = 0; i < POP_SIZE / 50; i++) {
                    if (new_population[i].fitness > population[i].fitness)
                        copy_chromosome(population[i], new_population[i]);
                }
                for (int i = POP_SIZE / 50; i < POP_SIZE; i++) {
                    copy_chromosome(population[i], new_population[i]);
                }
                std::sort(population, population + POP_SIZE, lessthan);
            }
           
            ThreadPool pool3(num_threads);
            for (int t = 0; t < POP_SIZE; t += 2) {

                int k = indices[t];
                pool3.enqueueTask([&, k]() {
                    two_opt(population[k], map);
                    });
            }
            pool3.~ThreadPool();
        }
        if (iter % 10 == 0) {
            std::random_shuffle(indices.begin(), indices.end());
            ThreadPool pool3(num_threads);
            for (int i = 0; i < POP_SIZE / 100; i++)
                pool3.enqueueTask([&, i]() {
                hill_climbing(population[indices[i]], map, generators[i]);
                    });
            pool3.~ThreadPool();
            hill_climbing(population[best_ind], map, generators[best_ind]);
        }
        for (int i = 0; i < POP_SIZE; i++)
            if (population[i].fitness > best_chromosome.fitness)
                copy_chromosome(best_chromosome, population[i]);
        update_pheromone(pheromone, population, POP_SIZE, V);
        

        printf("\nGen %d:\n", iter);
        printf("Routes: ");
        for (int i = 0; i < M; i++) {
            printf("[");
            for (int j = 0; j < best_chromosome.routes[i].len; j++)
                printf("%d ", best_chromosome.routes[i].cities[j]);
            printf("] ");
        }
        printf("Fitness: %f F: %f Amplitude: %f Total Cost: %f, ab %f %f", best_chromosome.fitness, best_chromosome.f, best_chromosome.amplitude, best_chromosome.total_cost, alpha, beta);
        f_evolution.push_back(best_chromosome.f);
        alpha += (BETA-ALPHA) / MAX_ITER;
        beta -= (BETA - ALPHA) / MAX_ITER;
        //if (best_chromosome.f < 4110.2131)
           //break;
    }
    for (int i = 0; i < POP_SIZE; i++) {
        for (int j = 0; j < M; j++) {
            free(population[i].routes[j].cities);
            free(new_population[i].routes[j].cities);
            free(temp_population[i].routes[j].cities);
        }
        free(population[i].routes);
        free(new_population[i].routes);
        free(temp_population[i].routes);
    }

    free(population);
    free(new_population);
    free(temp_population);
    log_to_csv(dataset_name, temp_gen, best_chromosome);

}

void signalHandler(int signum) {
    std::cout << "Interrupt signal (" << signum << ") received.\n";
    log_to_csv("eil76", temp_gen, best_chromosome);
    exit(signum);
}

void testing(double* map) {
    for (int i = 0; i < 30; i++) {
        start_time = clock();
        //MinMaxTSP_GAHC(map, "rat99");
        alpha = ALPHA;
        beta = BETA;
        MinMaxTSP_ACO(map, "eil76");
    }
}


int main() {
    signal(SIGINT, signalHandler);
    //eil51
    /*
    std::vector<std::pair<int, int>> coordinates = {
        {37, 52}, {49, 49}, {52, 64}, {20, 26}, {40, 30},
        {21, 47}, {17, 63}, {31, 62}, {52, 33}, {51, 21},
        {42, 41}, {31, 32}, {5, 25}, {12, 42}, {36, 16},
        {52, 41}, {27, 23}, {17, 33}, {13, 13}, {57, 58},
        {62, 42}, {42, 57}, {16, 57}, {8, 52}, {7, 38},
        {27, 68}, {30, 48}, {43, 67}, {58, 48}, {58, 27},
        {37, 69}, {38, 46}, {46, 10}, {61, 33}, {62, 63},
        {63, 69}, {32, 22}, {45, 35}, {59, 15}, {5, 6},
        {10, 17}, {21, 10}, {5, 64}, {30, 15}, {39, 10},
        {32, 39}, {25, 32}, {25, 55}, {48, 28}, {56, 37},
        {30, 40}
    };
    */
    //berlin52
    /*
    std::vector<std::pair<int, int>> coordinates = {
    {565, 575}, {25, 185}, {345, 750}, {945, 685}, {845, 655},
    {880, 660}, {25, 230}, {525, 1000}, {580, 1175}, {650, 1130},
    {1605, 620}, {1220, 580}, {1465, 200}, {1530, 5}, {845, 680},
    {725, 370}, {145, 665}, {415, 635}, {510, 875}, {560, 365},
    {300, 465}, {520, 585}, {480, 415}, {835, 625}, {975, 580},
    {1215, 245}, {1320, 315}, {1250, 400}, {660, 180}, {410, 250},
    {420, 555}, {575, 665}, {1150, 1160}, {700, 580}, {685, 595},
    {685, 610}, {770, 610}, {795, 645}, {720, 635}, {760, 650},
    {475, 960}, {95, 260}, {875, 920}, {700, 500}, {555, 815},
    {830, 485}, {1170, 65}, {830, 610}, {605, 625}, {595, 360},
    {1340, 725}, {1740, 245}
    };
    */
    //eil76
    
    std::vector<std::pair<int, int>> coordinates = {
    {22, 22}, {36, 26}, {21, 45}, {45, 35}, {55, 20},
    {33, 34}, {50, 50}, {55, 45}, {26, 59}, {40, 66},
    {55, 65}, {35, 51}, {62, 35}, {62, 57}, {62, 24},
    {21, 36}, {33, 44}, {9, 56}, {62, 48}, {66, 14},
    {44, 13}, {26, 13}, {11, 28}, {7, 43}, {17, 64},
    {41, 46}, {55, 34}, {35, 16}, {52, 26}, {43, 26},
    {31, 76}, {22, 53}, {26, 29}, {50, 40}, {55, 50},
    {54, 10}, {60, 15}, {47, 66}, {30, 60}, {30, 50},
    {12, 17}, {15, 14}, {16, 19}, {21, 48}, {50, 30},
    {51, 42}, {50, 15}, {48, 21}, {12, 38}, {15, 56},
    {29, 39}, {54, 38}, {55, 57}, {67, 41}, {10, 70},
    {6, 25}, {65, 27}, {40, 60}, {70, 64}, {64, 4},
    {36, 6}, {30, 20}, {20, 30}, {15, 5}, {50, 70},
    {57, 72}, {45, 42}, {38, 33}, {50, 4}, {66, 8},
    {59, 5}, {35, 60}, {27, 24}, {40, 20}, {40, 37},
    {40, 40}
    };
    
    //rat99
    /*
    std::vector<std::pair<int, int>> coordinates = {
    {6, 4}, {15, 15}, {24, 18}, {33, 12}, {48, 12},
    {57, 14}, {67, 10}, {77, 10}, {86, 15}, {6, 21},
    {17, 26}, {23, 25}, {32, 35}, {43, 23}, {55, 35},
    {65, 36}, {78, 39}, {87, 35}, {3, 53}, {12, 44},
    {28, 53}, {33, 49}, {47, 46}, {55, 52}, {64, 50},
    {71, 57}, {87, 57}, {4, 72}, {15, 78}, {22, 70},
    {34, 71}, {42, 79}, {54, 77}, {66, 79}, {78, 67},
    {87, 73}, {7, 81}, {17, 95}, {26, 98}, {32, 97},
    {43, 88}, {57, 89}, {64, 85}, {78, 83}, {83, 98},
    {5, 109}, {13, 111}, {25, 102}, {38, 119}, {46, 107},
    {58, 110}, {67, 110}, {74, 113}, {88, 110}, {2, 124},
    {17, 134}, {23, 129}, {36, 131}, {42, 137}, {53, 123},
    {63, 135}, {72, 134}, {87, 129}, {2, 146}, {16, 147},
    {25, 153}, {38, 155}, {42, 158}, {57, 154}, {66, 151},
    {73, 151}, {86, 149}, {5, 177}, {13, 162}, {25, 169},
    {35, 177}, {46, 172}, {54, 166}, {65, 174}, {73, 161},
    {86, 162}, {2, 195}, {14, 196}, {28, 189}, {38, 187},
    {46, 195}, {57, 194}, {63, 188}, {77, 193}, {85, 194},
    {8, 211}, {12, 217}, {22, 210}, {34, 216}, {47, 203},
    {58, 213}, {66, 206}, {78, 210}, {85, 204}
    };
    */

    for (int i = 0; i < POP_SIZE; ++i) {
        best_chromosome.routes = (route*)malloc(M * sizeof(route));
        for (int j = 0; j < M; j++) {
            best_chromosome.routes[j].cities = (int*)malloc((V + 10) * sizeof(int));
        }
    }
    double* map = (double*)malloc(V * V * sizeof(double));
    for (int i = 0; i < V; ++i) {
        for (int j = 0; j < V; ++j) {
            if (i != j) {
                map[i * V + j] = euclidean_distance(
                    coordinates[i].first, coordinates[i].second,
                    coordinates[j].first, coordinates[j].second);
            }
            else {
                map[i * V + j] = 0.0;
            }
        }
    }
    testing(map);
    //start_time = clock();
    //MinMaxTSP_GAHC(map, "eil51");
    //MinMaxTSP_ACO(map);
    return 0;
}