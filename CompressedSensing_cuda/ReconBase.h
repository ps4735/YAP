#pragma once

struct ParameterSet
{
	unsigned int max_line_search_times;
	double gradient_tollerance;
	unsigned int max_conjugate_gradient_iteration_times;
	unsigned int typenorm;
	double tv_weight;
	double wavelet_weight;
	double line_search_alpha;
	double line_search_beta;
	double initial_line_search_step;
	std::vector<float> mask;
	std::vector<std::complex<float>> undersampling_k_space;
};

