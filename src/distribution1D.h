#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <glm/glm.hpp>
#include "cudaUtils.hpp"

//https://www.pbr-book.org/3ed-2018/Monte_Carlo_Integration/Sampling_Random_Variables#Distribution1D
class Distribution1D
{
public:
	std::vector<float> func, cdf;
	float funcInt;

	Distribution1D() = default;

	Distribution1D(std::vector<float> vals);

	Distribution1D(const float* vals, int n);

	int Count() const;

	float sampleContinuous(float u, float& pdf);

	int sampleDiscrete(float u, float& pdf);
};

class DevDistribution1D
{
public:
	float* func = nullptr, * cdf = nullptr;
	float funcInt = 0;
	int size = 0;

	void create(Distribution1D& hstSampler);

	void destroy();

	__host__ __device__ float sampleContinuous(float u, float& pdf);

	__host__ __device__ int sampleDiscrete(float u, float& pdf);
};