#include "image_loader\image_loader.h"
#include "median_gpu_shared\MedianGPU_sh.cuh"
#include "median_gpu\MedianGPU.cuh"
#include "median_cpu\MedianCPU.h"
#include <chrono>
#include <iostream>
using namespace std::chrono;

using FilterFn = RGBImage(*)(const RGBImage&);

auto ApplyFilterAndMeasureTime(FilterFn filter, const RGBImage& image, const int numberOfIterations)
{
	RGBImage filteredImage;

	//ImageLoader::ShowImage(image);

	auto start = high_resolution_clock::now();

	for (int i = 0; i < numberOfIterations; i++)
	{
		filteredImage = filter(image);
	}

	auto stop = high_resolution_clock::now();
	auto duration = duration_cast<milliseconds>(stop - start);

	//ImageLoader::ShowImage(filteredImage);

	return duration;
}

int main()
{
	RGBImage image = ImageLoader::FromFile("C:\\Users\\____\\Desktop\\nasa3.jpg");

	const int arrIterations[] = { 1, 2, 4, 8, 16, 32, 64, 128, 256, 512 };

	std::chrono::milliseconds duration;

	for(const auto numberOfIterations : arrIterations)
	{
		std::cout << "============= Number of iterations: " << numberOfIterations << std::endl;

		duration = ApplyFilterAndMeasureTime(ApplyMedianFilter_CPU, image, numberOfIterations);
		std::cout << "[CPU] :: Duration: " << duration.count() << " ms" << std::endl;

		duration = ApplyFilterAndMeasureTime(ApplyMedianFilter_GPU, image, numberOfIterations);
		std::cout << "[GPU] :: Duration: " << duration.count() << " ms" << std::endl;

		duration = ApplyFilterAndMeasureTime(ApplyMedianFilter_shared, image, numberOfIterations);
		std::cout << "[GPU SHMEM] :: Duration: " << duration.count() << " ms" << std::endl;

		std::cout << std::endl;
	}

	return 0;
}

