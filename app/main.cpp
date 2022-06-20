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

	ImageLoader::ShowImage(image);

	auto start = high_resolution_clock::now();

	for (int i = 0; i < 100; i++)
	{
		filteredImage = filter(image);
	}

	auto stop = high_resolution_clock::now();
	auto duration = duration_cast<milliseconds>(stop - start);

	ImageLoader::ShowImage(filteredImage);

	return duration;
}

int main()
{
	RGBImage image = ImageLoader::FromFile("C:\\Users\\KASO\\Desktop\\noisy.jpg");

	const int arrIterations[] = { 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048 };

	for(const auto numberOfIterations : arrIterations)
	{

	}


	
	
	auto filter = ApplyMedianFilter_shared;
	//auto filter = ApplyMedianFilter_GPU;
	//auto filter = ApplyMedianFilter_CPU;
	

	


	std::cout << "Duration: " << duration.count() << " ms";

	auto rawFiltered = ImageLoader::FromRawImage(filteredImage);

	

	return 0;
}

