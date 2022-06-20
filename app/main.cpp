#include "image_loader\image_loader.h"
#include "median_gpu_shared\MedianGPU_sh.cuh"
#include "median_gpu\MedianGPU.cuh"
#include "median_cpu\MedianCPU.h"
#include <chrono>
#include <iostream>
using namespace std::chrono;

int main()
{
	RGBImage image = ImageLoader::FromFile("C:\\Users\\KASO\\Desktop\\noisy.jpg");

	auto start = high_resolution_clock::now();

	auto filter = ApplyMedianFilter_shared;
	//auto filter = ApplyMedianFilter_GPU;
	//auto filter = ApplyMedianFilter_CPU;

	auto filteredImage = filter(image);
	
	for (int i = 0; i < 100; i++)
	{
		filteredImage = filter(image);
	}
	
	auto stop = high_resolution_clock::now();
	auto duration = duration_cast<microseconds>(stop - start);

	std::cout << "Duration: " << duration.count()/1000 << " ms";

	auto rawFiltered = ImageLoader::FromRawImage(filteredImage);

	

	ImageLoader::ShowImage(rawFiltered);
	ImageLoader::SaveImage("C:\\Users\\KASO\\Desktop\\image.jpg", rawFiltered);

	return 0;
}

