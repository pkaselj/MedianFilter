#include "image_loader\image_loader.h"
#include "median_gpu_shared\MedianGPU_sh.cuh"
#include "median_gpu\MedianGPU.cuh"
#include "median_cpu\MedianCPU.h"
#include <chrono>
#include <iostream>
using namespace std::chrono;

int main()
{
	RGBImage image = ImageLoader::FromFile("C:\\Users\\Admin-PC\\Desktop\\noisy.jpg");

	//auto image2 = ImageLoader::FromRawImage(image);

	auto start = high_resolution_clock::now();
	auto filteredImage = ApplyMedianFilter_CPU(image);
	
	for (int i = 0; i < 100; i++)
	{
		filteredImage = ApplyMedianFilter_CPU(image);
	}
	
	auto stop = high_resolution_clock::now();
	auto duration = duration_cast<microseconds>(stop - start);

	std::cout << "Duration: " << duration.count()/1000 << " ms";

	auto rawFiltered = ImageLoader::FromRawImage(filteredImage);

	

	ImageLoader::ShowImage(rawFiltered);
	ImageLoader::SaveImage("C:\\Users\\Admin-PC\\Desktop\\image.jpg", rawFiltered);

	return 0;
}

