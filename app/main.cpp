#include "image_loader\image_loader.h"
#include "median_gpu\MedianGPU.cuh"

int main()
{
	RGBImage image = ImageLoader::FromFile("C:\\Users\\___\\Desktop\\noisy.jpg");

	auto image2 = ImageLoader::FromRawImage(image);

	auto filteredImage = ApplyMedianFilter(image);

	auto rawFiltered = ImageLoader::FromRawImage(filteredImage);

	ImageLoader::ShowImage(rawFiltered);
	ImageLoader::SaveImage("C:\\Users\\_____\\Desktop\\image.jpg", rawFiltered);

	return 0;
}

