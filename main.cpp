#include "image_loader.h"

int main()
{
	RGBImage image = ImageLoader::FromFile("C:\\Users\\KASO\\Desktop\\tesla-roadster.jpg");

	auto image2 = ImageLoader::FromRawImage(image);

	ImageLoader::ShowImage(image2);

	return 0;
}

