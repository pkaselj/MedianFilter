#include "image_loader\image_loader.h"

int main()
{
	RGBImage image = ImageLoader::FromFile("res\\tesla-roadster.jpg");

	auto image2 = ImageLoader::FromRawImage(image);

	ImageLoader::ShowImage(image2);

	return 0;
}

