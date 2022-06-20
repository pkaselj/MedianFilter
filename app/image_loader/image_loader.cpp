#include "image_loader.h"

#include <stdexcept>
#include <opencv2\opencv.hpp>

#define _IM_LDR_THROW_ERROR(ERR_STREAM_BUILDER_ARGS) { \
					std::stringstream _internal_throw_error_stream; \
					_internal_throw_error_stream << ERR_STREAM_BUILDER_ARGS ; \
					throw std::runtime_error(_internal_throw_error_stream.str()); \
				}

using namespace cv;

static void ClearAndResize2DVector(std::vector<std::vector<uint8_t>>& vec, int height, int width);
static void FillRGBVectorsFromMat(
	const Mat& image,
	std::vector<std::vector<uint8_t>>& R,
	std::vector<std::vector<uint8_t>>& G,
	std::vector<std::vector<uint8_t>>& B
);

RGBImage ImageLoader::FromFile(const std::string& image_path)
{

	if (sizeof(uchar) != sizeof(uint8_t))
	{
		_IM_LDR_THROW_ERROR( "Precondition failed: sizeof(uchar) [= " << sizeof(uchar) << " ] != sizeof(uint8_t) [= " << sizeof(uint8_t) << " ]!" )
	}

	Mat image = imread(image_path, ImreadModes::IMREAD_COLOR);

	if (image.empty())
	{
		_IM_LDR_THROW_ERROR( "Could not load image: " << image_path )
	}

	ImageLoader::ShowImage(image);

	if (image.isContinuous() == false)
	{
		// Assures data countiguity
		image = image.clone();
	}

	RGBImage image_wrapper {
		/* name		*/	image_path,
		/* channelR */	{},
		/* channelG */	{},
		/* channelB */	{},
		/* width	*/	image.cols,
		/* height	*/	image.rows
	};

	FillRGBVectorsFromMat(
		image,
		image_wrapper.channelR,
		image_wrapper.channelG,
		image_wrapper.channelB
	);

	return image_wrapper;
}

static void FillRGBVectorsFromMat(
	const Mat& image,
	std::vector<std::vector<uint8_t>>& R,
	std::vector<std::vector<uint8_t>>& G,
	std::vector<std::vector<uint8_t>>& B
)
{
	const int num_channels = image.channels();
	if (num_channels != 3)
	{
		_IM_LDR_THROW_ERROR("Image has " << num_channels << " channels. Expected: 3")
	}

	const int channel_height = image.rows;
	const int channel_width = image.cols;

	const int image_height = image.rows;
	const int image_width = channel_width * num_channels;

	const uint8_t* pRawImage = (uint8_t*) image.data;

	ClearAndResize2DVector(R, channel_height, channel_width);
	ClearAndResize2DVector(G, channel_height, channel_width);
	ClearAndResize2DVector(B, channel_height, channel_width);

	int image_flat_index = 0;

	for (int i = 0; i < channel_height; i++)
	{
		for (int j = 0; j < channel_width; j++)
		{
			image_flat_index = (i * channel_width + j) * num_channels;

			B[i][j] = pRawImage[image_flat_index + 0];
			G[i][j] = pRawImage[image_flat_index + 1];
			R[i][j] = pRawImage[image_flat_index + 2];
		}
	}
}

static void ClearAndResize2DVector(std::vector<std::vector<uint8_t>>& vec, int height, int width)
{
	vec.clear();
	vec.resize(height);

	for (int i = 0; i < height; i++)
	{
		vec[i].clear();
		vec[i].resize(width);
	}
}


cv::Mat ImageLoader::FromRawImage(const RGBImage& image)
{
	if (sizeof(uchar) != sizeof(uint8_t))
	{
		_IM_LDR_THROW_ERROR( "Precondition failed: sizeof(uchar) [= " << sizeof(uchar) << " ] != sizeof(uint8_t) [= " << sizeof(uint8_t) << " ]!" )
	}

	const int num_channels = 3;
	const int image_flat_array_size = image.width * image.height * num_channels;

	Mat parsedImage(image.height, image.width, CV_8UC3);

	uchar* pRawImage = parsedImage.data;

	int image_flat_index = 0;

	for (int i = 0; i < image.height; i++)
	{
		for (int j = 0; j < image.width; j++)
		{
			image_flat_index = (i * image.width + j) * num_channels;

			pRawImage[image_flat_index + 0] = (uchar) image.channelB[i][j];
			pRawImage[image_flat_index + 1] = (uchar) image.channelG[i][j];
			pRawImage[image_flat_index + 2] = (uchar) image.channelR[i][j];
		}
	}

	return parsedImage;
}

void ImageLoader::ShowImage(const cv::Mat& image)
{
	imshow("Image", image);
	waitKey(0);
}

void ImageLoader::SaveImage(const std::string& path, const cv::Mat& image)
{
	imwrite(path, image);
}

void ImageLoader::ShowImage(const RGBImage& image)
{
	auto rawImage = FromRawImage(image);
	ShowImage(rawImage);
}
