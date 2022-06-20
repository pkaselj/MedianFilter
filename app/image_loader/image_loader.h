#pragma once

#include <string>
#include <vector>
#include <cstdint>
#include <opencv2\imgcodecs.hpp>

struct RGBImage {
	std::string name;
	std::vector<std::vector<uint8_t>> channelR;
	std::vector<std::vector<uint8_t>> channelG;
	std::vector<std::vector<uint8_t>> channelB;
	int width;
	int height;
};

class ImageLoader {
public:
	static RGBImage FromFile(const std::string& image_path);
	static cv::Mat	FromRawImage(const RGBImage& image);
	static void		ShowImage(const cv::Mat& image);
	static void		ShowImage(const RGBImage& image);
	static void		SaveImage(const std::string& path, const cv::Mat& image);
};