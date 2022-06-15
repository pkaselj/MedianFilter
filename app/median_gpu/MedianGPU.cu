#include "MedianGPU.cuh"

#include <vector>
#include <cstdint>
#include <memory>

//#define N 1280*720
#define BLOCK_SIZE 256

std::vector<std::vector<uint8_t>> ApplyMedianFilterToChannel(const std::vector<std::vector<uint8_t>>& channel, int width, int height);
std::unique_ptr<uint8_t[]> VectorTo1DArray(const std::vector<std::vector<uint8_t>>& vec, int width, int height);
std::vector<std::vector<uint8_t>> Array1DToVector(const uint8_t* arr, int width, int height);

__global__
void cuMedianKernel(uint8_t* pOriginal, uint8_t* pResult)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	pResult[i] = pOriginal[i] / 4;
}

RGBImage ApplyMedianFilter(const RGBImage& originalImage)
{
	return RGBImage
	{
		originalImage.name,
		ApplyMedianFilterToChannel(originalImage.channelR, originalImage.width, originalImage.height),
		ApplyMedianFilterToChannel(originalImage.channelG, originalImage.width, originalImage.height),
		ApplyMedianFilterToChannel(originalImage.channelB, originalImage.width, originalImage.height),
		originalImage.width,
		originalImage.height
	};
}

std::vector<std::vector<uint8_t>> ApplyMedianFilterToChannel(const std::vector<std::vector<uint8_t>>& channel, int width, int height)
{
	// Size of flat (1D) array for a channel (R, G or B) of filtered image
	// Image will be padded with zeroes on each border (1px width of padding)
	const int resultChannelArraySize = height * width;

	auto originalChannel1DArray = VectorTo1DArray(channel, width, height);
	auto resultChannel1DArray = std::make_unique<uint8_t[]>(resultChannelArraySize);

	uint8_t* cudaOriginalChannelArray;
	uint8_t* cudaResultChannelArray;

	const int size = resultChannelArraySize * sizeof(uint8_t);

	cudaMalloc(&cudaOriginalChannelArray, size);
	cudaMalloc(&cudaResultChannelArray, size);

	cudaMemcpy(cudaOriginalChannelArray, originalChannel1DArray.get(), size, cudaMemcpyHostToDevice);
	cudaMemcpy(cudaResultChannelArray, resultChannel1DArray.get(), size, cudaMemcpyHostToDevice);

	const int N = width * height;
	cuMedianKernel <<< (N + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE >>> (cudaOriginalChannelArray, cudaResultChannelArray);

	cudaMemcpy(resultChannel1DArray.get(), cudaResultChannelArray, size, cudaMemcpyDeviceToHost);

	cudaFree(cudaOriginalChannelArray);
	cudaFree(cudaResultChannelArray);

	return Array1DToVector(resultChannel1DArray.get(), width, height);
}

std::unique_ptr<uint8_t[]> VectorTo1DArray(const std::vector<std::vector<uint8_t>>& vec, int width, int height)
{
	const int size = width * height;

	auto array_1D = std::make_unique<uint8_t[]>(size);

	const int rows = vec.size();
	for (int i = 0; i < rows; i++)
	{
		const std::vector<uint8_t>& currentRow = vec[i];
		const int columns = currentRow.size();

		const int offset = i * columns;

		uint8_t* pDestination = &(array_1D.get())[offset]; // &array_1D[offset]
		const uint8_t* pSource = currentRow.data();					// &vec[i]
		const int count = columns;									

		memcpy(pDestination, pSource, count * sizeof(uint8_t));
	}

	return std::move(array_1D);
}

std::vector<std::vector<uint8_t>> Array1DToVector(const uint8_t* arr, int width, int height)
{
	const int size = width * height;

	std::vector<std::vector<uint8_t>> vec;
	vec.resize(height);

	for (int i = 0; i < height; i++)
	{
		auto& currentRow = vec[i];
		currentRow.resize(width);

		const int offset = i * width;
		const uint8_t* pSource = (uint8_t*)(arr + offset);	// &arr[offset]
		uint8_t* pDestination = currentRow.data();			// &vec[i]
		int count = width;

		memcpy(pDestination, pSource, count * sizeof(uint8_t));
	}

	return vec;
}