#include "MedianCPU.h"

#include <vector>
#include <cstdint>
#include <memory>

#define BLOCK_DIM		32
#define FILTER_WIDTH    3
#define FILTER_HEIGHT   3

#define FLAT_INDEX_TO_ROW(INDEX, COLUMNS)		(int)((INDEX) / (COLUMNS))
#define FLAT_INDEX_TO_COLUMN(INDEX, COLUMNS)	(int)((INDEX) % (COLUMNS))

#define ROW_COL_TO_FLAT_INDEX(ROW, COL, WIDTH)	(int)((ROW) * (WIDTH) + (COL))


static std::vector<std::vector<uint8_t>> ApplyMedianFilterToChannel(const std::vector<std::vector<uint8_t>>& channel, int width, int height);
static std::unique_ptr<uint8_t[]> VectorTo1DArray(const std::vector<std::vector<uint8_t>>& vec, int width, int height);
static std::vector<std::vector<uint8_t>> Array1DToVector(const uint8_t* arr, int width, int height);



static void sort(uint8_t* filterVector)
{
	for (int i = 0; i < FILTER_WIDTH * FILTER_HEIGHT; i++) {
		for (int j = i + 1; j < FILTER_WIDTH * FILTER_HEIGHT; j++) {
			if (filterVector[i] > filterVector[j]) {
				//Swap the variables
				uint8_t tmp = filterVector[i];
				filterVector[i] = filterVector[j];
				filterVector[j] = tmp;
			}
		}
	}
}


void MedianKernel(uint8_t* pOriginal, uint8_t* pResult, const int width, const int height)
{
	for (int i = 0; i < width * height; i++) 
	{

		int row = FLAT_INDEX_TO_ROW(i, width);
		int col = FLAT_INDEX_TO_COLUMN(i, width);

		bool isRowBoundary = (col == 0 || col == width - 1);
		bool isColumnBoundary = (row == 0 || row == height - 1);

		// Boundary condition
		if (isRowBoundary || isColumnBoundary)
		{
			pResult[i] = pOriginal[i];
		}
		else
		{
			// +=====================+=====================+=====================+
			// | (row - 1, col - 1)  | (row - 1, col - 0)  | (row - 1, col + 1)  |
			// +=====================+=====================+=====================+
			// | (row - 0, col - 1)  | (row - 0, col - 0)  | (row - 0, col + 1)  |
			// +=====================+=====================+=====================+
			// | (row + 1, col - 1)  | (row + 1, col - 0)  | (row + 1, col + 1)  |
			// +=====================+=====================+=====================+
			uint8_t	pNeighbourhood[] =
			{
				pOriginal[ROW_COL_TO_FLAT_INDEX(row - 1, col - 1, width)],
				pOriginal[ROW_COL_TO_FLAT_INDEX(row - 1, col - 0, width)],
				pOriginal[ROW_COL_TO_FLAT_INDEX(row - 1, col + 1, width)],

				pOriginal[ROW_COL_TO_FLAT_INDEX(row - 0, col - 1, width)],
				pOriginal[ROW_COL_TO_FLAT_INDEX(row - 0, col - 0, width)],
				pOriginal[ROW_COL_TO_FLAT_INDEX(row - 0, col + 1, width)],

				pOriginal[ROW_COL_TO_FLAT_INDEX(row + 1, col - 1, width)],
				pOriginal[ROW_COL_TO_FLAT_INDEX(row + 1, col - 0, width)],
				pOriginal[ROW_COL_TO_FLAT_INDEX(row + 1, col + 1, width)]
			};


			sort(pNeighbourhood);

			uint8_t medianValue = pNeighbourhood[4];

			pResult[i] = medianValue;
		}
	}
}


RGBImage ApplyMedianFilter_CPU(const RGBImage& originalImage)
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

	// Create CPU arrays for original and filtered image
	auto originalChannel1DArray = VectorTo1DArray(channel, width, height);
	auto resultChannel1DArray = std::make_unique<uint8_t[]>(resultChannelArraySize);

	const int size = resultChannelArraySize * sizeof(uint8_t);

	uint8_t* originalChannelArray = (uint8_t*)malloc(size);
	uint8_t* resultChannelArray = (uint8_t*)malloc(size);

	memcpy(originalChannelArray, originalChannel1DArray.get(), size);

	MedianKernel(originalChannelArray, resultChannelArray, width, height);
	
	memcpy(resultChannel1DArray.get(), resultChannelArray, size);

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