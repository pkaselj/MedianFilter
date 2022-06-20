#include "MedianGPU_sh.cuh"

#include <vector>
#include <cstdint>
#include <memory>

#define BLOCK_DIM		32
#define FILTER_WIDTH    3
#define FILTER_HEIGHT   3

static std::vector<std::vector<uint8_t>> ApplyMedianFilterToChannel(const std::vector<std::vector<uint8_t>>& channel, int width, int height);
static std::unique_ptr<uint8_t[]> VectorTo1DArray(const std::vector<std::vector<uint8_t>>& vec, int width, int height);
static std::vector<std::vector<uint8_t>> Array1DToVector(const uint8_t* arr, int width, int height);



static __device__ void sort(uint8_t* filterVector)
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


static __global__
void cuMedianKernel(uint8_t* pOriginal, uint8_t* pResult, const int width, const int height)
{
	const int tx_l = threadIdx.x;                           // --- Local thread x index
	const int ty_l = threadIdx.y;                           // --- Local thread y index

	const int tx_g = blockIdx.x * blockDim.x + tx_l;        // --- Global thread x index
	const int ty_g = blockIdx.y * blockDim.y + ty_l;        // --- Global thread y index

	__shared__ uint8_t smem[BLOCK_DIM + 2][BLOCK_DIM + 2];
	
	// --- Fill the shared memory border with zeros
	if (tx_l == 0)                      
		smem[tx_l][ty_l + 1] = 0;    // --- left border
	else if (tx_l == BLOCK_DIM - 1)     
		smem[tx_l + 2][ty_l + 1] = 0;    // --- right border
	if (ty_l == 0) {
		smem[tx_l + 1][ty_l] = 0;    // --- upper border
		if (tx_l == 0)                  
			smem[tx_l][ty_l] = 0;    // --- top-left corner
		else if (tx_l == BLOCK_DIM - 1) 
			smem[tx_l + 2][ty_l] = 0;    // --- top-right corner
	}
	else if (ty_l == BLOCK_DIM - 1) {
		smem[tx_l + 1][ty_l + 2] = 0;    // --- bottom border
		if (tx_l == 0)                  
			smem[tx_l][ty_l + 2] = 0;    // --- bottom-left corder
		else if (tx_l == BLOCK_DIM - 1) 
			smem[tx_l + 2][ty_l + 2] = 0;    // --- bottom-right corner
	}
	
	// --- Fill shared memory
	smem[tx_l + 1][ty_l + 1] = pOriginal[ty_g * width + tx_g];      // --- center
	if ((tx_l == 0) && ((tx_g > 0)))
		smem[tx_l][ty_l + 1] = pOriginal[ty_g * width + tx_g - 1];      // --- left border
	else if ((tx_l == BLOCK_DIM - 1) && (tx_g < width - 1))         
		smem[tx_l + 2][ty_l + 1] = pOriginal[ty_g * width + tx_g + 1];      // --- right border
	if ((ty_l == 0) && (ty_g > 0)) {
		smem[tx_l + 1][ty_l] = pOriginal[(ty_g - 1) * width + tx_g];    // --- upper border
		if ((tx_l == 0) && ((tx_g > 0)))                                  
			smem[tx_l][ty_l] = pOriginal[(ty_g - 1) * width + tx_g - 1];  // --- top-left corner
		else if ((tx_l == BLOCK_DIM - 1) && (tx_g < width - 1))     
			smem[tx_l + 2][ty_l] = pOriginal[(ty_g - 1) * width + tx_g + 1];  // --- top-right corner
	}
	else if ((ty_l == BLOCK_DIM - 1) && (ty_g < height - 1)) {
		smem[tx_l + 1][ty_l + 2] = pOriginal[(ty_g + 1) * width + tx_g];    // --- bottom border
		if ((tx_l == 0) && ((tx_g > 0)))                                 
			smem[tx_l][ty_l + 2] = pOriginal[(ty_g - 1) * width + tx_g - 1];  // --- bottom-left corder
		else if ((tx_l == BLOCK_DIM - 1) && (tx_g < width - 1))    
			smem[tx_l + 2][ty_l + 2] = pOriginal[(ty_g + 1) * width + tx_g + 1];  // --- bottom-right corner
	}
	
	__syncthreads();
	
	// --- Pull the 3x3 window in a local array
	uint8_t v[9] = {	smem[tx_l][ty_l],		smem[tx_l + 1][ty_l],		smem[tx_l + 2][ty_l],
						smem[tx_l][ty_l + 1],	smem[tx_l + 1][ty_l + 1],   smem[tx_l + 2][ty_l + 1],
						smem[tx_l][ty_l + 2],	smem[tx_l + 1][ty_l + 2],   smem[tx_l + 2][ty_l + 2] };
	
	/*
	uint8_t v[25] = {	smem[tx_l][ty_l],			smem[tx_l + 1][ty_l],			smem[tx_l + 2][ty_l],			smem[tx_l + 3][ty_l],			smem[tx_l + 4][ty_l],
						smem[tx_l][ty_l + 1],		smem[tx_l + 1][ty_l + 1],		smem[tx_l + 2][ty_l + 1],		smem[tx_l + 3][ty_l + 1],		smem[tx_l + 4][ty_l + 1],
						smem[tx_l][ty_l + 2],		smem[tx_l + 1][ty_l + 2],		smem[tx_l + 2][ty_l + 2],		smem[tx_l + 3][ty_l + 2],		smem[tx_l + 4][ty_l + 2],
						smem[tx_l][ty_l + 3],		smem[tx_l + 1][ty_l + 3],		smem[tx_l + 2][ty_l + 3],		smem[tx_l + 3][ty_l + 3],		smem[tx_l + 4][ty_l + 3],
						smem[tx_l][ty_l + 4],		smem[tx_l + 1][ty_l + 4],		smem[tx_l + 2][ty_l + 4],		smem[tx_l + 3][ty_l + 4],		smem[tx_l + 4][ty_l + 4] };
	*/
	sort(v);

	// --- Pick the middle one
	pResult[ty_g * width + tx_g] = v[(FILTER_WIDTH * FILTER_HEIGHT)/2];
}

RGBImage ApplyMedianFilter_shared(const RGBImage& originalImage)
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


	// Create CUDA GPU arrays for original and filtered images
	uint8_t* cudaOriginalChannelArray;
	uint8_t* cudaResultChannelArray;

	const int size = resultChannelArraySize * sizeof(uint8_t);

	cudaMalloc(&cudaOriginalChannelArray, size);
	cudaMalloc(&cudaResultChannelArray, size);

	cudaMemcpy(cudaOriginalChannelArray, originalChannel1DArray.get(), size, cudaMemcpyHostToDevice);
	//cudaMemcpy(cudaResultChannelArray, resultChannel1DArray.get(), size, cudaMemcpyHostToDevice);

	// Apply kernel to image
	//const int N = width * height;
	//cuMedianKernel <<< (N + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>> (cudaOriginalChannelArray, cudaResultChannelArray, width, height);

	// Specify block size
	const dim3 block(BLOCK_DIM, BLOCK_DIM);

	// Calculate grid size to cover the whole image
	const dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);


	cuMedianKernel <<<grid, block>>> (cudaOriginalChannelArray, cudaResultChannelArray, width, height);

	
	// Copy results from GPU
	cudaMemcpy(resultChannel1DArray.get(), cudaResultChannelArray, size, cudaMemcpyDeviceToHost);


	// Release resources
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