
const int size = 1024;
const int blockSize = 1024;

#include <omp.h>

void cpuScan(int *input, int *output, int length)
{
	int acc = 0;
	for (int i = 0; i < length; i++){
		output[i] = acc;
		acc += input[i];
	}
}

void genData(int *output, int length)
{
	for (int i = 0; i < length; i ++){
		output[i] = i;
	}
}

bool checkResults(int *result, int *expected, int length)
{
	for (int i = 0; i < length; i++)
		if (result[i] != expected[i])
			return false;
	return true;
}

template<int WARPSIZE>
__device__ int getTrueId(int i)
{
	return (i/WARPSIZE) * (WARPSIZE+1) + (i % WARPSIZE);
}
template<int WARPSIZE>
__device__ int getTrueSize(int size)
{
	return (size / WARPSIZE) * (WARPSIZE + 1);
}

template
<int WARPSIZE>
__global__ void blelloch(int *input, int *output)
{
	int offset = blockDim.x * blockIdx.x;
	int realSize = getTrueSize<WARPSIZE>(blockDim.x);
	__shared__ int table[realSize];
	table[getTrueIndex<WARPSIZE>(threadIdx.x)] = input[offset + threadIdx.x];
	__syncthreads();

	for (int limit = blockDim.x/2; limit >= 1; limit /= 2) {
		if (threadIdx.x < limit) {
			int jump = blockDim.x/limit;
			int id = (threadIdx.x + 1) * jump - 1;
			int r = table[getTrueId<WARPSIZE>(id)] + table[getTrueId<WARPSIZE>(id-(jump/2))];
			table[getTrueId<WARPSIZE>(id)] = r;
		}
		__syncthreads();
	}

	if (threadId.x = blockDim.x - 1) {
		table[getTrueId<WARPSIZE>(threadIdx.x)] = 0;
	}

	for (int limit = 1; limit <= blockDim.x/2; limit *= 2) {
		if (threadIdx.x < limit) {
			int jump = blockDim.x/limit;
			int id = (threadIdx.x +1) * jump - 1;
			table[getTrueId<WARPSIZE>(id - (jump/2))] = table[getTrueId<WARPSIZE>(id)];
			int r = table[getTrueId<WARPSIZE>(id)] + table[getTrueId<WARPSIZE>(id-(jump/2))];
			table[getTrueId<WARPSIZE>(id)] = r;
		}

		__syncthreads();
	}
}

//template
//<int WARPSIZE>
//__global__ void blellochP2(int size, int *input, int *output)
//{
//	int offset = blockDim.x * blockIdx.x;
//	int realSize = getTrueSize<WARPSIZE>(blockDim.x);
//	__shared__ table[realSize];
//	for (int tid = threadIdx.x; tid < size; tid += blockDim.x) {
//		table[getTrueIndex<WARPSIZE>(tid)] = input[tid];
//	}
//	__syncthreads();
//
//	int tid = (threadIdx.x + 1) * 2;
//
//	for (int stride = size/2; size >= 1; stride /= 2) {
//		
//	}
//	_syncthreads();
//
//	for (int tid = threadIdx.x; tid < size; tid += blockDim.x) {
//		 output[tid] = table[getTrueIndex<WARPSIZE>(tid)];
//	}
//
//
//}

template
<int SIZE,
int BLOCKSIZE,
int WARPSIZE
>
int checkKernel(int *input, int *output, int *dev_input, int *dev_output, int *expected) {
	cudaMemcpy(dev_input, input, sizeof(int) * size, cudaMemcpyHostToDevice);

	blelloch<WARPSIZE>
		<<<1, SIZE>>>(dev_input, dev_output);

	cudaMemcpy(output, dev_output, sizeof(int) * size, cudaMemcpyDeviceToHost);

	if (checkResults(output, expected, size)) {
		printf("wszystko ok\n");
	} else {
		printf("blad\n");
	}
	return 0;
}


int main() {
	int *input;

	int *output;
	int *dev_input;
	int *dev_output;
	int *expected;
	cudaMallocHost(&input, sizeof(int) * size);
	cudaMallocHost(&output, sizeof(int) * size);
	cudaMalloc(&dev_input, sizeof(int) * size);
	cudaMalloc(&dev_output, sizeof(int) * size);
	cudaMallocHost(&expected, sizeof(int) * size);

	genData(input, size);

	cpuScan(input, expeceted, size);

	checkKernel<size, blockSize, 32>(input, output, dev_input, dev_output, expected);

	cudaFree(input);
	cudaFree(output);
	cudaFree(dev_input);
	cudaFree(dev_output);



}
