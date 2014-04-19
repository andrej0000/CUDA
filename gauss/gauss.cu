#include <omp.h>

#include<cstdio>
#include<cstdlib>
#include<ctime>
#include<cmath>




const int dataSize = 16;
const int blockSize = 16;
const int gridSize = dataSize/blockSize;
const int maskSize = 11;

const float pi = 3.14;
const float e = 2.71828;
const int stdDev = 1;

texture<float, 2> tex;


__global__ void gauss(float * output, int dataSize, float * mask, int maskRadius) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	float result = 0;
	int mS = maskRadius*2+1;
	for (int dx = -maskRadius; dx <= maskRadius; dx++){
		for (int dy = -maskRadius; dy <= maskRadius; dy++) {
			float pix = 0;
			if (x+dx >= 0 && x+dx < dataSize)
				if (y+dy >= 0 && y+dy < dataSize)
					pix = tex2D(tex, x+dx, y+dy);
			result += mask[mS * (maskRadius + dx) + dy + maskRadius]*pix;
		}
	}
	output[(dataSize*x)+y] = result;

}

float filter(int dx, int dy) {
	float exp = 0;
	exp -= dx*dx + dy*dy;
	exp /= 2*pi*stdDev*stdDev;
	float p = pow(e, exp);
	return p/2*pi*stdDev*stdDev;
}

int main() {
	srand(time(NULL));

	float * inputDev;
	float * input;
	float * outputDev;
	float * output;
	float * mask;
	float * maskDev;
	cudaMalloc(&inputDev, dataSize*dataSize*sizeof(float));
	cudaMallocHost(&input, dataSize*dataSize*sizeof(float));
	cudaMalloc(&outputDev, dataSize*dataSize*sizeof(float));
	cudaMallocHost(&output, dataSize*dataSize*sizeof(float));
	cudaMallocHost(&mask, maskSize*maskSize*sizeof(float));
	cudaMalloc(&maskDev, maskSize*maskSize*sizeof(float));


	
	for (int x = 0; x < maskSize; x++) {
		for (int y = 0; y < maskSize; y++) {
			int cx = x - maskSize/2;
			int cy = y - maskSize/2;
			mask[x*maskSize+y] = filter(cx, cy);
		}
	}
	cudaMemcpy(maskDev, mask, maskSize*maskSize*sizeof(float), cudaMemcpyHostToDevice);
	//Generowanie floatow z zakresu [0,1)
	for(int i=0;i<dataSize*dataSize;i++) {
		input[i] = 0; //rand()/(float)(RAND_MAX);
	}
	input[7*16+7] = 1;
	cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();
	cudaBindTexture2D(NULL, tex, inputDev, desc, dataSize, dataSize, dataSize*sizeof(float));

	cudaMemcpy(inputDev, input, dataSize*dataSize*sizeof(float), cudaMemcpyHostToDevice);

	dim3 dimBlock(blockSize, blockSize);
	dim3 dimGrid(gridSize, gridSize);
	gauss<<<dimGrid, dimBlock>>>(outputDev, dataSize, maskDev, maskSize/2);

	cudaMemcpy(output, outputDev, dataSize*dataSize*sizeof(float), cudaMemcpyDeviceToHost);

	for (int x = 0; x < dataSize; x++) {
		for (int y = 0; y < dataSize; y++) {
			printf("%f\t", output[x*dataSize + y]);
		}
		printf("\n");
	}

	cudaUnbindTexture(tex);
	cudaFree(inputDev);
	cudaFree(outputDev);
	cudaFree(maskDev);
	return 0;
}
