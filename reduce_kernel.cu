// Make sure that you use this kernel using number of threads = number of elements of d_loss_all
__global__ void ReduceLoss(const float *d_loss_all, float *d_loss_sum, int size)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	for(int stride = 1; stride < blockDim.x; stride *= 2)
	{
		if(i < blockDim.x)
		{
			d_loss_all[i] += d_loss_all[i + stride];
		}
	}
}