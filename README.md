# Hybrid GPU parallelization for CNN: Course project for EECS 587 - Parallel Computing
Contributors: Aditya Modi, Vivek Veeriah

The projects investigates a combination of model parallelism and data parallelism for CNN as follows:

* Use data parallelism in convolutional layers in the network as they have very few parameters but involve a lot of computations over a minibatch.
* Use model parallelism for fully convolutional layers as they have a large number of parameters which makes model synchronization costly.

## Implementation
We use [`cudnn`](https://developer.nvidia.com/cudnn) and [`cuBlas`](https://developer.nvidia.com/cublas) for implementing the network and the back-propagation updates. Parallelization is achieved using a combination of [CUDA-aware MPI](https://developer.nvidia.com/blog/introduction-cuda-aware-mpi/) and cuda libraries.