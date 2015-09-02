"""Class containing the gpu execution function.

    Use: Initialize with no argument to let pycuda compile the kernel, then call exec() with the array of inputs, the
         size of all outputs, and the NEAT network to use. Be sure to look at the function definition for the specific
         type for the input.
"""

import sys
import numpy
import MultiNEAT as NEAT
import math
import pycuda.driver
import pycuda.autoinit
import pycuda.gpuarray
from pycuda.compiler import SourceModule
import pycuda.cumath
import pycuda.elementwise


class GpuExec:

    def __init__(self):
        sys.path.append('/usr/local/cuda/bin')
        sys.path.append('/usr/local/cuda/lib64')

        # The kernel
        gpu_exec_mod = SourceModule("""
        __global__ void gpu_exec(float* d_inputs,
                                 float* d_results, float* d_m_source, float* d_m_weight,
                                 float* d_m_target,
                                 float* d_m_a, float* d_m_b, int c_size,
                                 int n_size, int i_size, int num_inputs, float* signal_malloc,
                                 float* activesum_malloc, int a_size){

            int tx(blockIdx.x * blockDim.x + threadIdx.x);

            if (tx < a_size){
                float y = 0.0;
                int in_size = int(i_size/a_size);
                float* m_signal(signal_malloc + tx * c_size);
                float* m_activesum(activesum_malloc + tx * c_size);

                for (int i = 0; i < 3; i++){
                    for (int i = 0; i < c_size; i++){
                        int id = d_m_source[i];
                        m_signal[i] = (d_inputs[in_size*tx + id]) * d_m_weight[i];
                    }

                    for (int i = 0; i < c_size; i++){
                        int id = d_m_target[i];
                        m_activesum[id] = m_activesum[id] + m_signal[i];
                    }

                    for (unsigned int i = num_inputs; i < n_size; i++)
                    {
                        float x = m_activesum[i];
                        float a = d_m_a[i];
                        float b = d_m_a[i];
                        //y = exp(-a*x*x - b);
                        y = 1.0/(1.0 + exp(-a*x-b));
                    }
                }

                d_results[tx] = y;
            }
        }
        """)

        self.gpu_exec = gpu_exec_mod.get_function("gpu_exec")

    def eval(self, full_input, input_size, output_size, net):
        """Evaluate the whole training space in parallel for a given network.

        :param full_input: Inputs for the all training samples concatenated into a single numpy array. Should be of the
                           type ndarray(input_size, dtype=numpy.float32).
        :param input_size: The size of one input sample.
        :param output_size: The total number of outputs for all training samples.
        :param net: The NEAT network to be evaluated.
        :return: A vector of outputs for all training samples.
        """

        # Host and device vectors for inputs and outputs
        h_results = numpy.zeros(output_size, dtype=numpy.float32)  # Container for device outputs to be transferred
        output = numpy.zeros(output_size, dtype=numpy.float32)     # Container for outputs
        d_results = pycuda.gpuarray.to_gpu(h_results)              # Container for outputs on device
        i_size = full_input.size                                   # Size of all inputs
        a_size = i_size/input_size                                 # Number of input samples
        d_full_input = pycuda.gpuarray.to_gpu(full_input)          # Device vector of all inputs

        # Make the net output its parameters
        net.ActivateFast()
        net_params = net.Output()

        # Get the size of various constants
        num_inputs_float = net_params[len(net_params) - 1]
        n_size_float = net_params[len(net_params) - 2]
        c_size_float = net_params[len(net_params) - 3]
        c_size_int = int(c_size_float)
        n_size_int = int(n_size_float)
        num_inputs = numpy.int32(num_inputs_float)
        n_size = numpy.int32(n_size_float)
        c_size = numpy.int32(c_size_float)

        # Allocate arrays for network parameters
        d_m_weight = numpy.zeros(c_size, dtype=numpy.float32)
        d_m_source = numpy.zeros(c_size, dtype=numpy.float32)
        d_m_target = numpy.zeros(c_size, dtype=numpy.float32)
        d_m_a = numpy.zeros(n_size, dtype=numpy.float32)
        d_m_b = numpy.zeros(n_size, dtype=numpy.float32)

        # Create arrays for network parameters
        for i in range(len(net_params)):
            if i < c_size:
                d_m_weight[i] = net_params[i]
            elif i < c_size*2:
                d_m_source[i - c_size_int] = net_params[i]
            elif i < c_size*3:
                d_m_target[i - c_size_int*2] = net_params[i]
            elif i < (c_size*3 + n_size):
                d_m_a[i - c_size_int*3] = net_params[i]
            elif i < (c_size*3 + 2*n_size - 3):
                d_m_b[i - (c_size_int*3 + n_size_int)] = net_params[i]

        # Constants for grid/thread size
        grid = int(math.ceil(i_size/1024))
        threads_per_launch_malloc = int((grid * 1024)*c_size)
        grid_s = int(math.ceil(i_size/512))
        threads_per_launch_malloc_s = int((grid_s * 512)*c_size)

        # Create/allocate device vectors for network parameters
        dev_m_weight = pycuda.gpuarray.to_gpu(d_m_weight)
        dev_m_source = pycuda.gpuarray.to_gpu(d_m_source)
        dev_m_target = pycuda.gpuarray.to_gpu(d_m_target)
        dev_m_a = pycuda.gpuarray.to_gpu(d_m_a)
        dev_m_b = pycuda.gpuarray.to_gpu(d_m_b)
        signal_malloc = numpy.zeros(threads_per_launch_malloc_s)
        activesum_malloc = numpy.zeros(threads_per_launch_malloc_s)
        signal_malloc = signal_malloc.astype(numpy.float32)
        activesum_malloc = activesum_malloc.astype(numpy.float32)
        d_signal_malloc = pycuda.gpuarray.to_gpu(signal_malloc)
        d_activesum_malloc = pycuda.gpuarray.to_gpu(activesum_malloc)

        # Convert these to numpy.int32 for gpu
        i_size = numpy.int32(i_size)
        a_size = numpy.int32(a_size)

        # Execute the kernel
        self.gpu_exec(d_full_input,                         # Array of inputs
                      d_results,                            # Malloc for outputs
                      dev_m_source,                         # Source neuron connections
                      dev_m_weight,                         # Weights
                      dev_m_target,                         # Target neuron connections
                      dev_m_a,                              # Function parameter
                      dev_m_b,                              # Function parameter
                      c_size, n_size, i_size, num_inputs,   # Num connections, num neurons, num input, num_input (dep.)
                      d_signal_malloc,                      # Neuron signal malloc
                      d_activesum_malloc,                   # Neuron activesum malloc
                      a_size,                               # Number of input samples
                      block=(512, 1, 1), grid=(grid_s, 1)   # Block and grid size
                      )

        pycuda.driver.Context.synchronize()

        # Transfer results to host
        d_results.get(output)

        return output
