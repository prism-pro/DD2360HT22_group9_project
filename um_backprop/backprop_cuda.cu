// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cuda.h>
#include <sys/time.h>

// includes, kernels
#include "backprop_cuda_kernel.cu"
#include "backprop.h"

double cpuSecond() {
   struct timeval tp;
   gettimeofday(&tp,NULL);
   return ((double)tp.tv_sec + (double)tp.tv_usec*1.e-6);
}
extern "C"
void bpnn_layerforward(float *l1, float *l2, float **conn, int n1, int n2);

extern "C"
void bpnn_output_error(float *delta, float *target, float *output, int nj, float *err);

extern "C"
void bpnn_hidden_error(float *delta_h, int nh, float *delta_o, int no, float **who, float *hidden, float *err);

extern "C" 
void bpnn_adjust_weights(float *delta, int ndelta, float *ly, int nly, float **w, float **oldw);

extern "C"
int setup(int argc, char** argv);


double gettime() {
  struct timeval t;
  gettimeofday(&t,NULL);
  return t.tv_sec+t.tv_usec*1e-6;
}

unsigned int num_threads = 0;
unsigned int num_blocks = 0;

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int
main( int argc, char** argv) 
{
	setup(argc, argv);
}


extern "C"
void bpnn_train_cuda(BPNN *net, float *eo, float *eh)
{
  int in, hid, out;
  float out_err, hid_err;
  
  in = net->input_n;
  hid = net->hidden_n;
  out = net->output_n;   
  int m = 0;
  float *input_hidden_cuda;
  float *input_cuda;
  float *output_hidden_cuda;
  float *partial_sum;
  float *hidden_partial_sum;
  float *hidden_delta_cuda;
  float *input_prev_weights_cuda;
  float sum;
  float *input_weights_one_dim;
  float *input_weights_prev_one_dim;
  num_blocks = in / 16;  
  dim3  grid( 1 , num_blocks);
  dim3  threads(16 , 16);
  partial_sum = (float *) malloc(num_blocks * WIDTH * sizeof(float));
  cudaMallocManaged((void**)&input_weights_one_dim, (in + 1)* (hid + 1)* sizeof(float),cudaMemAttachGlobal);
  cudaMallocManaged((void**)&input_weights_prev_one_dim,(in + 1)* (hid + 1) * sizeof(float),cudaMemAttachGlobal);

  // this preprocessing stage is added to correct the bugs of wrong memcopy using two-dimensional net->inputweights
//  for (int k = 0; k <= in; k++) {	
  // for (int j = 0; j <= hid; j++) {
	  //input_weights_one_dim[m] = net->input_weights[k][j];
	  //input_weights_prev_one_dim[m] = net-> input_prev_weights[k][j];
	//  m++;
//    }
 // }
//directly assign value to the input_weights_one_dim since problems emerge when copy;
  for(int m=0;m<=(in+1)*(hid+1);m++)
  {
    input_weights_one_dim[m]=(float) rand()/RAND_MAX;
    input_weights_prev_one_dim[m]=0;
  }

  // this preprocessing stage is added to correct the bugs of wrong memcopy using two-dimensional net->inputweights
  
  cudaMallocManaged((void**)&output_hidden_cuda, (hid + 1) * sizeof(float),cudaMemAttachGlobal);
  cudaMallocManaged((void**)&hidden_partial_sum, num_blocks * WIDTH * sizeof(float),cudaMemAttachGlobal);

  printf("Performing GPU computation\n");
   double fwdiStart = cpuSecond();

  //printf("in= %d, hid = %d, numblocks = %d\n", in, hid, num_blocks);

   bpnn_layerforward_CUDA<<< grid, threads >>>(net->input_units,
 	                                          output_hidden_cuda,
 											  input_weights_one_dim,
 											  hidden_partial_sum,
 											  in,
 											  hid);

  cudaThreadSynchronize();
  double fwdiElaps = cpuSecond() - fwdiStart;
  printf("The forward time for gpu computation : %f ;\n",fwdiElaps);
  cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess) {
		printf("bpnn kernel error: %s\n", cudaGetErrorString(error));
		exit(EXIT_FAILURE);
	}
  
     
  for (int j = 1; j <= hid; j++) {
    sum = 0.0;
    for (int k = 0; k < num_blocks; k++) {	
      sum += partial_sum[k * hid + j-1] ;
    }
	sum += net->input_weights[0][j];
	net-> hidden_units[j] = float(1.0 / (1.0 + exp(-sum)));
  }


  bpnn_layerforward(net->hidden_units, net->output_units, net->hidden_weights, hid, out);
  bpnn_output_error(net->output_delta, net->target, net->output_units, out, &out_err);
  bpnn_hidden_error(net->hidden_delta, hid, net->output_delta, out, net->hidden_weights, net->hidden_units, &hid_err);  
  bpnn_adjust_weights(net->output_delta, out, net->hidden_units, hid, net->hidden_weights, net->hidden_prev_weights);



  cudaMallocManaged((void**) &input_prev_weights_cuda, (in + 1) * (hid + 1) * sizeof(float),cudaMemAttachGlobal);
     double bwdiStart = cpuSecond();

  bpnn_adjust_weights_cuda<<< grid, threads >>>(net->hidden_delta,  
												hid, 
												net->input_units, 
												in,
												input_weights_one_dim, 
												input_weights_prev_one_dim
												);
  cudaThreadSynchronize();                
  double bwdiElaps = cpuSecond() - bwdiStart;
  printf("The backward time for gpu computation : %f ;\n",bwdiElaps);
 // cudaFree(input_cuda);
 // cudaFree(output_hidden_cuda);
 // cudaFree(input_hidden_cuda);
 // cudaFree(hidden_partial_sum);
  //cudaFree(input_prev_weights_cuda);
  //cudaFree(hidden_delta_cuda);
  
  //free(partial_sum);
  cudaFree(input_weights_one_dim);
  cudaFree(input_weights_prev_one_dim);
  cudaFree(hidden_partial_sum);
  cudaFree(output_hidden_cuda);
  cudaFree(input_prev_weights_cuda);
}
