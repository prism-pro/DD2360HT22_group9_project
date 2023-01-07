

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "backprop.h"
#include "omp.h"
#include <cuda.h>
#include <cuda_runtime.h>
extern char *strcpy();
extern void exit();

int layer_size = 0;

backprop_face()
{
  BPNN *net;
  int i;
  float out_err, hid_err;
  cudaMallocManaged((void**)&net,sizeof(BPNN), cudaMemAttachGlobal);
//   printf("%ld\n",sizeof(BPNN));
  net->input_n = layer_size;
  net->hidden_n = 16;
  net->output_n = 1;
  cudaMallocManaged((void**)&net->input_units,(layer_size+1)*sizeof(float), cudaMemAttachGlobal);
//   printf("%ld\n",sizeof(net->input_units));

  cudaMallocManaged((void**)&net->hidden_units,(16+1)*sizeof(float), cudaMemAttachGlobal);
//   printf("%ld\n",sizeof(net->hidden_units));

  cudaMallocManaged((void**)&net->output_units,(1+1)*sizeof(float), cudaMemAttachGlobal);
//   printf("%ld\n",sizeof(net->output_units));

  cudaMallocManaged((void**)&net->hidden_delta,(16+1)*sizeof(float), cudaMemAttachGlobal);
//   printf("%ld\n",sizeof(net->hidden_delta));

  cudaMallocManaged((void**)&net->output_delta,(1+1)*sizeof(float), cudaMemAttachGlobal);
//   printf("%ld\n",sizeof(net->output_delta));

  cudaMallocManaged((void**)&net->target,(1+1)*sizeof(float), cudaMemAttachGlobal);
//   printf("%ld\n",sizeof(net->target));

  cudaMallocManaged((void**)&net->input_weights,(layer_size+1)*sizeof(float), cudaMemAttachGlobal);
  for(int i=0;i<(layer_size+1);i++)
  {
    cudaMallocManaged((void**)&net->input_weights[i], (16+1)*sizeof(float), cudaMemAttachGlobal);
  }
//   printf("%ld\n",sizeof(net->input_weights));

  cudaMallocManaged((void**)&net->hidden_weights,(16+1)*sizeof(float), cudaMemAttachGlobal);
  for(int i=0;i<(16+1);i++)
  {
    cudaMallocManaged((void**)&net->hidden_weights[i], (1+1)*sizeof(float), cudaMemAttachGlobal);
  }
//   printf("%ld\n",sizeof(net->hidden_weights));
  
  cudaMallocManaged((void**)&net->input_prev_weights,(layer_size+1)*sizeof(float), cudaMemAttachGlobal);
  for(int i=0;i<(layer_size+1);i++)
  {
    cudaMallocManaged((void**)&net->input_prev_weights[i], (16+1)*sizeof(float), cudaMemAttachGlobal);
  }
//   printf("%ld\n",sizeof(net->input_prev_weights));
  cudaMallocManaged((void**)&net->hidden_prev_weights,(16+1)*sizeof(float), cudaMemAttachGlobal);
  for(int i=0;i<(16+1);i++)
  {
    cudaMallocManaged((void**)&net->hidden_prev_weights[i], (1+1)*sizeof(float), cudaMemAttachGlobal);
  }
//   printf("%ld\n", sizeof(net->hidden_prev_weights));
    // #ifdef INITZERO
    //   bpnn_zero_weights(net->input_weights, layer_size, 16);
    // #else
    //   bpnn_randomize_weights(net->input_weights, layer_size, 16);
//randomize input weights
    for (int i = 0; i <= 16; i++) {
     for (int j = 0; j <= 1; j++) {
      net->input_weights[i][j] = (float) rand()/RAND_MAX;
    //   printf("%f\n",net->input_weights[i][j]);

     }
    }

    // #endif
//randomize hidden weights 
    //   bpnn_randomize_weights(net->hidden_weights, 16, 1);
    for (int i = 0; i <= 16; i++) {
     for (int j = 0; j <= 1; j++) {
      net->hidden_weights[i][j] = (float) rand()/RAND_MAX;
    //   printf("hid%f\n",net->hidden_weights[i][j]);

     }
    }

//   bpnn_zero_weights(net->input_prev_weights, layer_size, 16);


//   bpnn_zero_weights(net->hidden_prev_weights, 16, 1);
       for (int i = 0; i <= 16; i++) {
        for (int j = 0; j <= 1; j++) {
          net->hidden_prev_weights[i][j] = 0.0;
        }
      }
//  bpnn_randomize_row(net->target, 1);

	for (i = 0; i <= 1; i++) {
     //w[i] = (float) rand()/RAND_MAX;
	 net->target[i] = 0.1;
    }
 
//   net = bpnn_create(layer_size, 16, 1); // (16, 1 can not be changed)
  
  //host memory space allocation
  printf("Input layer size : %d\n", layer_size);

 // load(net);
  //entering the training kernel, only one iteration

  printf("Starting training kernel\n");
  bpnn_train_cuda(net, &out_err, &hid_err);
  printf("Training done\n");
  cudaFree(net);
  cudaFree(net->input_units);
  cudaFree(net->hidden_units);
  cudaFree(net->output_units);
  cudaFree(net->hidden_delta);
  cudaFree(net->output_delta);
  cudaFree(net->target);
  for(int i=0;i<layer_size+1;i++)
  {
      cudaFree(net->input_weights[i]);
  }
  cudaFree(net->input_weights);
  for(int i=0;i<16+1;i++)
  {
      cudaFree(net->hidden_weights[i]);
  }
  cudaFree(net->hidden_weights);
  for(int i=0;i<layer_size+1;i++)
  {
      cudaFree(net->input_prev_weights[i]);
  }
  cudaFree(net->input_prev_weights);
  for(int i=0;i<16+1;i++)
  {
      cudaFree(net->hidden_prev_weights[i]);
  }
  cudaFree(net->hidden_prev_weights);
}

int setup(argc, argv)
int argc;
char *argv[];
{
	
  int seed;

  if (argc!=2){
  fprintf(stderr, "usage: backprop <num of input elements>\n");
  exit(0);
  }
  layer_size = atoi(argv[1]);
  if (layer_size%16!=0){
  fprintf(stderr, "The number of input points must be divided by 16\n");
  exit(0);
  }
  

  seed = 7;   
  srand(seed);    

  backprop_face();

  exit(0);
}
