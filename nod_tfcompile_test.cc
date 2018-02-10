/*
 * Copyright 2018, Nod Labs
 */
#include <iostream>

#define EIGEN_USE_THREADS
#define EIGEN_USE_CUSTOM_THREAD_POOL
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/platform/test.h"

#include "out.h"

using namespace tensorflow;
using namespace testgraph;

int main(int argc, char **argv) {
  Eigen::ThreadPool tp(2);
  Eigen::ThreadpoolDevice device(&tp, tp.NumThreads());
  MatMul matmul;

  matmul.set_thread_pool(&device);
  matmul.arg0(0, 0) = 1;
  matmul.arg0(0, 1) = 2;
  matmul.arg0(0, 2) = 3;
  matmul.arg0(1, 0) = 4;
  matmul.arg0(1, 1) = 5;
  matmul.arg0(1, 2) = 6;

  matmul.arg1(0 ,0) = 7;
  matmul.arg1(0, 1) = 8;
  matmul.arg1(1, 0) = 9;
  matmul.arg1(1, 1) = 10;
  matmul.arg1(2, 0) = 11;
  matmul.arg1(2, 1) = 12;

  if (matmul.Run() != true) {
    std::cout << "Run failed" << std::endl;
    return -1;
  }

  if (matmul.error_msg() != "") {
    std::cout << "Failed to execute the MatMul graph" << std::endl;
    return -1;
  }

  for (int i = 0; i < 4; i++) {
    std::cout << matmul.result0(i / 2, i % 2) << " ";
  }

  std::cout << std::endl;

  return 0;
}
