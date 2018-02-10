/*
 * Copyright 2018, Nod Labs
 */
#define EIGEN_USE_THREADS
#define EIGEN_USE_CUSTOM_THREAD_POOL

#include <iostream>

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/compiler/aot/tests/test_graph_tfmatmul.h"
#include "tensorflow/compiler/aot/tests/test_graph_tfmatmulandadd.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/grappler/devices.h"

namespace tensorflow {
namespace tfcompile {
namespace {
TEST(TFCompileTest, NodMatMul) {
  Eigen::ThreadPool tp(port::NumSchedulableCPUs());
  Eigen::ThreadPoolDevice device(&tp, tp.NumThreads());

  foo::bar::MatMulComp matmul;
  matmul.set_thread_pool(&device);
  EXPECT_EQ(matmul.arg0_data(), matmul.args()[0]);
  EXPECT_EQ(matmul.arg1_data(), matmul.args()[1]);

  {
    matmul.arg0(0, 0) = 1;
    matmul.arg0(0, 1) = 2;
    matmul.arg0(0, 2) = 3;
    matmul.arg0(1, 0) = 4;
    matmul.arg0(1, 1) = 5;
    matmul.arg0(1, 2) = 6;

    matmul.arg1(0, 0) = 7;
    matmul.arg1(0, 1) = 8;
    matmul.arg1(1, 0) = 9;
    matmul.arg1(1, 1) = 10;
    matmul.arg1(2, 0) = 11;
    matmul.arg1(2, 1) = 12;

    EXPECT_TRUE(matmul.Run());
    EXPECT_EQ(matmul.error_msg(), "");

    const float results[4] = {58, 64, 139, 154};
    for (int i = 0; i < 4; i++) {
      EXPECT_EQ(matmul.result0(i / 2, i % 2), results[i]);
      EXPECT_EQ(matmul.result0_data()[i], results[i]);
    }
    EXPECT_EQ(matmul.result0_data(), matmul.results()[0]);
  }

  int32_t num_gpus = tensorflow::grappler::GetNumAvailableGPUs();
  int64_t available_memory = tensorflow::grappler::AvailableGPUMemory(num_gpus - 1);  // assume 1 GPU for now
  int32_t logical_cpus = tensorflow::grappler::GetNumAvailableLogicalCPUCores();

  std::cout << "NumGPUs() = " << num_gpus << std::endl;
  std::cout << "Available GPU memory = " << available_memory / 1024 / 1024 / 1024 << " GB" << std::endl;
  std::cout << "Available CPU cores = " << logical_cpus << std::endl;
}
}}}
