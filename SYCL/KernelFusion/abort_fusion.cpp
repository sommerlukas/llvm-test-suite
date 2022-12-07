// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: env SYCL_RT_WARNING_LEVEL=1 %CPU_RUN_PLACEHOLDER %t.out 2>&1\
// RUN: %CPU_CHECK_PLACEHOLDER
// RUN: env SYCL_RT_WARNING_LEVEL=1 %GPU_RUN_PLACEHOLDER %t.out 2>&1\
// RUN: %GPU_CHECK_PLACEHOLDER
// UNSUPPORTED: cuda || hip
// REQUIRES: fusion

// Test fusion being aborted: Different scenarios causing the JIT compiler
// to abort fusion due to constraint violations for fusion. Also check that
// warnings are printed when SYCL_RT_WARNING_LEVEL=1.

#include <sycl/sycl.hpp>

using namespace sycl;

constexpr size_t dataSize = 512;

enum class Internalization { None, Local, Private };

template <typename Kernel1Name, typename Kernel2Name, int Kernel1Dim>
void performFusion(queue &q, int *in1, int *in2, int *in3, int *tmp, int *out,
                   range<Kernel1Dim> k1Global, range<Kernel1Dim> k1Local) {
  {
    buffer<int> bIn1{in1, range{dataSize}};
    buffer<int> bIn2{in2, range{dataSize}};
    buffer<int> bIn3{in3, range{dataSize}};
    buffer<int> bTmp{tmp, range{dataSize}};
    buffer<int> bOut{out, range{dataSize}};

    ext::codeplay::experimental::fusion_wrapper fw(q);
    fw.start_fusion();

    assert(fw.is_in_fusion_mode() && "Queue should be in fusion mode");

    q.submit([&](handler &cgh) {
      auto accIn1 = bIn1.get_access(cgh);
      auto accIn2 = bIn2.get_access(cgh);
      auto accTmp = bTmp.get_access(cgh);
      cgh.parallel_for<Kernel1Name>(
          nd_range<Kernel1Dim>{k1Global, k1Local}, [=](item<Kernel1Dim> i) {
            accTmp[i.get_linear_id()] =
                accIn1[i.get_linear_id()] + accIn2[i.get_linear_id()];
          });
    });

    q.submit([&](handler &cgh) {
      auto accTmp = bTmp.get_access(cgh);
      auto accIn3 = bIn3.get_access(cgh);
      auto accOut = bOut.get_access(cgh);
      cgh.parallel_for<Kernel2Name>(nd_range<1>{{dataSize}, {8}}, [=](id<1> i) {
        accOut[i] = accTmp[i] * accIn3[i];
      });
    });

    fw.complete_fusion({ext::codeplay::experimental::property::no_barriers{}});

    assert(!fw.is_in_fusion_mode() &&
           "Queue should not be in fusion mode anymore");
  }

  // Check the results
  size_t numErrors = 0;
  for (size_t i = 0; i < k1Global.size(); ++i) {
    if (out[i] != (20 * i * i)) {
      ++numErrors;
    }
    tmp[i] = -1;
    out[i] = -1;
  }
  if (numErrors) {
    std::cout << "COMPUTATION ERROR\n";
  } else {
    std::cout << "COMPUTATION OK\n";
  }
}

int main() {
  int in1[dataSize], in2[dataSize], in3[dataSize], tmp[dataSize], out[dataSize];

  for (size_t i = 0; i < dataSize; ++i) {
    in1[i] = i * 2;
    in2[i] = i * 3;
    in3[i] = i * 4;
    tmp[i] = -1;
    out[i] = -1;
  }

  queue q{ext::codeplay::experimental::property::queue::enable_fusion{}};

  // Scenario: Fusing two kernels with different dimensionality should lead to
  // fusion being aborted.
  performFusion<class Kernel1_1, class Kernel2_1>(
      q, in1, in2, in3, tmp, out, range<2>{32, 16}, range<2>{1, 8});

  // Scenario: Fusing two kernels with different global size should lead to
  // fusion being aborted.
  performFusion<class Kernel1_2, class Kernel2_2>(q, in1, in2, in3, tmp, out,
                                                  range<1>{256}, range<1>{8});

  // Scenario: Fusing two kernels with different local size should lead to
  // fusion being aborted.
  performFusion<class Kernel1_3, class Kernel2_3>(
      q, in1, in2, in3, tmp, out, range<1>{dataSize}, range<1>{16});

  return 0;
}

// CHECK: WARNING: Cannot fuse kernels with different dimensionality
// CHECK-NEXT: COMPUTATION OK
// CHECK-NEXT: WARNING: Cannot fuse kerneles with different global size
// CHECK-NEXT: COMPUTATION OK
// CHECK-NEXT: WARNING: Cannot fuse kernels with different local size
// CHECK-NEXT: COMPUTATION OK
