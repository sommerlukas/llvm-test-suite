// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: env SYCL_ENABLE_FUSION_CACHING=0 SYCL_RT_WARNING_LEVEL=1 %CPU_RUN_PLACEHOLDER %t.out 2>&1\
// RUN: %CPU_CHECK_PLACEHOLDER
// RUN: env SYCL_ENABLE_FUSION_CACHING=0 SYCL_RT_WARNING_LEVEL=1 %GPU_RUN_PLACEHOLDER %t.out 2>&1\
// RUN: %GPU_CHECK_PLACEHOLDER
// UNSUPPORTED: cuda || hip
// REQUIRES: fusion

// Test incomplete internalization: Different scenarios causing the JIT compiler
// to abort internalization due to target or parameter mismatch. Also check that
// warnings are printed when SYCL_RT_WARNING_LEVEL=1.

#include <sycl/sycl.hpp>

using namespace sycl;

constexpr size_t dataSize = 512;

enum class Internalization { None, Local, Private };

void performFusion(queue &q, int *in1, int *in2, int *in3, int *tmp, int *out,
                   Internalization intKernel1, size_t localSizeKernel1,
                   Internalization intKernel2, size_t localSizeKernel2,
                   bool expectInternalization = false) {
  {
    buffer<int> bIn1{in1, range{dataSize}};
    buffer<int> bIn2{in2, range{dataSize}};
    buffer<int> bIn3{in3, range{dataSize}};
    buffer<int> bTmp{tmp, range{dataSize}};
    buffer<int> bOut{out, range{dataSize}};

    ext::codeplay::experimental::fusion_wrapper fw{q};
    fw.start_fusion();

    assert(fw.is_in_fusion_mode() && "Queue should be in fusion mode");

    q.submit([&](handler &cgh) {
      auto accIn1 = bIn1.get_access(cgh);
      auto accIn2 = bIn2.get_access(cgh);
      accessor<int> accTmp =
          (intKernel1 == Internalization::Private)
              ? bTmp.get_access(cgh, sycl::ext::codeplay::experimental::
                                         property::promote_private{})
              : (intKernel1 == Internalization::Local)
                    ? bTmp.get_access(cgh, sycl::ext::codeplay::experimental::
                                               property::promote_local{})
                    : bTmp.get_access(cgh);
      if (localSizeKernel1 > 0) {
        cgh.parallel_for<class Kernel1>(
            nd_range<1>{{dataSize}, {localSizeKernel1}},
            [=](id<1> i) { accTmp[i] = accIn1[i] + accIn2[i]; });
      } else {
        cgh.parallel_for<class KernelOne>(
            dataSize, [=](id<1> i) { accTmp[i] = accIn1[i] + accIn2[i]; });
      }
    });

    q.submit([&](handler &cgh) {
      accessor<int> accTmp =
          (intKernel2 == Internalization::Private)
              ? bTmp.get_access(cgh, sycl::ext::codeplay::experimental::
                                         property::promote_private{})
              : (intKernel2 == Internalization::Local)
                    ? bTmp.get_access(cgh, sycl::ext::codeplay::experimental::
                                               property::promote_local{})
                    : bTmp.get_access(cgh);
      auto accIn3 = bIn3.get_access(cgh);
      auto accOut = bOut.get_access(cgh);
      if (localSizeKernel2 > 0) {
        cgh.parallel_for<class Kernel2>(
            nd_range<1>{{dataSize}, {localSizeKernel2}},
            [=](id<1> i) { accOut[i] = accTmp[i] * accIn3[i]; });
      } else {
        cgh.parallel_for<class KernelTwo>(
            dataSize, [=](id<1> i) { accOut[i] = accTmp[i] * accIn3[i]; });
      }
    });

    fw.complete_fusion({ext::codeplay::experimental::property::no_barriers{}});

    assert(!fw.is_in_fusion_mode() &&
           "Queue should not be in fusion mode anymore");
  }

  // Check the results
  size_t numErrors = 0;
  size_t numInternalized = 0;
  for (size_t i = 0; i < dataSize; ++i) {
    if (out[i] != (20 * i * i)) {
      ++numErrors;
    }
    if (tmp[i] == -1) {
      ++numInternalized;
    }
    tmp[i] = -1;
    out[i] = -1;
  }
  if (numErrors) {
    std::cout << "COMPUTATION ERROR\n";
    return;
  }
  if (!expectInternalization && numInternalized) {
    std::cout << "WRONG INTERNALIZATION\n";
    return;
  }
  std::cout << "COMPUTATION OK\n";
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

  // Scenario: One accessor without internalization, one with local
  // internalization. Should fall back to no internalization and print a
  // warning.
  std::cout << "None, Local(0)\n";
  performFusion(q, in1, in2, in3, tmp, out, Internalization::None, 0,
                Internalization::Local, 0);
  // Scenario: One accessor without internalization, one with private
  // internalization. Should fall back to no internalization and print a
  // warning.
  std::cout << "None, Private\n";
  performFusion(q, in1, in2, in3, tmp, out, Internalization::None, 0,
                Internalization::Private, 0);

  // Scenario: Both accessor with local promotion, but the second kernel does
  // not specify a work-group size. No promotion should happen and a warning
  // should be printed.
  std::cout << "Local(8), Local(0)\n";
  performFusion(q, in1, in2, in3, tmp, out, Internalization::Local, 8,
                Internalization::Local, 0);

  // Scenario: Both accessor with local promotion, but the first kernel does
  // not specify a work-group size. No promotion should happen and a warning
  // should be printed.
  std::cout << "Local(0), Local(8)\n";
  performFusion(q, in1, in2, in3, tmp, out, Internalization::Local, 0,
                Internalization::Local, 8);

  // Scenario: Both accessor with local promotion, but the kernels specify
  // different work-group sizes. No promotion should happen and a warning should
  // be printed.
  std::cout << "Local(8), Local(16)\n";
  performFusion(q, in1, in2, in3, tmp, out, Internalization::Local, 8,
                Internalization::Local, 16);

  // Scenario: One accessor with local internalization, one with private
  // internalization. Should fall back to local internalization and print a
  // warning.
  std::cout << "Local(8), Private(8)\n";
  performFusion(q, in1, in2, in3, tmp, out, Internalization::Local, 8,
                Internalization::Private, 8, /* expectInternalization */ true);

  return 0;
}

// CHECK: None, Local(0)
// CHECK-NEXT: WARNING: Not performing specified local promotion, due to previous mismatch or because previous accessor specified no promotion
// CHECK-NEXT: COMPUTATION OK
// CHECK-NEXT: None, Private
// CHECK-NEXT: WARNING: Not performing specified private promotion, due to previous mismatch or because previous accessor specified no promotion
// CHECK-NEXT: COMPUTATION OK
// CHECK-NEXT: Local(8), Local(0)
// CHECK-NEXT: WARNING: Work-group size for local promotion not specified, not performing internalization
// CHECK-NEXT: COMPUTATION OK
// CHECK-NEXT: Local(0), Local(8)
// CHECK-NEXT: WARNING: Work-group size for local promotion not specified, not performing internalization
// CHECK-NEXT: WARNING: Not performing specified local promotion, due to previous mismatch or because previous accessor specified no promotion
// CHECK-NEXT: WARNING: Cannot fuse kernels with different local size
// CHECK-NEXT: COMPUTATION OK
// CHECK-NEXT: Local(8), Local(16)
// CHECK-NEXT: WARNING: Not performing specified local promotion due to work-group size mismatch
// CHECK-NEXT: WARNING: Cannot fuse kernels with different local size
// CHECK-NEXT: COMPUTATION OK
// CHECK-NEXT: Local(8), Private(8)
// CHECK-NEXT: WARNING: Performing local internalization instead, because previous accessor specified local promotion
// CHECK-NEXT: COMPUTATION OK
