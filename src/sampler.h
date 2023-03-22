#pragma once

#include "cudaUtil.h"
#include "driver_types.h"
#include "mathUtil.h"
#include <thrust/random.h>
#include <vcruntime.h>

#define SAMPLER_USE_SOBOL false

#if SAMPLER_USE_SOBOL
#else
using Sampler = thrust::default_random_engine;
#endif

__host__ __device__ static thrust::default_random_engine
makeSeededRandomEngine(int iter, int index, int depth) {
  int h =
      Math::utilhash((1 << 31) | (depth << 22) | iter) ^ Math::utilhash(index);
  return thrust::default_random_engine(h);
}

__device__ inline float sample1D(thrust::default_random_engine &rng) {
  return thrust::uniform_real_distribution<float>(0.f, 1.f)(rng);
}

__device__ inline glm::vec2 sample2D(Sampler &sampler) {
  return glm::vec2(sample1D(sampler), sample1D(sampler));
}

__device__ inline glm::vec3 sample3D(Sampler &sampler) {
  return glm::vec3(sample2D(sampler), sample1D(sampler));
}

__device__ inline glm::vec4 sample4D(Sampler &sampler) {
  return glm::vec4(sample3D(sampler), sample1D(sampler));
}

template <typename T> struct BinomialDistrib {
  T prob;
  int failId;
};

/**
 * Transform a discrete distribution to a set of binomial distributions
 *   so that an O(1) sampling approach can be applied
 */

template <typename T> struct DiscreteSampler1D {
  using DistribT = BinomialDistrib<T>;

  DiscreteSampler1D() = default;

  DiscreteSampler1D(std::vector<T> values) {

    for (const auto &val : values) {
      sum += val;
    }
    T sumInv = static_cast<float>(values.size()) / sum;

    for (auto &val : values) {
      val *= sumInv;
    }

    binomDistribs.resize(values.size());
    std::vector<DistribT> stackGtOne(values.size() * 2);
    std::vector<DistribT> stackLsOne(values.size() * 2);
    int topGtOne = 0;
    int topLsOne = 0;

    for (int i = 0; i < values.size(); ++i) {
      auto &val = values[i];
      (val > static_cast<T>(1) ? stackGtOne[topGtOne++]
                               : stackLsOne[topLsOne++]) = DistribT{val, i};
    }

    while (topGtOne && topLsOne) {
      DistribT gtOne = stackGtOne[--topGtOne];
      DistribT lsOne = stackLsOne[--topLsOne];
      binomDistribs[lsOne.failId] = DistribT{lsOne.prob, gtOne.failId};
      // Place ls in the table, and "fill" the rest of probability with gt.prob
      gtOne.prob -= (static_cast<T>(1) - lsOne.prob);
      // See if gt.prob is still greater than 1 that it needs more iterations to
      //   be splitted to different binomial distributions
      (gtOne.prob > static_cast<T>(1) ? stackGtOne[topGtOne++]
                                      : stackLsOne[topLsOne++]) = gtOne;
    }

    for (int i = topGtOne - 1; i >= 0; --i) {
      DistribT gtOne = stackGtOne[i];
      binomDistribs[gtOne.failId] = gtOne;
    }

    for (int i = topLsOne - 1; i >= 0; --i) {
      DistribT lsOne = stackLsOne[i];
      binomDistribs[lsOne.failId] = lsOne;
    }
  }

  int sample(float r1, float r2) {
    int passId = int(float(binomDistribs.size()) * r1);
    DistribT distrib = binomDistribs[passId];
    return (r2 < distrib.prob) ? passId : distrib.failId;
  }
  T sum = static_cast<T>(0);
  std::vector<DistribT> binomDistribs;
};

template <typename T> struct DiscreteSampler2D {
  using DistribT = BinomialDistrib<T>;

  DiscreteSampler2D() = default;

  DiscreteSampler2D(const T *data, int width, int height) {
    columnSamplers.resize(height);
    std::vector<T> sumRows(height);
    std::vector<T> rowData(width);

    for (int i = 0; i < height; i++) {
      for (int j = 0; j < width; j++) {
        sumRows[i] += data[i * width + j];
      }
      float sumRowInv = static_cast<T>(1) / sumRows[i];

      for (int j = 0; j < width; j++) {
        rowData[j] = data[i * width + j] * sumRowInv;
      }
      columnSamplers[i] = DiscreteSampler1D<T>(rowData);
      sum += sumRows[i];
    }

    T sumInv = static_cast<T>(1) / sum;
    for (int i = 0; i < height; i++) {
      sumRows[i] *= sumInv;
    }
    rowSampler = DiscreteSampler1D<T>(sumRows);
  }

  std::pair<int, int> sample(float r1, float r2, float r3, float r4) {
    int rowId = rowSampler.sample(r1, r2);
    int colId = columnSamplers[rowId].sample(r3, r4);
    return {rowId, colId};
  }

  std::vector<DiscreteSampler1D<T>> columnSamplers;
  DiscreteSampler1D<T> rowSampler;
  T sum = static_cast<T>(0);
};

template <typename T> struct DevDiscreteSampler1D {
  using DistribT = BinomialDistrib<T>;

  void create(const DiscreteSampler1D<T> &hstSampler) {
    size_t size = byteSizeOf<DistribT>(hstSampler.binomDistribs);
    cudaMalloc(&devBinomDistribs, size);
    cudaMemcpyHostToDev(devBinomDistribs, hstSampler.binomDistribs.data(),
                        size);
    length = hstSampler.binomDistribs.size();
  }

  void destroy() {
    cudaSafeFree(devBinomDistribs);
    length = 0;
  }

  __device__ int sample(float r1, float r2) {
    int passId = glm::min(int(float(length) * r1), length - 1);
    DistribT distrib = devBinomDistribs[passId];
    return (r2 < distrib.prob) ? passId : distrib.failId;
  }

  DistribT *devBinomDistribs = nullptr;
  int length = 0;
};

template <typename T> struct DevDiscreteSampler2D {
  using DistribT = BinomialDistrib<T>;

  void create(const std::vector<DiscreteSampler1D<T>> &hstSamplers) {
    width = hstSamplers[0].size();
    height = hstSamplers.size();
  }

  T *devBinomDistribs = nullptr;
  int width = 0;
  int height = 0;
};
