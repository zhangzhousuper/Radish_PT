#include "bvh.h"
#include <vcruntime_string.h>

/**
 * MTBVH builder
 https://cs.uwaterloo.ca/~thachisu/tdf2015.pdf
 */

int BVHBuilder::build(const std::vector<glm::vec3> &vertices,
                      std::vector<AABB> &boundingBoxes,
                      std::vector<std::vector<MTBVHNode>> &BVHNodes) {
  std::cout << "[BVH building...]" << std::endl;
  int numPrimitives = vertices.size() / 3;
  int BVHSize = numPrimitives * 2 - 1;

  std::vector<PrimInfo> primInfo(numPrimitives);
  std::vector<NodeInfo> nodeInfo(BVHSize);
  boundingBoxes.resize(BVHSize);

  for (int i = 0; i < numPrimitives; i++) {
    primInfo[i].primId = i;
    primInfo[i].bound =
        AABB(vertices[i * 3], vertices[i * 3 + 1], vertices[i * 3 + 2]);
    primInfo[i].center = primInfo[i].bound.center();
  }

  // array stack for faster
  std::vector<BuildInfo> stack(BVHSize);
  int stackTop = 0;
  stack[stackTop++] = {0, 0, numPrimitives - 1};

  const int NumBuckets = 16;

  // using non-recursive approach to build BVH directly flatteded
  int depth = 0;
  while (stackTop) {
    depth = std::max(depth, stackTop);
    stackTop--;
    int offset = stack[stackTop].offset;
    int start = stack[stackTop].start;
    int end = stack[stackTop].end;

    int numSubPrimitives = end - start + 1;
    int nodeSize = numSubPrimitives * 2 - 1;
    bool isLeaf = numSubPrimitives == 1;
    nodeInfo[offset] = {isLeaf, isLeaf ? primInfo[start].primId : nodeSize};

    AABB nodeBound, centerBound;
    for (int i = start; i <= end; i++) {
      nodeBound = nodeBound(primInfo[i].bound);
      centerBound = centerBound(primInfo[i].center);
    }
    boundingBoxes[offset] = nodeBound;

    std::cout << std::setw(10) << offset << " " << start << " " << end << " "
              << nodeBound.toString() << "\n";

    if (isLeaf) {
      continue;
    }

    int splitAxis = centerBound.longestAxis();

    if (nodeSize == 2) {
      if (primInfo[start].center[splitAxis] > primInfo[end].center[splitAxis]) {
        std::swap(primInfo[start], primInfo[end]);
      }
      boundingBoxes[offset + 1] = primInfo[start].bound;
      boundingBoxes[offset + 2] = primInfo[end].bound;
      nodeInfo[offset + 1] = {true, primInfo[start].primId};
      nodeInfo[offset + 2] = {true, primInfo[end].primId};
    }

    AABB bucketBounds[NumBuckets];
    int bucketCounts[NumBuckets];
    memset(bucketCounts, 0, sizeof(bucketCounts));

    float dimMin = centerBound.pMin[splitAxis];
    float dimMax = centerBound.pMax[splitAxis];

    for (int i = start; i <= end; i++) {
      int bid = glm::clamp(int((primInfo[i].center[splitAxis] - dimMin) /
                               (dimMax - dimMin) * NumBuckets),
                           0, NumBuckets - 1);
      bucketBounds[bid] = bucketBounds[bid](primInfo[i].bound);
      bucketCounts[bid]++;
    }

    AABB leftBounds[NumBuckets];
    AABB rightBounds[NumBuckets];
    int countPrefix[NumBuckets];

    leftBounds[0] = bucketBounds[0];
    rightBounds[NumBuckets - 1] = bucketBounds[NumBuckets - 1];
    countPrefix[0] = bucketCounts[0];
    for (int i = 1, j = NumBuckets - 2; i < NumBuckets; i++, j--) {
      leftBounds[i] = leftBounds[i](bucketBounds[i - 1]);
      rightBounds[j] = rightBounds[j](bucketBounds[j + 1]);
      countPrefix[i] = countPrefix[i - 1] + bucketCounts[i];
    }

    float minSAH = FLT_MAX;
    int divBucket = 0;
    for (int i = 0; i < NumBuckets - 1; i++) {
      float SAH = glm::mix(leftBounds[i].surfaceArea(),
                           rightBounds[i + 1].surfaceArea(),
                           countPrefix[i] / float(numSubPrimitives));
      if (SAH < minSAH) {
        minSAH = SAH;
        divBucket = i;
      }
    }

    std::vector<PrimInfo> temp(numSubPrimitives);
    memcpy(temp.data(), primInfo.data() + start,
           sizeof(PrimInfo) * numSubPrimitives);

    int divPrim = start, divEnd = end;
    for (int i = 0; i < numSubPrimitives; i++) {
      int bid = glm::clamp(int((temp[i].center[splitAxis] - dimMin) /
                               (dimMax - dimMin) * NumBuckets),
                           0, NumBuckets - 1);
      (bid <= divBucket ? primInfo[divPrim++] : primInfo[divEnd--]) = temp[i];
    }

    divPrim = glm::clamp(divPrim, start, end - 1);
    int leftSize = 2 * (divPrim - start + 1) - 1;

    stack[stackTop++] = {offset + 1 + leftSize, divPrim + 1, end};
    stack[stackTop++] = {offset + 1, start, divPrim};
  }

  std::cout << "\t[Size = " << BVHSize << ", depth = " << depth << "]"
            << std::endl;
  buildMTBVH(boundingBoxes, nodeInfo, BVHSize, BVHNodes);
  return BVHSize;
}

void BVHBuilder::buildMTBVH(const std::vector<AABB> &boundingBoxes,
                            const std::vector<NodeInfo> &nodeInfo, int BVHSize,
                            std::vector<std::vector<MTBVHNode>> &BVHNodes) {
  BVHNodes.resize(6);
  for (auto &node : BVHNodes) {
    node.resize(BVHSize);
  }

  std::vector<int> stack(BVHSize);

  for (int i = 0; i < 6; i++) {
    auto &nodes = BVHNodes[i];
    nodes.resize(BVHSize);

    int stackTop = 0;
    stack[stackTop++] = 0;
    int nodeIdNew = 0;

    while (stackTop) {
      int nodeIdOrig = stack[--stackTop];
      bool isLeaf = nodeInfo[nodeIdOrig].isLeaf;
      int nodeSize = isLeaf ? 1 : nodeInfo[nodeIdOrig].primIdOrSize;

      nodes[nodeIdNew] = {isLeaf ? nodeInfo[nodeIdOrig].primIdOrSize
                                 : NullPrimitive,
                          nodeIdOrig, nodeSize + nodeIdNew};
      nodeIdNew++;

      if (isLeaf) {
        continue;
      }

      bool isLeftLeaf = nodeInfo[nodeIdOrig + 1].isLeaf;
      int leftSize = isLeftLeaf ? 1 : nodeInfo[nodeIdOrig + 1].primIdOrSize;

      int left = nodeIdOrig + 1;
      int right = nodeIdOrig + 1 + leftSize;

      int dim = i / 2;
      bool lesser = dim & 1;

      if (boundingBoxes[left].center()[dim] <
              boundingBoxes[right].center()[dim] ^
          lesser) {
        std::swap(left, right);
      }

      stack[stackTop++] = right;
      stack[stackTop++] = left;
    }
  }
  for (const auto &nodes : BVHNodes) {
    for (const auto &node : nodes) {
      std::cout << std::setw(3) << node.primitiveId << " ";
    }
    std::cout << "\n";
    for (const auto &node : nodes) {
      std::cout << std::setw(3) << node.nextNodeIfMiss << " ";
    }
    std::cout << "\n\n";
  }
}