#pragma once

#include <cstdint>

#include "VisualData.h"

/**
 * Represents an inner node of a BFSOctree.
 */
class BFSInnerNode {
 public:
  VisualData vd;
  std::uint32_t mask;
  std::uint32_t childPtr;

  void deserialize(FILE* inFile);
};