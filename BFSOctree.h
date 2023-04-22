#pragma once

#include <thrust/device_vector.h>

#include <memory>

#include "BFSInnerNode.h"
#include "BFSJob.h"
#include "Texture.h"
#include "float4x4.h"


class BFSOctree {
 public:
  BFSOctree();

  BFSOctree(char const* model, char const* diffuse, char const* illum,
            char const* spec, char const* normal);

  /* Updates the character animation */
  int update(double lastFrameTimeInMilliseconds);

  thrust::device_vector<BFSInnerNode> const& innerNodes() const;
  thrust::device_vector<VisualData> const& leaves() const;
  thrust::device_vector<BFSJob> const& jobs() const;
  thrust::device_vector<float4x4> const& animation() const;

  int level() const;
  float dim() const;
  int boneCount() const;

  Texture const& diffuse() const;
  Texture const& illum() const;
  Texture const& spec() const;
  Texture const& normal() const;

 private:
  thrust::device_vector<BFSInnerNode> m_innerNodes;
  thrust::device_vector<VisualData> m_leaves;
  thrust::device_vector<BFSJob> m_jobs;
  thrust::device_vector<float4x4> m_animation;

  unsigned char m_level;
  float m_dim;
  /* #frames of the character animation */
  int m_boneCount;
  std::uint32_t m_frameCount;
  double m_currentFrame;

  Texture m_diffuse;
  Texture m_illum;
  Texture m_spec;
  Texture m_normal;
};