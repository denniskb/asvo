#include "BFSOctree.h"

#include <thrust/host_vector.h>

#include <cassert>

#include "Glue.h"
#include "float4x4.h"

BFSOctree::BFSOctree() {}

BFSOctree::BFSOctree(char const* model, char const* diffuse, char const* illum,
                     char const* spec, char const* normal)
    :  // TODO: Replace hard-coded values
      m_diffuse(diffuse, 1024, 1024),
      m_illum(illum, 1024, 1024),
      m_spec(spec, 1024, 1024),
      m_normal(normal, 1024, 1024) {
  FILE* file = fopen(model, "rb");

  assert(file != nullptr);

  int innerNodeCount;
  int leafCount;

  fread(&m_dim, 4, 1, file);
  fread(&innerNodeCount, 4, 1, file);
  fread(&leafCount, 4, 1, file);

  thrust::host_vector<BFSInnerNode> innerNodes(innerNodeCount);
  thrust::host_vector<VisualData> leaves(leafCount);

  for (int i = 0; i < innerNodeCount; i++) {
    innerNodes[i].deserialize(file);
  }

  for (int i = 0; i < leafCount; i++) {
    leaves[i].deserialize(file);
  }

  fread(&m_frameCount, 4, 1, file);
  fread(&m_boneCount, 4, 1, file);

  thrust::host_vector<float4x4> animation(m_frameCount * m_boneCount);
  fread(&animation[0], sizeof(float4x4), m_frameCount * m_boneCount, file);

  fclose(file);

  m_currentFrame = 0.0;

  /* Copy data to device */

  m_innerNodes = innerNodes;

  m_leaves = leaves;

  thrust::host_vector<BFSJob> queue;
  for (std::uint32_t i = 0; i < 8; ++i) {
    queue.push_back(
        make_BFSJob(i + 1, i & 1ul, (i & 2ul) >> 1, (i & 4ul) >> 2));
  }

  int level = 1, queueStart = 0, queueEnd = queue.size();
  // TODO: Test for level == nLevels
  while ((queueEnd - queueStart) <= 512 || queueStart == queueEnd) {
    for (int i = queueStart; i < queueEnd; ++i) {
      BFSJob job = queue[i];
      BFSInnerNode node = innerNodes[job.index];
      unsigned char childIndex = 0;
      for (unsigned int j = 0; j < 8; ++j) {
        if ((node.mask & (1ul << j)) != 0) {
          queue.push_back(make_BFSJob(
              node.childPtr + childIndex, 2 * job.x + (j & 1u),
              2 * job.y + ((j & 2u) >> 1), 2 * job.z + ((j & 4u) >> 2)));
          ++childIndex;
        }
      }
    }
    ++level;
    queueStart = queueEnd;
    queueEnd = queue.size();
  }

  m_level = level;

  m_jobs.assign(queue.cbegin() + queueStart, queue.cbegin() + queueEnd);

  m_animation = animation;
}

int BFSOctree::update(double lastFrameTimeInMilliseconds) {
  m_currentFrame += lastFrameTimeInMilliseconds;
  return (int)(m_currentFrame * 0.06) % m_frameCount;
}

thrust::device_vector<BFSInnerNode> const& BFSOctree::innerNodes() const {
  return m_innerNodes;
}

thrust::device_vector<VisualData> const& BFSOctree::leaves() const {
  return m_leaves;
}

thrust::device_vector<BFSJob> const& BFSOctree::jobs() const { return m_jobs; }

thrust::device_vector<float4x4> const& BFSOctree::animation() const {
  return m_animation;
}

int BFSOctree::level() const { return m_level; }

float BFSOctree::dim() const { return m_dim; }

int BFSOctree::boneCount() const { return m_boneCount; }

Texture const& BFSOctree::diffuse() const { return m_diffuse; }

Texture const& BFSOctree::illum() const { return m_illum; }

Texture const& BFSOctree::spec() const { return m_spec; }

Texture const& BFSOctree::normal() const { return m_normal; }