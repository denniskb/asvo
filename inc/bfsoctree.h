#ifndef bfsoctree_h

#define bfsoctree_h

#include <memory>

#include <thrust/device_vector.h>

#include "bfsinnernode.h"
#include "BFSJob.h"
#include "matrix.h"
#include "Texture.h"

/**
 * Represents a BFSOctree.
 */
typedef struct
{
	// HACK: Use of managed ptr to defer initialization (we have hidden copies of BFSOctree)
	// HACK: Use of shared ptr because you can't copy a unique_ptr
	std::shared_ptr< thrust::device_vector< BFSInnerNode > > d_innerNodes;
	std::shared_ptr< thrust::device_vector< VisualData > > d_leaves;
	std::shared_ptr< thrust::device_vector< BFSJob > > d_jobs;
	unsigned long int innerNodeCount;
	unsigned long int leafCount;
	unsigned short int jobCount;
	unsigned char level;
	float dim;
	Matrix *d_animation;
	unsigned long int frameCount;
	unsigned long int boneCount;
	double *currentFrame;
	// HACK: Prevents us from unnecessarily copying textures but
	// forces all BFSOctrees to share one set of textures.
	std::shared_ptr< Texture > diffuse;
	std::shared_ptr< Texture > illum;
	std::shared_ptr< Texture > spec;
	std::shared_ptr< Texture > normal;
} BFSOctree;

#endif