#ifndef bfsoctree_h

#define bfsoctree_h

#include <memory>

#include "bfsinnernode.h"
#include "bfsleaf.h"
#include "bfsjob.h"
#include "matrix.h"
#include "Texture.h"

/**
 * Represents a BFSOctree.
 */
typedef struct
{
	BFSInnerNode *innerNodes;
	BFSLeaf *leaves;
	BFSInnerNode *d_innerNodes;
	BFSLeaf *d_leaves;
	BFSJob *d_jobs;
	unsigned long int innerNodeCount;
	unsigned long int leafCount;
	unsigned short int jobCount;
	unsigned char level;
	float dim;
	Matrix *animation;
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