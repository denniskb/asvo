#ifndef bfsoctree_h

#define bfsoctree_h

#include "bfsinnernode.h"
#include "bfsleaf.h"
#include "bfsjob.h"
#include "matrix.h"
#include "texture.h"

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
	Texture diffuse;
	Texture illum;
	Texture spec;
	Texture normal;
} BFSOctree;

#endif