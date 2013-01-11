#include "../inc/bfsoctree_operations.h"

#include <cassert>
#include <cstdio>
#include <cstdlib>

#include <cuda_runtime.h>

#include "../inc/bfsinnernode.h"
#include "../inc/bfsleaf.h"
#include "../inc/bfsoctree.h"
#include "../inc/glue.h"
#include "../inc/matrix.h"

BFSOctree BFSOctreeImport(char const * path, char const * diffuse, char const * illum, char const * spec, char const * normal)
{
	BFSOctree result;
	
	FILE *file = fopen(path, "rb");

	assert( file != nullptr );

	fread(&result.dim, 4, 1, file);
	fread(&result.innerNodeCount, 4, 1, file);
	fread(&result.leafCount, 4, 1, file);

	result.innerNodes = (BFSInnerNode*) malloc(result.innerNodeCount * sizeof(BFSInnerNode));
	result.leaves = (BFSLeaf*) malloc(result.leafCount * sizeof(BFSLeaf));
	result.d_innerNodes = NULL;
	result.d_leaves = NULL;
	result.d_jobs = NULL;

	fread(result.innerNodes, sizeof(BFSInnerNode), result.innerNodeCount, file);	

	fread(result.leaves, sizeof(BFSLeaf), result.leafCount, file);

	fread(&result.frameCount, 4, 1, file);
	fread(&result.boneCount, 4, 1, file);

	result.animation = (Matrix*) malloc(result.frameCount * result.boneCount * sizeof(Matrix));
	fread(result.animation, sizeof(Matrix), result.frameCount * result.boneCount, file);
	result.d_animation = NULL;

	fclose(file);

	result.currentFrame = (double*) malloc(sizeof(double));
	*(result.currentFrame) = 0.0;

	// TODO: Replace hard-coded values
	result.diffuse.reset( new Texture( diffuse, 1024, 1024 ) );
	result.illum.reset( new Texture( illum, 1024, 1024 ) );
	result.spec.reset( new Texture( spec, 1024, 1024 ) );
	result.normal.reset( new Texture( normal, 1024, 1024 ) );

	return result;
}

void BFSOctreeCopyToDevice(BFSOctree *octree)
{
	if (octree->d_innerNodes == NULL)
	{
		cudaMalloc((void**) &(octree->d_innerNodes), octree->innerNodeCount * sizeof(BFSInnerNode));	
		cudaMemcpy(octree->d_innerNodes, octree->innerNodes, octree->innerNodeCount * sizeof(BFSInnerNode), cudaMemcpyHostToDevice);
	}
	if (octree->d_leaves == NULL)
	{
		cudaMalloc((void**) &(octree->d_leaves), octree->leafCount * sizeof(BFSLeaf));	
		cudaMemcpy(octree->d_leaves, octree->leaves, octree->leafCount * sizeof(BFSLeaf), cudaMemcpyHostToDevice);
	}

	if (octree->d_jobs == NULL)
	{
		// TODO: Allocate on heap
		BFSJob queue[10000];
		for (unsigned long int i = 0; i < 8; ++i)
		{
			BFSJob job = { i + 1, i & 1ul, (i & 2ul) >> 1, (i & 4ul) >> 2 };
			queue[i] = job;
		}

		int level = 1, queueStart = 0, queueEnd = 8, queuePtr = 8;
		// TODO: Test for level == nLevels
		while ((queueEnd - queueStart) <= 512 || queueStart == queueEnd)
		{
			for (int i = queueStart; i < queueEnd; ++i)
			{
				BFSJob job = queue[i];
				BFSInnerNode node = octree->innerNodes[job.index];				
				unsigned char childIndex = 0;
				for (unsigned int j = 0; j < 8; ++j)
				{
					if ((node.mask & (1ul << j)) != 0)
					{						
						BFSJob iJob = { node.childPtr + childIndex,
									    2 * job.x + (j & 1u),
									    2 * job.y + ((j & 2u) >> 1),
									    2 * job.z + ((j & 4u) >> 2)};

						queue[queuePtr++] = iJob;
						++childIndex;
					}
				}
			}			
			++level;
			queueStart = queueEnd;
			queueEnd = queuePtr;			
		}
		
		cudaMalloc((void**) &(octree->d_jobs), (queueEnd - queueStart) * sizeof(BFSJob));	
		cudaMemcpy(octree->d_jobs, queue + queueStart, (queueEnd - queueStart) * sizeof(BFSJob), cudaMemcpyHostToDevice);
		octree->jobCount = queueEnd - queueStart;	
		octree->level = level;
	}

	if (octree->d_animation == NULL)
	{
		cudaMalloc((void**) &(octree->d_animation), octree->frameCount * octree->boneCount * sizeof(Matrix));	
		cudaMemcpy(octree->d_animation, octree->animation, octree->frameCount * octree->boneCount * sizeof(Matrix), cudaMemcpyHostToDevice);
	}
}

void BFSOctreeCleanup(BFSOctree *octree)
{
	if (octree->innerNodes != NULL)
	{
		free(octree->innerNodes);
		octree->innerNodes = NULL;
		octree->innerNodeCount = 0;
	}
	if (octree->leaves != NULL)
	{
		free(octree->leaves);
		octree->leaves = NULL;
		octree->leafCount = 0;
	}	
	if (octree->d_innerNodes != NULL)
	{
		cudaFree(octree->d_innerNodes);
		octree->d_innerNodes = NULL;
	}
	if (octree->d_leaves != NULL)
	{
		cudaFree(octree->d_leaves);
		octree->d_leaves = NULL;
	}
	if (octree->d_jobs != NULL)
	{
		cudaFree(octree->d_jobs);
		octree->d_jobs = NULL;
		octree->jobCount = 0;
	}
	if (octree->animation != NULL)
	{
		free(octree->animation);
		octree->animation = NULL;
		octree->frameCount = octree->boneCount = 0;
	}
	if (octree->d_animation != NULL)
	{
		cudaFree(octree->d_animation);
		octree->d_animation = NULL;
		octree->frameCount = octree->boneCount = 0;
	}
	if (octree->currentFrame != NULL)
	{
		free(octree->currentFrame);
		octree->currentFrame = NULL;
	}

	octree->diffuse.reset< Texture >( nullptr );
	octree->illum.reset< Texture >( nullptr );
	octree->spec.reset< Texture >( nullptr );
	octree->normal.reset< Texture >( nullptr );
}

unsigned long int h_getChildCountFromMask(unsigned long int mask)
{
	return (1ul & mask) +
		  ((2ul & mask) >> 1) +
		  ((4ul & mask) >> 2) +
		  ((8ul & mask) >> 3) +
		  ((16ul & mask) >> 4) +
		  ((32ul & mask) >> 5) +
		  ((64ul & mask) >> 6) +
		  ((128ul & mask) >> 7);
}

unsigned short int BFSOctreeUpdate(BFSOctree *octree)
{
	*octree->currentFrame += glueGetLastFrameTime();
	return (int)(*octree->currentFrame * 0.06) % octree->frameCount;
}