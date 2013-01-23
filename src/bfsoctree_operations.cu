#include "../inc/bfsoctree_operations.h"

#include <cassert>
#include <cstdio>
#include <cstdlib>

#include <cuda_runtime.h>

#include "../inc/bfsinnernode.h"
#include "../inc/bfsoctree.h"
#include "../inc/glue.h"
#include "../inc/matrix.h"

BFSOctree BFSOctreeImport(char const * path, char const * diffuse, char const * illum, char const * spec, char const * normal)
{
	BFSOctree result;
	
	FILE *file = fopen(path, "rb");

	assert( file != nullptr );

	fread( & result.dim, 4, 1, file);
	fread( & result.innerNodeCount, 4, 1, file);
	fread( & result.leafCount, 4, 1, file);

	std::vector< BFSInnerNode > innerNodes( result.innerNodeCount );
	std::vector< VisualData > leaves( result.leafCount );
	result.d_leaves = NULL;

	fread( & innerNodes[ 0 ], sizeof( BFSInnerNode ), result.innerNodeCount, file );
	fread( & leaves[ 0 ], sizeof( VisualData ), result.leafCount, file );

	fread(&result.frameCount, 4, 1, file);
	fread(&result.boneCount, 4, 1, file);

	std::vector< Matrix > animation( result.frameCount * result.boneCount );
	fread( & animation[ 0 ], sizeof( Matrix ), result.frameCount * result.boneCount, file );
	result.d_animation = NULL;

	fclose(file);

	result.currentFrame = (double*) malloc(sizeof(double));
	*(result.currentFrame) = 0.0;

	// TODO: Replace hard-coded values
	result.diffuse.reset( new Texture( diffuse, 1024, 1024 ) );
	result.illum.reset( new Texture( illum, 1024, 1024 ) );
	result.spec.reset( new Texture( spec, 1024, 1024 ) );
	result.normal.reset( new Texture( normal, 1024, 1024 ) );

	/* Copy data to device */

	result.d_innerNodes.reset( new thrust::device_vector< BFSInnerNode >( result.innerNodeCount ) );
	cudaMemcpy
	( 
		thrust::raw_pointer_cast( result.d_innerNodes->data() ),
		& innerNodes[ 0 ], 
		result.innerNodeCount * sizeof( BFSInnerNode ),
		cudaMemcpyHostToDevice
	);
	
	cudaMalloc( ( void ** ) &( result.d_leaves ), result.leafCount * sizeof( VisualData ) );	
	cudaMemcpy( result.d_leaves, & leaves[ 0 ], result.leafCount * sizeof( VisualData ), cudaMemcpyHostToDevice );

	result.d_jobs.reset( new thrust::device_vector< BFSJob > );
	// TODO: Allocate on heap
	// TODO: Make this a thrust::host_vector
	BFSJob queue[10000];
	for (unsigned long int i = 0; i < 8; ++i)
	{
		queue[i] = make_BFSJob
		( 
			i + 1, 
			i & 1ul, 
			( i & 2ul ) >> 1, 
			( i & 4ul ) >> 2 
		);
	}

	int level = 1, queueStart = 0, queueEnd = 8, queuePtr = 8;
	// TODO: Test for level == nLevels
	while( ( queueEnd - queueStart ) <= 512 || queueStart == queueEnd )
	{
		for (int i = queueStart; i < queueEnd; ++i)
		{
			BFSJob job = queue[i];
			BFSInnerNode node = innerNodes[ job.index ];				
			unsigned char childIndex = 0;
			for (unsigned int j = 0; j < 8; ++j)
			{
				if ((node.mask & (1ul << j)) != 0)
				{						
					queue[queuePtr++] = make_BFSJob
					( 
						node.childPtr + childIndex,
						2 * job.x + ( j & 1u ),
						2 * job.y + ( ( j & 2u ) >> 1 ),
						2 * job.z + ( ( j & 4u ) >> 2 )
					);
					++childIndex;
				}
			}
		}			
		++level;
		queueStart = queueEnd;
		queueEnd = queuePtr;			
	}
		
	result.d_jobs->resize( queueEnd - queueStart );
	cudaMemcpy
	(
		thrust::raw_pointer_cast( result.d_jobs->data() ),
		queue + queueStart,
		( queueEnd - queueStart ) * sizeof( BFSJob ),
		cudaMemcpyHostToDevice
	);
	result.jobCount = queueEnd - queueStart;	
	result.level = level;

	cudaMalloc( ( void ** ) &( result.d_animation ), result.frameCount * result.boneCount * sizeof( Matrix ) );	
	cudaMemcpy( result.d_animation, & animation[ 0 ], result.frameCount * result.boneCount * sizeof( Matrix ), cudaMemcpyHostToDevice );

	return result;
}

void BFSOctreeCleanup(BFSOctree *octree)
{
	octree->d_innerNodes.reset< thrust::device_vector< BFSInnerNode > >( nullptr );
	octree->innerNodeCount = 0;
	
	cudaFree(octree->d_leaves);
	octree->d_leaves = NULL;

	octree->d_jobs.reset< thrust::device_vector< BFSJob > >( nullptr );
	octree->jobCount = 0;

	cudaFree(octree->d_animation);
	octree->d_animation = NULL;
	octree->frameCount = octree->boneCount = 0;

	free(octree->currentFrame);
	octree->currentFrame = NULL;

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