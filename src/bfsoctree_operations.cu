#include "../inc/bfsoctree_operations.h"

#include <cassert>
#include <cstdio>
#include <cstdlib>

#include <cuda_runtime.h>
#include <thrust/host_vector.h>

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

	thrust::host_vector< BFSInnerNode > innerNodes( result.innerNodeCount );
	thrust::host_vector< VisualData > leaves( result.leafCount );

	fread( & innerNodes[ 0 ], sizeof( BFSInnerNode ), result.innerNodeCount, file );
	fread( & leaves[ 0 ], sizeof( VisualData ), result.leafCount, file );

	fread( & result.frameCount, 4, 1, file );
	fread( & result.boneCount, 4, 1, file );

	thrust::host_vector< Matrix > animation( result.frameCount * result.boneCount );
	fread( & animation[ 0 ], sizeof( Matrix ), result.frameCount * result.boneCount, file );

	fclose(file);

	result.currentFrame = 0.0;

	// TODO: Replace hard-coded values
	result.diffuse.reset( new Texture( diffuse, 1024, 1024 ) );
	result.illum.reset( new Texture( illum, 1024, 1024 ) );
	result.spec.reset( new Texture( spec, 1024, 1024 ) );
	result.normal.reset( new Texture( normal, 1024, 1024 ) );

	/* Copy data to device */

	result.d_innerNodes.reset( new thrust::device_vector< BFSInnerNode > );
	* result.d_innerNodes = innerNodes;

	result.d_leaves.reset( new thrust::device_vector< VisualData > );
	* result.d_leaves = leaves;

	thrust::host_vector< BFSJob > queue;
	for (unsigned long int i = 0; i < 8; ++i)
	{
		queue.push_back( make_BFSJob
		( 
			i + 1, 
			i & 1ul, 
			( i & 2ul ) >> 1, 
			( i & 4ul ) >> 2 
		));
	}

	int level = 1, queueStart = 0, queueEnd = queue.size();
	// TODO: Test for level == nLevels
	while( ( queueEnd - queueStart ) <= 512 || queueStart == queueEnd )
	{
		for (int i = queueStart; i < queueEnd; ++i)
		{
			BFSJob job = queue[ i ];
			BFSInnerNode node = innerNodes[ job.index ];				
			unsigned char childIndex = 0;
			for (unsigned int j = 0; j < 8; ++j)
			{
				if ((node.mask & (1ul << j)) != 0)
				{						
					queue.push_back( make_BFSJob
					( 
						node.childPtr + childIndex,
						2 * job.x + ( j & 1u ),
						2 * job.y + ( ( j & 2u ) >> 1 ),
						2 * job.z + ( ( j & 4u ) >> 2 )
					));
					++childIndex;
				}
			}
		}			
		++level;
		queueStart = queueEnd;
		queueEnd = queue.size();			
	}

	result.jobCount = queueEnd - queueStart;	
	result.level = level;

	result.d_jobs.reset( new thrust::device_vector< BFSJob > );
	result.d_jobs->assign( queue.cbegin() + queueStart, queue.cbegin() + queueEnd );

	result.d_animation.reset( new thrust::device_vector< Matrix > );
	* result.d_animation = animation;

	return result;
}

void BFSOctreeCleanup(BFSOctree *octree)
{
	octree->d_innerNodes.reset< thrust::device_vector< BFSInnerNode > >( nullptr );
	octree->innerNodeCount = 0;
	
	octree->d_leaves.reset< thrust::device_vector< VisualData > >( nullptr );
	octree->leafCount = 0;

	octree->d_jobs.reset< thrust::device_vector< BFSJob > >( nullptr );
	octree->jobCount = 0;

	octree->d_animation.reset< thrust::device_vector< Matrix > >( nullptr );
	octree->frameCount = octree->boneCount = 0;

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

unsigned short int BFSOctreeUpdate( BFSOctree * octree )
{
	octree->currentFrame += glueGetLastFrameTime();
	return (int)( octree->currentFrame * 0.06 ) % octree->frameCount;
}