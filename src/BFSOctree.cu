#include "../inc/BFSOctree.h"

#include <cassert>

#include "../inc/glue.h"

BFSOctree::BFSOctree()
{

}

BFSOctree::BFSOctree
(
	char const * model,
	char const * diffuse, 
	char const * illum, 
	char const * spec, 
	char const * normal
)
{	
	FILE * file = fopen( model, "rb" );

	assert( file != nullptr );

	int innerNodeCount;
	int leafCount;

	fread( & m_dim, 4, 1, file);
	fread( & innerNodeCount, 4, 1, file);
	fread( & leafCount, 4, 1, file);

	thrust::host_vector< BFSInnerNode > innerNodes( innerNodeCount );
	thrust::host_vector< VisualData > leaves( leafCount );

	for( int i = 0; i < innerNodeCount; i++ )
	{
		innerNodes[ i ].deserialize( file );
	}

	for( int i = 0; i < leafCount; i++ )
	{
		leaves[ i ].deserialize( file );
	}

	fread( & m_frameCount, 4, 1, file );
	fread( & m_boneCount, 4, 1, file );

	thrust::host_vector< Matrix > animation( m_frameCount * m_boneCount );
	fread( & animation[ 0 ], sizeof( Matrix ), m_frameCount * m_boneCount, file );

	fclose(file);

	m_currentFrame = 0.0;

	// TODO: Replace hard-coded values
	m_pDiffuse.reset( new Texture( diffuse, 1024, 1024 ) );
	m_pIllum.reset( new Texture( illum, 1024, 1024 ) );
	m_pSpec.reset( new Texture( spec, 1024, 1024 ) );
	m_pNormal.reset( new Texture( normal, 1024, 1024 ) );

	/* Copy data to device */

	m_dpInnerNodes.reset( new thrust::device_vector< BFSInnerNode > );
	* m_dpInnerNodes = innerNodes;

	m_dpLeaves.reset( new thrust::device_vector< VisualData > );
	* m_dpLeaves = leaves;

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

	m_level = level;

	m_dpJobs.reset( new thrust::device_vector< BFSJob > );
	m_dpJobs->assign( queue.cbegin() + queueStart, queue.cbegin() + queueEnd );

	m_dpAnimation.reset( new thrust::device_vector< Matrix > );
	* m_dpAnimation = animation;
}


 
int BFSOctree::update()
{
	m_currentFrame += glueGetLastFrameTime();
	return (int)( m_currentFrame * 0.06 ) % m_frameCount;
}



thrust::device_vector< BFSInnerNode > const * BFSOctree::innerNodes() const
{
	return m_dpInnerNodes.get();
}

thrust::device_vector< VisualData > const * BFSOctree::leaves() const
{
	return m_dpLeaves.get();
}

thrust::device_vector< BFSJob > const * BFSOctree::jobs() const
{
	return m_dpJobs.get();
}

thrust::device_vector< Matrix > const * BFSOctree::animation() const
{
	return m_dpAnimation.get();
}



int BFSOctree::level() const
{
	return m_level;
}

float BFSOctree::dim() const
{
	return m_dim;
}

int BFSOctree::boneCount() const
{
	return m_boneCount;
}
	


Texture const * BFSOctree::diffuse() const
{
	return m_pDiffuse.get();
}

Texture const * BFSOctree::illum() const
{
	return m_pIllum.get();
}

Texture const * BFSOctree::spec() const
{
	return m_pSpec.get();
}

Texture const * BFSOctree::normal() const
{
	return m_pNormal.get();
}