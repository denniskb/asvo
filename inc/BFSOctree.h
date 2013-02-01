#pragma once

#include <memory>

#include <thrust/device_vector.h>

#include "BFSInnerNode.h"
#include "BFSJob.h"
#include "Matrix.h"
#include "Texture.h"

class BFSOctree
{
public:

	BFSOctree();

	BFSOctree
	(
		char const * model,
		char const * diffuse, 
		char const * illum, 
		char const * spec, 
		char const * normal
	);
 
	/* Updates the character animation */
	int update();
 
	thrust::device_vector< BFSInnerNode > const * innerNodes() const;
	thrust::device_vector< VisualData > const * leaves() const;
	thrust::device_vector< BFSJob > const * jobs() const;
	thrust::device_vector< Matrix > const * animation() const;

	int level() const;
	float dim() const;
	int boneCount() const;
	
	Texture const * diffuse() const;
	Texture const * illum() const;
	Texture const * spec() const;
	Texture const * normal() const;

private:
 	 
 	// HACK: Use of managed ptr to defer initialization (we have hidden copies of BFSOctree)
	// HACK: Use of shared ptr because you can't copy a unique_ptr
	std::shared_ptr< thrust::device_vector< BFSInnerNode > > m_dpInnerNodes;
	std::shared_ptr< thrust::device_vector< VisualData > > m_dpLeaves;
	std::shared_ptr< thrust::device_vector< BFSJob > > m_dpJobs;
	std::shared_ptr< thrust::device_vector< Matrix > > m_dpAnimation;

	unsigned char m_level;
	float m_dim;
	/* #frames of the character animation */
	int m_boneCount;
	unsigned long int m_frameCount;
	double m_currentFrame;

	// HACK: Prevents us from unnecessarily copying textures but
	// forces all BFSOctrees to share one set of textures.
	std::shared_ptr< Texture > m_pDiffuse;
	std::shared_ptr< Texture > m_pIllum;
	std::shared_ptr< Texture > m_pSpec;
	std::shared_ptr< Texture > m_pNormal;
};