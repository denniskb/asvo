#pragma once

#include <vector_types.h>

#include "BFSOctree.h"
#include "float4x4.h"

/**
 * Encapsulates the raw data representation of a 3D object (the BFSOctree)
 * in a model that is easier to handle. It provides basic transformation.
 */
class Object3D
{
public:

	Object3D();
	Object3D( BFSOctree * data, bool rhsCoordianteSystem );

	void assignTransform( float4x4 const & transform );

	BFSOctree const * data() const;
	BFSOctree * data();
	float4x4 const & transform() const;

private:

	std::shared_ptr< BFSOctree > m_data;
	float4x4 m_transform;
	bool m_rhsCoordSystem;
};