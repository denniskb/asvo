#include "../inc/Object3D.h"

#include <vector_types.h>

#include "../inc/extended_helper_math.h"

Object3D::Object3D()
{

}

Object3D::Object3D( BFSOctree * data, bool rhsCoordianteSystem ) :
	m_data( data ),
	m_rhsCoordSystem( rhsCoordianteSystem )
{
	m_transform = make_identity();
	if( rhsCoordianteSystem )
	{
		m_transform.m33 = -1;
	}
}



void Object3D::assignTransform( float4x4 const & transform )
{
	float4x4 mul = make_identity();

	if( m_rhsCoordSystem )
	{
		mul.m33 = -1.f;
	}

	m_transform = ( mul * transform );
}



BFSOctree const * Object3D::data() const
{
	return m_data.get();
}

BFSOctree * Object3D::data()
{
	return m_data.get();
}

float4x4 const & Object3D::transform() const
{
	return m_transform;
}