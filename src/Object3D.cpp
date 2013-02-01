#include "../inc/Object3D.h"

#include "../inc/math3d.h"

Object3D::Object3D()
{

}

Object3D::Object3D( BFSOctree * data, bool rhsCoordianteSystem ) :
	m_data( data ),
	m_rhsCoordSystem( rhsCoordianteSystem )
{
	m_transform = IDENTITY;
	if( rhsCoordianteSystem )
	{
		m_transform.m33 = -1;
	}
}



void Object3D::assignTransform( Matrix const & transform )
{
	Matrix mul = IDENTITY;

	if( m_rhsCoordSystem )
	{
		mul.m33 = -1.f;
	}

	m_transform = h_mMulM( mul, transform );
}



BFSOctree const * Object3D::data() const
{
	return m_data.get();
}

BFSOctree * Object3D::data()
{
	return m_data.get();
}

Matrix const & Object3D::transform() const
{
	return m_transform;
}