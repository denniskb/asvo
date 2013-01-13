#include "../inc/Light.h"

#include "../inc/math3d.h"

Light::Light()
{

}

Light::Light( Vector3 position, float diffusePower ) :
	m_direction( h_vecNormalize( h_vecNegate( position ) ) ),
	m_diffusePower( diffusePower )
{

}



Vector3 Light::direction() const
{
	return m_direction;
}
	


float Light::diffusePower() const
{
	return m_diffusePower;
}

float Light::ambientPower() const
{
	return 1.0 - m_diffusePower;
}



Camera Light::camera() const
{
	return Camera
	(
		h_vecMulS( m_direction, 50.0f ), ZERO,
		h_createOrthographic( 100, 100, 10.f, 200.f )
	);
}