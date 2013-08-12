#include "../inc/Light.h"

#include "../inc/extended_helper_math.h"

Light::Light()
{

}

Light::Light( float3 position, float diffusePower ) :
	m_direction( normalize( -position ) ),
	m_diffusePower( diffusePower )
{

}



float3 Light::direction() const
{
	return m_direction;
}
	


float Light::diffusePower() const
{
	return m_diffusePower;
}

float Light::ambientPower() const
{
	return 1.0f - m_diffusePower;
}



Camera Light::camera() const
{
	return Camera
	(
		m_direction * 50.0f,
		make_float3( 0 ),
		// TODO: Implement float4x4
		make_orthographic( 100, 100, 10.f, 200.f )
	);
}