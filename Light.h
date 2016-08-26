#pragma once

#include <vector_types.h>

#include "Camera.h"

class Light
{
public:

	// Creates a directional light at position looking at the origin
	Light( float3 position, float diffusePower );

	float3 direction() const;
	
	float diffusePower() const;
	float ambientPower() const;

	// Returns a camera which is equivalent to the
	// light's frustum
	Camera camera() const;

private:

	float3 m_direction;
	float m_diffusePower;
};