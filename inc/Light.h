#pragma once

#include "Camera.h"
#include "vector3.h"

class Light
{
public:

	// HACK: Remove once we get rid of a global light source
	Light();

	// Creates a directional light at position looking at the origin
	Light( Vector3 position, float diffusePower );

	Vector3 direction() const;
	
	float diffusePower() const;
	float ambientPower() const;

	// Returns a camera which is equivalent to the
	// light's frustum
	Camera camera() const;

private:

	Vector3 m_direction;
	float m_diffusePower;
};