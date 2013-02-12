#pragma once

#include "vector3.h"

/**
 * Stores all the neccessary data for drawing a
 * voxel. Used by the renderer. The equivalent in
 * OpenGL would be the input to the pixel/fragment shader.
 */
class VoxelData
{
public:

	Vector3 normal;
	Vector3 tangent;
	float4 pos;
	float2 texCoord;
	Vector3 eyeVec;
	Vector3 center;
};