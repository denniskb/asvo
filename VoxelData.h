#pragma once

#include <vector_types.h>

/**
 * Stores all the neccessary data for drawing a
 * voxel. Used by the renderer. The equivalent in
 * OpenGL would be the input to the pixel/fragment shader.
 */
struct VoxelData
{
	float3 normal;
	float3 tangent;
	float4 pos;
	float2 texCoord;
	float3 eyeVec;
	float3 center;
};