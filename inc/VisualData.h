#pragma once

#include <cstdio>

#include <vector_types.h>

/**
 * Stores the visual data of a voxel.
 */
class VisualData
{
public:

	float3 normal;
	float3 tangent;
	float2 texCoord;
	unsigned char boneIndex0;
	unsigned char boneIndex1;
	unsigned char boneIndex2;
	unsigned char boneIndex3;
	float4 boneWeights;

	void deserialize( FILE * inFile );
};