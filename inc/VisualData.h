#pragma once

#include "vector2.h"
#include "vector3.h"
#include "vector4.h"

/**
 * Stores the visual data of a voxel.
 */
class VisualData
{
public:

	Vector3 normal;
	Vector3 tangent;
	Vector2 texCoord;
	unsigned char boneIndex0;
	unsigned char boneIndex1;
	unsigned char boneIndex2;
	unsigned char boneIndex3;
	Vector4 boneWeights;
};