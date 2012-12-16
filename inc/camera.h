#ifndef camera_h

#define camera_h

#include "matrix.h"
#include "vector3.h"

/**
 * Represents a virtual camera in 3D space.
 */
typedef struct
{
	Matrix view;
	Matrix projection;
	Matrix viewProjection;
	Vector3 pos;
} Camera;

#endif