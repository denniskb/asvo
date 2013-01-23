#ifndef object3d_h

#define object3d_h

#include <helper_cuda.h>

#include "BFSOctree.h"
#include "matrix.h"

/**
 * Encapsulates the raw data representation of a 3D object (the BFSOctree)
 * in a model that is easier to handle. It provides basic transformation.
 */
typedef struct
{
	BFSOctree data;
	Matrix transform;
	bool rhsCoordSystem;
} Object3d;

#endif