/** \file */

#ifndef object3d_operations_h

#define object3d_operations_h

#include <helper_cuda.h>

#include "object3d.h"
#include "BFSOctree.h"
#include "Matrix.h"

/**
 * Initializes a 3D model.
 *
 * @param data                The BFSOctree to encapsulate by a model.
 * @param rhsCoordinateSystem Specifies whether the caller uses a
 *                            right-handed coordinate system or not.
 *
 * @return	A Object3d representing the provided BFSOctree.
 */
Object3d obj3dInit(BFSOctree data, bool rhsCoordinateSystem);

/**
 * Assigns a transformation matrix to a model.
 *
 * @param obj       The model to assign the transformation matrix to.
 * @param transform The transformation matrix to assign to the model.
 */
void obj3dAssignTransform(Object3d *obj, Matrix transform);

#endif