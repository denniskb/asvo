#include "../inc/object3d_operations.h"

#include <cutil.h>

#include "../inc/bfsoctree.h"
#include "../inc/math3d.h"
#include "../inc/matrix.h"
#include "../inc/object3d.h"

Object3d obj3dInit(BFSOctree data, CUTBoolean rhsCoordinateSystem)
{
	Object3d result;
	result.data = data;
	result.transform = IDENTITY;
	result.rhsCoordSystem = rhsCoordinateSystem;

	if (rhsCoordinateSystem)
		result.transform.m33 = -1.f;

	return result;
}

void obj3dAssignTransform(Object3d *obj, Matrix transform)
{
	if (obj->rhsCoordSystem)
	{
		Matrix mul = IDENTITY;
		mul.m33 = -1.f;
		obj->transform = h_mMulM(mul, transform);
	}
	else
		obj->transform = transform;
}