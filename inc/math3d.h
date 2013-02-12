/** \file
 * Defines some helper functions for handling 3D math.
 */

#ifndef math3d_h

#define math3d_h

#include "util.h"
#include "vector3.h"
#include "Matrix.h"

// Definition of some constants.

#ifdef __CUDACC__
__constant__ Matrix IDENTITY =
#else
const Matrix IDENTITY = 
#endif
{ 1.f, 0.f, 0.f, 0.f,
  0.f, 1.f, 0.f, 0.f,
  0.f, 0.f, 1.f, 0.f,
  0.f, 0.f, 0.f, 1.f };

#ifdef __CUDACC__
__constant__ Vector3 UNIT_X = { 1.f, 0.f, 0.f };
__constant__ Vector3 UNIT_Y = { 0.f, 1.f, 0.f };
__constant__ Vector3 UNIT_Z = { 0.f, 0.f, 1.f };
__constant__ Vector3 ZERO = { 0.f, 0.f, 0.f };
__constant__ Vector3 ONE = { 1.f, 1.f, 1.f };
#else
const Vector3 UNIT_X = { 1.f, 0.f, 0.f };
const Vector3 UNIT_Y = { 0.f, 1.f, 0.f };
const Vector3 UNIT_Z = { 0.f, 0.f, 1.f };
const Vector3 ZERO = { 0.f, 0.f, 0.f };
const Vector3 ONE = { 1.f, 1.f, 1.f };
#endif

/**
 * Normalizes a vector.
 *
 * @param vec The vector to be normalized.
 *
 * @return A vector that has the same direction as vec,
 *         but whose length equals 1.0
 */
Vector3 $$(vecNormalize)(Vector3 vec);

/**
 * Computes the sqaured length of a vector.
 *
 * @param vec The vector whose length to compute.
 *
 * @return The length of vec squared.
 */
float $$(vecLenSquared)(Vector3 vec);

/**
 * Computes the cross product of two vectors.
 *
 * @param vec1 First vector of the cross product.
 * @param vec2 Second vector of the cross product.
 *
 * @return vec1 multiplied by vec2 using the cross product.
 */
Vector3 $$(vecCross)(Vector3 vec1, Vector3 vec2);

/**
 * Computes the dot product of two vectors.
 *
 * @param vec1 The first vector of the dot product.
 * @param vec2 The second vector of the dot product.
 *
 * @return vec1 multiplied by vec2 using the dot product.
 */
float $$(vecDot)(Vector3 vec1, Vector3 vec2);

/**
 * Negates a vector.
 *
 * @param vec The vector to be negated.
 *
 * @return -vec
 */
Vector3 $$(vecNegate)(Vector3 vec);

/**
 * Computes the sum of two vectors.
 *
 * @param vec1 The first vector of the sum.
 * @param vec2 The second vector of the sum.
 *
 * @return vec1 + vec2
 */
Vector3 $$(vecAddVec)(Vector3 vec1, Vector3 vec2);

/**
 * Subtracts a vector from another vector.
 *
 * @param vec1 The vector from which to subtract another vector.
 * @param vec2 The vector to subtract from the first vector.
 *
 * @return vec1 - vec2
 */
Vector3 $$(vecSubVec)(Vector3 vec1, Vector3 vec2);

/**
 * Computes the product of a vector and a scalar.
 *
 * @param vec The vector of the product.
 * @param s   The scalar of the product.
 *
 * @return vec * s
 */
Vector3 $$(vecMulS)(Vector3 vec, float s);

/**
 * Computes the product of a vector and a matrix (vector on left side/row major!).
 *
 * @param vec The vector of the product.
 * @param m   The matrix of the product.
 *
 * @return vec * m (row major)
 */
Vector3 $$(vecMulM)(Vector3 vec, Matrix m);

/**
 * Computes the product of two matrices.
 *
 * @param m1 The first matrix of the product.
 * @param m2 The second matrix of the product.
 *
 * @return m1 * m2
 */
Matrix $$(mMulM)(Matrix m1, Matrix m2);

/**
 * Creates a translation Matrix from an offset vector.
 *
 * @param vec The vector to create the translation matrix out of.
 *
 * return A translation matrix that, multiplied by a vector,
 *        has the same effect as adding vec to that vector.
 */
Matrix $$(createTranslation)(Vector3 vec);

/*
 * Creates a rotation matrix from an axis vector and an angle.
 * Axis must be normalized!
 * Original code by David Schergen.
 *
 * @param axis  A vector representing the axis to rotate around.
 * @param angle The angle by which to rotate.
 *
 * @return A rotation matrix that, multiplied by a vector,
 *         has the effect of rotating that vector around axis by angle.
 */
Matrix $$(createRotation)(Vector3 axis, float angle);

/**
 * Creates a camera Matrix.
 * Original code by David Scherfgen.
 *
 * @param pos    Position of the virtual camera.
 * @param lookAt Point at which the virtual camera looks.
 * @param up     The up-vector of the virtual camera.
 *
 * @return A matrix representing a transformation into the
 *         local coordinate system of the specified virtual camera.
 */
Matrix $$(createCam)(Vector3 pos, Vector3 lookAt, Vector3 up);

/**
 * Creates a projection Matrix.
 * Original code by Joe Farrell.
 *
 * @param fov       The field of vision, in radians, of the virtual camera.
 * @param ratio     The aspect ratio (width / height) of the virtual camera.
 * @param nearPlane The distance, in world space units, to the near clipping plane of the virtual camera.
 * @param farPlane  The distance, in world space units, to the far clipping plane of the virtual camera.
 *
 * @return A matrix representing a projective transformation defined by
 *         the virtual camera.
 */
Matrix $$(createProjection)(float fov, float ratio, float nearPlane, float farPlane);

/**
 * Creates an orthographic projection Matrix.
 * Original code by Joe Farrell.
 *
 * @param width     The width, in world space units, of the virtual camera.
 * @param height    The height, in world space units, of the virtual camera.
 * @param nearPlane The distance, in world space units, to the near clipping plane of the virtual camera.
 * @param farPlane  The distance, in world space units, to the far clipping plane of the virtual camera.
 *
 * @return A matrix representing an orthographic-projective transformation defined by
 *         the virtual camera.
 */
Matrix $$(createOrthographic)(float width, float height,
                              float nearPlane, float farPlane);							  

#endif