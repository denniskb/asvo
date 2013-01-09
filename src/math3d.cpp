#include "../inc/math3d.h"

#include "../inc/matrix.h"
#include "../inc/util.h"
#include "../inc/vector3.h"

#ifndef __CUDACC__
#include <math.h>
#endif

Vector3 $$(vecNormalize)(Vector3 vec)
{
	const float lengthMul = 1.f / sqrtf($(vecLenSquared)(vec));
	return $(vecMulS)(vec, lengthMul);
}

float $$(vecLenSquared)(Vector3 vec)
{
	return vec.x * vec.x + vec.y * vec.y + vec.z * vec.z;
}

Vector3 $$(vecCross)(Vector3 vec1, Vector3 vec2)
{
	Vector3 result = { vec1.y * vec2.z - vec1.z * vec2.y,
					   vec1.z * vec2.x - vec1.x * vec2.z,
					   vec1.x * vec2.y - vec1.y * vec2.x };
	return result;
}

float $$(vecDot)(Vector3 vec1, Vector3 vec2)
{
	return vec1.x * vec2.x +
		   vec1.y * vec2.y +
		   vec1.z * vec2.z;
}

Vector3 $$(vecNegate)(Vector3 vec)
{
	Vector3 result = { -vec.x,
					   -vec.y,
					   -vec.z };
	return result;
}

Vector3 $$(vecAddVec)(Vector3 vec1, Vector3 vec2)
{
	Vector3 result = { vec1.x + vec2.x,
					   vec1.y + vec2.y,
					   vec1.z + vec2.z };
	return result;
}

Vector3 $$(vecSubVec)(Vector3 vec1, Vector3 vec2)
{
	return $(vecAddVec)(vec1, $(vecNegate)(vec2));
}

Vector3 $$(vecMulS)(Vector3 vec, float s)
{
	Vector3 result = { vec.x * s,
					   vec.y * s,
					   vec.z * s };
	return result;
}

Vector3 $$(vecMulM)(Vector3 vec, Matrix m)
{
	Vector3 result = { vec.x * m.m11 + vec.y * m.m21 + vec.z * m.m31 + m.m41,
					   vec.x * m.m12 + vec.y * m.m22 + vec.z * m.m32 + m.m42,
					   vec.x * m.m13 + vec.y * m.m23 + vec.z * m.m33 + m.m43 };
	float w  = vec.x * m.m14 + vec.y * m.m24 + vec.z * m.m34 + m.m44;

	const float wMul = 1.f / w;
	
	return $(vecMulS)(result, wMul);
}

Matrix $$(mMulM)(Matrix m1, Matrix m2)
{
	Matrix result = { m1.m11 * m2.m11 + m1.m12 * m2.m21 + m1.m13 * m2.m31 + m1.m14 * m2.m41,
					  m1.m11 * m2.m12 + m1.m12 * m2.m22 + m1.m13 * m2.m32 + m1.m14 * m2.m42,
					  m1.m11 * m2.m13 + m1.m12 * m2.m23 + m1.m13 * m2.m33 + m1.m14 * m2.m43,
					  m1.m11 * m2.m14 + m1.m12 * m2.m24 + m1.m13 * m2.m34 + m1.m14 * m2.m44,
	
					  m1.m21 * m2.m11 + m1.m22 * m2.m21 + m1.m23 * m2.m31 + m1.m24 * m2.m41,
					  m1.m21 * m2.m12 + m1.m22 * m2.m22 + m1.m23 * m2.m32 + m1.m24 * m2.m42,
					  m1.m21 * m2.m13 + m1.m22 * m2.m23 + m1.m23 * m2.m33 + m1.m24 * m2.m43,
					  m1.m21 * m2.m14 + m1.m22 * m2.m24 + m1.m23 * m2.m34 + m1.m24 * m2.m44,
	
					  m1.m31 * m2.m11 + m1.m32 * m2.m21 + m1.m33 * m2.m31 + m1.m34 * m2.m41,
					  m1.m31 * m2.m12 + m1.m32 * m2.m22 + m1.m33 * m2.m32 + m1.m34 * m2.m42,
					  m1.m31 * m2.m13 + m1.m32 * m2.m23 + m1.m33 * m2.m33 + m1.m34 * m2.m43,
					  m1.m31 * m2.m14 + m1.m32 * m2.m24 + m1.m33 * m2.m34 + m1.m34 * m2.m44,
	
					  m1.m41 * m2.m11 + m1.m42 * m2.m21 + m1.m43 * m2.m31 + m1.m44 * m2.m41,
					  m1.m41 * m2.m12 + m1.m42 * m2.m22 + m1.m43 * m2.m32 + m1.m44 * m2.m42,
					  m1.m41 * m2.m13 + m1.m42 * m2.m23 + m1.m43 * m2.m33 + m1.m44 * m2.m43,
					  m1.m41 * m2.m14 + m1.m42 * m2.m24 + m1.m43 * m2.m34 + m1.m44 * m2.m44 };
	return result;
}

Matrix $$(createTranslation)(Vector3 vec)
{
	Matrix result = IDENTITY;
	result.m41 = vec.x;
	result.m42 = vec.y;
	result.m43 = vec.z;
	return result;
}

Matrix $$(createRotation)(Vector3 axis, float angle)
{
	const float sinAngle = sinf(-angle);
	const float cosAngle = cosf(-angle);
	const float oneMinusCosAngle = 1.f - cosAngle;
	
	const float xTimesY = axis.x * axis.y;
	const float xTimesZ = axis.x * axis.z;
	const float yTimesZ = axis.y * axis.z;

	const float xTimesSin = axis.x * sinAngle;
	const float yTimesSin = axis.y * sinAngle;
	const float zTimesSin = axis.z * sinAngle;

	Matrix result = { axis.x * axis.x * oneMinusCosAngle + cosAngle,
					  xTimesY		  *	oneMinusCosAngle - zTimesSin,
					  xTimesZ		  *	oneMinusCosAngle + yTimesSin,
					  0.f,
	
					  xTimesY		  *	oneMinusCosAngle + zTimesSin,
					  axis.y * axis.y * oneMinusCosAngle + cosAngle,
					  yTimesZ		  *	oneMinusCosAngle - xTimesSin,
					  0.f,
	
					  xTimesZ		  *	oneMinusCosAngle - yTimesSin,
					  yTimesZ		  *	oneMinusCosAngle + xTimesSin,
					  axis.z * axis.z * oneMinusCosAngle + cosAngle,
					  0.f,
	
					  0.f, 0.f, 0.f, 1.f};
	return result;
}

Matrix $$(createCam)(Vector3 pos, Vector3 lookAt, Vector3 up)
{
	Vector3 zAxis = $(vecNormalize)($(vecSubVec)(lookAt, pos));
	
	Vector3 xAxis = $(vecNormalize)($(vecCross)(up, zAxis));	

	Vector3 yAxis = $(vecNormalize)($(vecCross)(zAxis, xAxis));
		
	Matrix axes = { xAxis.x, yAxis.x, zAxis.x, 0.f,
					xAxis.y, yAxis.y, zAxis.y, 0.f,
					xAxis.z, yAxis.z, zAxis.z, 0.f,
					0.f,	 0.f,	  0.f,	   1.f };

	return $(mMulM)($(createTranslation)($(vecNegate)(pos)), axes);
}

Matrix $$(createProjection)(float fov, float ratio, float nearPlane, float farPlane)
{
	float cotAlpha2 = 1.f / tan(fov * .5f);	
	
	float fThroughFMinusN = farPlane / (farPlane - nearPlane);

	Matrix result = { cotAlpha2 / ratio,  0.f,			  0.f,							0.f,
					  0.f,				  cotAlpha2,      0.f,							0.f,
					  0.f,				  0.f,			  fThroughFMinusN,				1.f,
					  0.f,				  0.f,			  -fThroughFMinusN * nearPlane,	0.f };
	return result;
}

Matrix $$(createOrthographic)(float width, float height,
							  float nearPlane, float farPlane)
{
	float oneThroughFMinusN = 1.0f / (farPlane - nearPlane);

	Matrix result = { 2.0f / width,	0.f,		   0.f,								0.f,
					  0.f,			2.0f / height, 0.f,								0.f,
					  0.f,			0.f,		   oneThroughFMinusN,				0.f,
					  0.f,			0.f,		   -nearPlane * oneThroughFMinusN,	1.f };

	return result;
}