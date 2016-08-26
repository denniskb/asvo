// Adding a few functions to CUDA's 'helper_math.h'

#pragma once

#include <vector_functions.h>
#include <vector_types.h>

#include "float4x4.h"

inline __host__ __device__ float3 operator/( float3 v, float f )
{
	return make_float3( v.x / f, v.y / f, v.z / f );
}

inline __host__ __device__ float3 operator*( float3 v, float f )
{
	return make_float3( v.x * f, v.y * f, v.z * f );
}

inline __host__ __device__ float3 operator+( float3 a, float3 b )
{
	return make_float3( a.x + b.x, a.y + b.y, a.z + b.z );
}

inline __host__ __device__ float3 & operator+=( float3 & a, float3 b )
{
	a.x += b.x;
	a.y += b.y;
	a.z += b.z;

	return a;
}

inline __host__ __device__ float3 operator-( float3 a, float3 b )
{
	return make_float3( a.x - b.x, a.y - b.y, a.z - b.z );
}

inline __host__ __device__ float3 & operator-=( float3 & a, float3 b )
{
	a.x -= b.x;
	a.y -= b.y;
	a.z -= b.z;

	return a;
}

inline __host__ __device__ float3 operator-( float3 v )
{
	return make_float3( -v.x, -v.y, -v.z );
}

inline __host__ __device__ float dot( float3 a, float3 b )
{
	return a.x * b.x + a.y * b.y + a.z * b.z;
}

inline __host__ __device__ float length( float3 v )
{
	return sqrtf( dot( v, v ) );
}

inline __host__ __device__ float3 normalize( float3 v )
{
	return v / length( v );
}

inline __host__ __device__ float3 lerp( float3 a, float3 b, float weightB )
{
	return a + ( b - a ) * weightB;
}

inline __host__ __device__ float3 cross( float3 a, float3 b )
{
	return make_float3(
		a.y * b.z - a.z * b.y,
		a.z * b.x - a.x * b.z,
		a.x * b.y - a.y * b.x
	);
}

// row vector times matrix with homogenization
inline __host__ __device__ float3 operator*( float3 v, float4x4 m )
{
	float3 result = make_float3( 
		v.x * m.m11 + v.y * m.m21 + v.z * m.m31 + m.m41,
		v.x * m.m12 + v.y * m.m22 + v.z * m.m32 + m.m42,
		v.x * m.m13 + v.y * m.m23 + v.z * m.m33 + m.m43 
	);
	
	float w  = v.x * m.m14 + v.y * m.m24 + v.z * m.m34 + m.m44;

	return result / w;
}

// TODO: Test effects of passing by reference
inline __host__ __device__ float4x4 operator*( float4x4 m1, float4x4 m2 )
{
	float4x4 result = { 
		m1.m11 * m2.m11 + m1.m12 * m2.m21 + m1.m13 * m2.m31 + m1.m14 * m2.m41,
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
		m1.m41 * m2.m14 + m1.m42 * m2.m24 + m1.m43 * m2.m34 + m1.m44 * m2.m44 
	};
	return result;
}

inline __host__ __device__ float4x4 make_identity()
{
	float4x4 result = {
		1, 0, 0, 0,
		0, 1, 0, 0,
		0, 0, 1, 0,
		0, 0, 0, 1
	};
	return result;
}

inline __host__ __device__ float4x4 make_translation( float3 vec )
{
	float4x4 result = make_identity();

	result.m41 = vec.x;
	result.m42 = vec.y;
	result.m43 = vec.z;

	return result;
}

inline __host__ __device__ float4x4 make_rotation( float3 axis, float angle )
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

	float4x4 result = { 
		axis.x * axis.x * oneMinusCosAngle + cosAngle,
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
	
		0.f, 0.f, 0.f, 1.f
	};
	return result;
}

inline __host__ __device__ float4x4 make_lookat( float3 pos, float3 lookAt, float3 up )
{
	float3 zAxis = normalize( lookAt - pos);
	
	float3 xAxis = normalize( cross( up, zAxis ) );	

	float3 yAxis = normalize( cross( zAxis, xAxis ) ); 
		
	float4x4 axes = { 
		xAxis.x, yAxis.x, zAxis.x, 0.f,
		xAxis.y, yAxis.y, zAxis.y, 0.f,
		xAxis.z, yAxis.z, zAxis.z, 0.f,
		0.f,	 0.f,	  0.f,	   1.f 
	};

	return make_translation( -pos ) * axes;
}

inline __host__ __device__ float4x4 make_orthographic
(
	float width, 
	float height,
	float nearPlane,
	float farPlane
)
{
	float oneThroughFMinusN = 1.0f / (farPlane - nearPlane);

	float4x4 result = { 
		2.0f / width,	0.0f,		   0.0f,							0.0f,
		0.0f,			2.0f / height, 0.0f,							0.0f,
		0.0f,			0.0f,		   oneThroughFMinusN,				0.0f,
		0.0f,			0.0f,		   -nearPlane * oneThroughFMinusN,	1.0f
	};

	return result;
}

inline __host__ __device__ float4x4 make_perspective
( 
	float fov, 
	float ratio, 
	float nearPlane, 
	float farPlane 
)
{
	float cotAlpha2 = 1.f / tan( fov * .5f );	
	
	float fThroughFMinusN = farPlane / (farPlane - nearPlane);

	float4x4 result = { 
		cotAlpha2 / ratio,  0.f,		0.f,						  0.f,
		0.f,				cotAlpha2,  0.f,						  0.f,
		0.f,				0.f,		fThroughFMinusN,			  1.f,
		0.f,				0.f,		-fThroughFMinusN * nearPlane, 0.f 
	};

	return result;
}