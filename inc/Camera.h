#pragma once

#include <memory>

#include <vector_types.h>

#include "float4x4.h"

class Camera
{
public:

	Camera
	(
		float3 const & position, float3 const & lookAt,
		float fov, float aspectRatio,
		float nearPlane, float farPlane
	);

	// HACK for Light::camera()
	// TODO: Improve design
	Camera
	(
		float3 const & position, float3 const & lookAt,
		float4x4 projectionMatrix
	);

	float3 position() const;
	float4x4 viewMatrix() const;
	float4x4 projectionMatrix() const;
	float4x4 viewProjectionMatrix() const;

	void update( double msLastFrameTime );

	void handleMouseButtonPress( int button, int state, int x, int y );
	void handleMouseMovement( int newX, int newY );

private:

	float3 m_position;
	float3 m_lookAt;

	float4x4 m_projectionMatrix;

	// For handling GLUT events:
	int m_startX;
	int m_startY;
	int m_startZ;
	int m_endX;
	int m_endY;
	int m_endZ;

	bool m_button1Down;
	bool m_button2Down;
};