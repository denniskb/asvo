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

	// HACK for GLUT so we can provide mouseFunc/motionFunc for GLUT
	// TODO: Improve design
	static void setGlobalCamera( Camera const & camera );
	static Camera & globalCamera();

	float3 position() const;
	float4x4 viewMatrix() const;
	float4x4 projectionMatrix() const;
	float4x4 viewProjectionMatrix() const;

	// For GLUT
	static void mouseFunc( int button, int state, int x, int y );

	// For GLUT
	static void motionFunc( int x, int y );

	void update( double msLastFrameTime );

private:

	static std::unique_ptr< Camera > m_globalCamera;

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

	void handleMouseButtonPress( int button, int state, int x, int y );

	void handleMouseMovement( int newX, int newY );
};