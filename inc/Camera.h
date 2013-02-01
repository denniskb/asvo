#pragma once

#include <memory>

#include "Matrix.h"
#include "vector3.h"

class Camera
{
public:

	Camera
	(
		Vector3 const & position, Vector3 const & lookAt,
		float fov, float aspectRatio,
		float nearPlane, float farPlane
	);

	// HACK for Light::camera()
	// TODO: Improve design
	Camera
	(
		Vector3 const & position, Vector3 const & lookAt,
		Matrix const & projectionMatrix
	);

	// HACK for GLUT so we can provide mouseFunc/motionFunc for GLUT
	// TODO: Improve design
	static void setGlobalCamera( Camera const & camera );
	static Camera & globalCamera();

	Vector3 position() const;
	Matrix viewMatrix() const;
	Matrix projectionMatrix() const;
	Matrix viewProjectionMatrix() const;

	// For GLUT
	static void mouseFunc( int button, int state, int x, int y );

	// For GLUT
	static void motionFunc( int x, int y );

	void update( double msLastFrameTime );

private:

	static std::unique_ptr< Camera > m_globalCamera;

	Vector3 m_position;
	Vector3 m_lookAt;

	Matrix m_projectionMatrix;

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