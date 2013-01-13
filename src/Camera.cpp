#include "../inc/Camera.h"

#include <Windows.h>

#include <GL/freeglut.h>

#include "../inc/glue.h"
#include "../inc/math3d.h"

std::unique_ptr< Camera > Camera::m_globalCamera( nullptr );

// static 
void Camera::setGlobalCamera( Camera const & camera )
{
	m_globalCamera.reset( new Camera( camera ) );
}

// static 
Camera & Camera::globalCamera()
{
	return * m_globalCamera.get();
}



Camera::Camera
(
	Vector3 const & position, Vector3 const & lookAt,
    float fov, float aspectRatio,
	float nearPlane, float farPlane
) :
	m_position( position ),
	m_lookAt( lookAt ),

	m_startX( 0 ),
	m_startY( 0 ),
	m_startZ( 0 ),
	m_endX( 0 ),
	m_endY( 0 ),
	m_endZ( 0 ),

	m_button1Down( false ),
	m_button2Down( false )
{
	m_projectionMatrix = h_createProjection( fov, aspectRatio, nearPlane, farPlane );
}

Camera::Camera
(
	Vector3 const & position, Vector3 const & lookAt,
	Matrix const & projectionMatrix
) :
	m_position( position ),
	m_lookAt( lookAt ),

	m_projectionMatrix( projectionMatrix ),

	m_startX( 0 ),
	m_startY( 0 ),
	m_startZ( 0 ),
	m_endX( 0 ),
	m_endY( 0 ),
	m_endZ( 0 ),

	m_button1Down( false ),
	m_button2Down( false )
{

}



Vector3 Camera::position() const
{
	return m_position;
}

Matrix Camera::viewMatrix() const
{
	return h_createCam( m_position, m_lookAt, UNIT_Y );
}

Matrix Camera::projectionMatrix() const
{
	return m_projectionMatrix;
}

Matrix Camera::viewProjectionMatrix() const
{
	return h_mMulM( viewMatrix(), projectionMatrix() );
}




// static 
void Camera::mouseFunc( int button, int state, int x, int y )
{
	globalCamera().handleMouseButtonPress( button, state, x, y );
}

// static 
void Camera::motionFunc( int x, int y )
{
	globalCamera().handleMouseMovement( x, y );
}

void Camera::update( double msLastFrameTime )
{
	if( m_button1Down )
	{
		Matrix horRot = h_createRotation(
			UNIT_Y, -( ( m_endX - m_startX ) / ( (double) glueGetWindowWidth() ) ) * msLastFrameTime * 0.01
		);
		Matrix vertRot = h_createRotation(
			h_vecNormalize( h_vecCross( h_vecNormalize( h_vecSubVec( m_lookAt, m_position ) ), UNIT_Y ) ),
			( ( m_endY - m_startY ) / ( (double) glueGetWindowHeight() ) ) * msLastFrameTime * 0.01
		);
		m_position = h_vecSubVec( m_position, m_lookAt );
		m_position = h_vecMulM( m_position, horRot );
		m_position = h_vecMulM( m_position, vertRot );
		m_position = h_vecAddVec( m_position, m_lookAt );
	}
	else if( m_button2Down )
	{
		m_position = h_vecSubVec( m_position, m_lookAt );
		m_position = h_vecAddVec
		(
			m_position, 
			h_vecMulS( m_position, ( ( m_endZ - m_startZ) / ( (double) glueGetWindowHeight() ) ) * msLastFrameTime * 0.01 )
		);
		m_position = h_vecAddVec( m_position, m_lookAt);
	}
}



void Camera::handleMouseButtonPress( int button, int state, int x, int y )
{
	if( button == GLUT_LEFT_BUTTON )
	{
		if( state == GLUT_DOWN )
		{
			m_button1Down = true;
			m_startX = m_endX = x;
			m_startY = m_endY = y;
		}
		else if( state == GLUT_UP )
		{
			m_button1Down = false;
			m_startX = m_endX;
			m_startY = m_endY;
		}
	}

	if( button == GLUT_RIGHT_BUTTON )
	{
		if( state == GLUT_DOWN )
		{
			m_button2Down = true;
			m_startZ = m_endZ = y;
		}
		else if( state == GLUT_UP )
		{
			m_button2Down = false;
			m_startZ = m_endZ;
		}
	}
}

void Camera::handleMouseMovement( int newX, int newY )
{
	if( m_button1Down )
	{
		m_endX = newX;
		m_endY = newY;
	}
	else if( m_button2Down )
	{
		m_endZ = newY;
	}
}