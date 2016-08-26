#include "Camera.h"

#include <GL/glew.h>
#include <GL/freeglut.h>

#include <vector_functions.h>
#include <vector_types.h>

#include "extended_helper_math.h"
#include "float4x4.h"
#include "Glue.h"



Camera::Camera
(
	float3 const & position, float3 const & lookAt,
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
	m_projectionMatrix = make_perspective( fov, aspectRatio, nearPlane, farPlane );
}

Camera::Camera
(
	float3 const & position, float3 const & lookAt,
	float4x4 projectionMatrix
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



float3 Camera::position() const
{
	return m_position;
}

float4x4 Camera::viewMatrix() const
{
	return make_lookat( m_position, m_lookAt, make_float3( 0, 1, 0 ) );
}

float4x4 Camera::projectionMatrix() const
{
	return m_projectionMatrix;
}

float4x4 Camera::viewProjectionMatrix() const
{
	return viewMatrix() * projectionMatrix();
}



void Camera::update
( 
	double msLastFrameTime,
	int windowWidthInPixels,
	int windowHeightInPixels
)
{
	if( m_button1Down )
	{
		float4x4 horRot = make_rotation(
			make_float3( 0, 1, 0 ),
			-( ( m_endX - m_startX ) / ( (double) windowWidthInPixels ) ) * msLastFrameTime * 0.01
		);
		float4x4 vertRot = make_rotation(
			normalize( cross( normalize( m_lookAt - m_position ), make_float3( 0, 1, 0 ) ) ),
			( ( m_endY - m_startY ) / ( (double) windowHeightInPixels ) ) * msLastFrameTime * 0.01
		);
		m_position += m_lookAt;
		m_position =  m_position * horRot;
		m_position =  m_position * vertRot;
		m_position -= m_lookAt;
	}
	else if( m_button2Down )
	{
		m_position -= m_lookAt;
		m_position += m_position * ( ( m_endZ - m_startZ ) / ( (float) windowHeightInPixels ) ) * msLastFrameTime * 0.01;
		m_position += m_lookAt;
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