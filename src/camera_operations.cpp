#include "../inc/camera_operations.h"

#include <Windows.h>

#include <GL/freeglut.h>

#include <cutil.h>

#include "../inc/camera.h"
#include "../inc/glue.h"
#include "../inc/math3d.h"
#include "../inc/vector3.h"

static CUTBoolean initialized = CUTFalse;

static Camera _cam;
static Vector3 _pos;
static Vector3 _lookAt;
static float _fov;
static float _ratio;
static float _nearPlane;
static float _farPlane;

//mouse movement variables
static int _startX, _endX, _startY, _endY, _startZ, _endZ;
static CUTBoolean _button1Down, _button2Down;

static void updateMatrices(void);

void camInit(Vector3 pos, Vector3 lookAt,
			 float fov, float ratio, float nearPlane, float farPlane)
{
	if (!initialized)
	{
		initialized = CUTTrue;
		_pos = pos;
		_lookAt = lookAt;
		_fov = fov;
		_ratio = ratio;
		_nearPlane = nearPlane;
		_farPlane = farPlane;

		_cam.view = h_createCam(pos, lookAt, UNIT_Y);
		_cam.projection = h_createProjection(fov, ratio, nearPlane, farPlane);
		_cam.viewProjection = h_mMulM(_cam.view, _cam.projection);
		_cam.pos = pos;

		_startX = _endX = _startY = _endY = _startZ = _endZ = 0;
		_button1Down = CUTFalse;
		_button2Down = CUTFalse;
	}
}

Camera camGet(void)
{
	return _cam;
}

void mouseFunc(int button, int state, int x, int y)
{
	if (button == GLUT_LEFT_BUTTON)
	{
		if (state == GLUT_DOWN)
		{
			_button1Down = CUTTrue;
			_startX = _endX = x;
			_startY = _endY = y;
		}
		else if (state == GLUT_UP)
		{
			_button1Down = CUTFalse;
			_startX = _endX;
			_startY = _endY;
		}
	}
	if (button == GLUT_RIGHT_BUTTON)
	{
		if (state == GLUT_DOWN)
		{
			_button2Down = CUTTrue;
			_startZ = _endZ = y;
		}
		else if (state == GLUT_UP)
		{
			_button2Down = CUTFalse;
			_startZ = _endZ;
		}
	}
}

void motionFunc(int x, int y)
{
	if (_button1Down)
	{
		_endX = x;
		_endY = y;
	}
	else if (_button2Down)
	{
		_endZ = y;
	}
}

void camUpdate(int lastFrameTime)
{
	if (_button1Down)
	{
		Matrix horRot = h_createRotation(
			UNIT_Y, -((_endX - _startX) / ((float)glueGetWindowWidth())) * lastFrameTime * 0.01f
		);
		Matrix vertRot = h_createRotation(
			h_vecNormalize(h_vecCross(h_vecNormalize(h_vecSubVec(_lookAt, _pos)), UNIT_Y)),
			((_endY - _startY) / ((float)glueGetWindowHeight())) * lastFrameTime * 0.01f
		);
		_pos = h_vecSubVec(_pos, _lookAt);
		_pos = h_vecMulM(_pos, horRot);
		_pos = h_vecMulM(_pos, vertRot);
		_pos = h_vecAddVec(_pos, _lookAt);
		updateMatrices();
	}
	else if (_button2Down)
	{
		_pos = h_vecSubVec(_pos, _lookAt);
		_pos = h_vecAddVec(_pos, h_vecMulS(_pos, ((_endZ - _startZ) / ((float)glueGetWindowHeight())) * lastFrameTime * 0.01f));
		_pos = h_vecAddVec(_pos, _lookAt);
		updateMatrices();
	}
}

void updateMatrices(void)
{
	_cam.view = h_createCam(_pos, _lookAt, UNIT_Y);
	_cam.viewProjection = h_mMulM(_cam.view, _cam.projection);
	_cam.pos = _pos;
}