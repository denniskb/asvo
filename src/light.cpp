#include "../inc/light.h"
#include "../inc/vector3.h"
#include "../inc/math3d.h"
#include "../inc/camera.h"
#include "../inc/camera_operations.h"

static Vector3 _light = UNIT_Y;
static float _diffuse = 0.f;
static float _ambient = 0.f;

void lightSet(Vector3 light, float diffusePower)
{
	Vector3 newLight = { -light.x, -light.y, -light.z };
	_light = h_vecNormalize(newLight);
	_diffuse = diffusePower;
	_ambient = 1.0f - diffusePower;
}

Camera lightGetCam(void)
{
	Camera result;

	result.pos = h_vecMulS(_light, 50.0f);
	result.view = h_createCam(result.pos, ZERO, UNIT_Y);
	result.projection = h_createOrthographic(100, 100, 10.f, 200.f);
	result.viewProjection = h_mMulM(result.view, result.projection);

	return result;
}

Vector3 lightGetDir(void)
{
	return _light;
}

float lightGetDiffusePower(void)
{
	return _diffuse;
}

float lightGetAmbientPower(void)
{
	return _ambient;
}