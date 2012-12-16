#include "../inc/kernel_caller.h"
#include "../inc/object3d.h"
#include "../inc/voxeldata.h"
#include <cutil.h>
#include <cuda_runtime.h>
#include "../inc/expand_raster.h"
#include "../inc/camera_operations.h"
#include "../inc/math3d.h"
#include "../inc/glue.h"
#include "../inc/light.h"

// runtime kernel call parameters
static Object3d _obj;
static unsigned int *_depthBuffer;
static VoxelData *_voxelBuffer;
static CUTBoolean _parametersSet = CUTFalse;
static float *_shadowMap;

void kernelRun(uchar4 *colorBuffer)
{
	if (_parametersSet)
	{
		expandRasterInvoke(_depthBuffer, colorBuffer, _voxelBuffer, _obj, camGet(), _shadowMap, h_mMulM(_obj.transform, lightGetCam().viewProjection));
	}
}

void kernelSetParams(Object3d obj)
{
	if (!_parametersSet)
	{
		_parametersSet = CUTTrue;
		_obj = obj;
		cudaMalloc((void**)&_depthBuffer, glueGetWindowResolution() * sizeof(unsigned int));
		cudaMalloc((void**)&_voxelBuffer, glueGetWindowResolution() * sizeof(VoxelData));
		cudaMalloc((void**)&_shadowMap, glueGetWindowResolution() * sizeof(float));
		expandRasterInit();
	}
}

void kernelCleanup(void)
{
	if (_parametersSet)
	{
		_parametersSet = CUTFalse;
		cudaFree(_depthBuffer);
		cudaFree(_voxelBuffer);
		cudaFree(_shadowMap);
	}
}