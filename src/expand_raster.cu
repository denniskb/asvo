#include "../inc/expand_raster.h"

#include <cuda_runtime.H>
#include <cutil.h>

#include "../inc/bfsinnernode.h"
#include "../inc/bfsjob.h"
#include "../inc/bfsleaf.h"
#include "../inc/bfsoctree_operations.h"
#include "../inc/camera.h"
#include "../inc/glue.h"
#include "../inc/light.h"
#include "../inc/math3d.h"
#include "../inc/matrix.h"
#include "../inc/object3d.h"
#include "../inc/vector3.h"
#include "../inc/voxeldata.h"

/**
 * If SHADOW is defined, shadows are rendered using a shadow map.
 */
#define SHADOW

#define CLEAR_COUNT 192
#define TRAV_COUNT 128
#define DRAW_COUNT 128
#define DRAW_SHADOW_COUNT 192
#define QUEUE_LENGTH 10000000
#define INT_MAX_VALUE 4294967295ul

static unsigned long int _clearNumBlocks, _drawNumBlocks, _drawShadowNumBlocks;
static unsigned long int _h_startIndex, _h_endIndex;
static unsigned char _h_level;

static __constant__ int _windowWidth, _windowHeight;
static __constant__ int _windowResolution;

static __constant__ unsigned short int _frame;

static __constant__ unsigned long int _startIndex, _endIndex;
static __constant__ unsigned char _level;
static __device__ unsigned int _travQueuePtr;
static __device__ BFSJob _queue[QUEUE_LENGTH];

static texture<unsigned int, 1, cudaReadModeElementType> _depthBuffer;
static cudaChannelFormatDesc _depthBufferDesc;

static texture<uchar4, 2, cudaReadModeNormalizedFloat> _diffuse;
static texture<uchar4, 2, cudaReadModeNormalizedFloat> _illum;
static texture<uchar4, 2, cudaReadModeNormalizedFloat> _spec;
static texture<uchar4, 2, cudaReadModeNormalizedFloat> _normal;
static cudaChannelFormatDesc _mapDesc;

/**
 * Initializes a BFSJob.
 *
 * @param index The index of the voxel.
 * @param x     The x-coordinate of the voxel.
 * @param y     The y-coordinate of the voxel.
 * @param z     The z-coordinate of the voxel.
 *
 * @return A BFSJob representing the specified voxel that can
 *         be processed by the GPU.
 */
static __device__ BFSJob jobInit(unsigned long int index,
                                 unsigned short int x,
                                 unsigned short int y,
                                 unsigned short int z);

/**
 * Clears all buffers before rendering a new frame.
 *
 * @param depthBuffer The depth-buffer.
 * @param colorBuffer The color-buffer.
 * @param shadowMap   The shadow-map.
 * @param jobCount    Every BFSOctree stores a couple of BFSJobs in device memory as would
 *                    have been created by traversal without LOD. This is done
 *                    in order to avoid invoking the kernel on a nearly unfilled queue (=> low occupancy)
 *                    jobCount determines the number of these BFSJobs.
 * @param jobs        The job queue residing in device memory.
 * @param shadowPass  Determines whether the output of this pass is an image or a shadow map.
 */
static __global__ void clearBuffers(unsigned int *depthBuffer, uchar4 *colorBuffer, float *shadowMap,
                                    unsigned short int jobCount, BFSJob *jobs, CUTBoolean shadowPass);

/**
 * The main kernel responsible for rendering. Equivalent to the rasterizer plus vertex shader.
 * The kernel is invoked with as many threads as the job queue contains elements.
 * Every thread processes one job and either draws the voxel that blongs ot it or
 * decides that the voxel needs further subdivision and adds new jobs to the queue
 * (one for every child of the voxel).
 *
 * In the first case, the voxel isn't actually drawn but rather its depth and
 * visual data is stored in the depth map and voxel buffer at the coordinates that
 * correspond to the voxel's center. This is done because it is cheaper to scan multiple
 * neighboring pixels from the depth map for a voxel than it is to write a voxel's depth into
 * multiple pixels of the depth map (which requires multiple atomicMin calls).
 *
 * Drawing a voxel consists of projecting it into screen space and determining visibility
 * through a depth buffer. The actual coloring happens at a later stage.
 *
 * @param innerNodeCount The number of inner nodes of the octree that is traversed.
 * @param innerNodes     The inner nodes of the octree.
 * @param leaves         The leaf nodes of the octree.
 * @param dimension      The dimension of the octree (cube).
 * @param world          The world transformation of the model the octree represents.
 * @param view           The view transformation of the virtual camera.
 * @param projection     The projection transformation of the virtual camera.
 * @param animation      A set of matrices encoding the bone transformations for every frame.
 * @param boneCount      The number of bones of the model.
 * @param depthBuffer    The depth buffer.
 * @param voxelBuffer    The voxel buffer (think input to the pixel/fragment shader).
 */
static __global__ void traverse(unsigned long int innerNodeCount,
                                BFSInnerNode *innerNodes,
                                BFSLeaf *leaves,
                                float dimension,
                                Matrix world, Vector3 camPos, Matrix view, Matrix projection,
                                Matrix *animation, unsigned char boneCount,
                                unsigned int *depthBuffer, VoxelData *voxelBuffer);

/**
 * Draws an image of a rendered voxel model to the screen. For every pixel p visible
 * on the screen it scans a certain number of neighboring pixels in the depth map for a voxel and
 * selects the nearest voxel that covers p.
 *
 * This function could be easily implemented in the form of a shader by
 * storing the voxel data in a set of textures (one for every voxel property like
 * normals, texCoords, etc.) and send them to the GPU, which could combine this data with
 * triangle meshes (since one can output depth information in pixel/fragment shaders).
 *
 * @param depthBuffer              The depth buffer.
 * @param colorBuffer              The color buffer.
 * @param voxelBuffer              The voxel buffer.
 * @param shadowMap                The shadow map.
 * @param light                    The light direction.
 * @param lightWorldViewProjection light transform * model world transform * camera view transform * camera projection transform
 * @param diffusPower              The diffuse intensity of the light source.
 */
static __global__ void draw(unsigned int *depthBuffer, uchar4 *colorBuffer, VoxelData *voxelBuffer, float *shadowMap,
                            Vector3 light, Matrix lightWorldViewProjection, float diffusePower);

/**
 * @see draw
 * Like draw but outputs a shadow map.
 *
 * @param depthBuffer The depth buffer.
 * @param shadowMap   The shadow map to output the data to.
 * @param voxelBuffer The voxel buffer.
 */
static __global__ void drawShadowMap(unsigned int *depthBuffer, float *shadowMap, VoxelData *voxelBuffer);

/**
 * Encapsulates the whole render process including clearBuffers, traverse and draw.
 * Manages the job queue and adjusts execution configurations of kernels to maximize performance.
 *
 * @param depthBuffer              The depth buffer.
 * @param colorBuffer              The color buffer.
 * @param voxelBuffer              The voxel buffer.
 * @param obj                      The model to be rendered.
 * @param cam                      The virtual camera.
 * @param shadowPass               Determines whether the output of this pass is an image or a shadow map.
 * @param shadowMap                The shadow map.
 * @param lightWorldViewProjection light transform * model world transform * camera view transform * camera projection transform
 */
static __host__ void render(unsigned int *depthBuffer, uchar4 *colorBuffer, VoxelData *voxelBuffer,
                            Object3d obj, Camera cam, CUTBoolean shadowPass, float *shadowMap, Matrix lightWorldViewProjection);

__host__ void expandRasterInit(void)
{
	unsigned short int windowWidth = glueGetWindowWidth(), windowHeight = glueGetWindowHeight();
	unsigned long int windowResolution = glueGetWindowResolution();
	cudaMemcpyToSymbol("_windowWidth", &windowWidth, sizeof(windowWidth));
	cudaMemcpyToSymbol("_windowHeight", &windowHeight, sizeof(windowHeight));
	cudaMemcpyToSymbol("_windowResolution", &windowResolution, sizeof(windowResolution));

	_clearNumBlocks = glueGetWindowResolution() / CLEAR_COUNT + 1;
	_drawNumBlocks = glueGetWindowResolution() / DRAW_COUNT + 1;
	_drawShadowNumBlocks = glueGetWindowResolution() / DRAW_SHADOW_COUNT + 1;

	_depthBufferDesc = cudaCreateChannelDesc<unsigned int>();

	_mapDesc = cudaCreateChannelDesc<uchar4>();
	_diffuse.normalized = true;
	_diffuse.filterMode = cudaFilterModeLinear;
	_diffuse.addressMode[0] = _diffuse.addressMode[1] = cudaAddressModeWrap;

	_illum.normalized = true;
	_illum.filterMode = cudaFilterModeLinear;
	_illum.addressMode[0] = _illum.addressMode[1] = cudaAddressModeWrap;

	_spec.normalized = true;
	_spec.filterMode = cudaFilterModeLinear;
	_spec.addressMode[0] = _spec.addressMode[1] = cudaAddressModeWrap;

	_normal.normalized = true;
	_normal.filterMode = cudaFilterModeLinear;
	_normal.addressMode[0] = _normal.addressMode[1] = cudaAddressModeWrap;
}

__host__ void expandRasterInvoke(unsigned int *depthBuffer, uchar4 *colorBuffer, VoxelData *voxelBuffer,
								 Object3d obj, Camera cam, float *shadowMap, Matrix lightWorldViewProjection)
{
	unsigned short int frame = BFSOctreeUpdate(&obj.data);
	cudaMemcpyToSymbol("_frame", &frame, sizeof(frame));

	cudaThreadSynchronize();
	clearBuffers<<<_clearNumBlocks, CLEAR_COUNT>>>(
		depthBuffer, colorBuffer, shadowMap, obj.data.jobCount, obj.data.d_jobs, CUTTrue
	);

#ifdef SHADOW
	_h_startIndex = 0;
	_h_endIndex = obj.data.jobCount;
	_h_level = obj.data.level;
	render(depthBuffer, colorBuffer, voxelBuffer, obj, lightGetCam(), CUTTrue, shadowMap, lightWorldViewProjection);
#endif

	cudaThreadSynchronize();
	clearBuffers<<<_clearNumBlocks, CLEAR_COUNT>>>(
		depthBuffer, colorBuffer, shadowMap, obj.data.jobCount, obj.data.d_jobs, CUTFalse
	);

	_h_startIndex = 0;
	_h_endIndex = obj.data.jobCount;
	_h_level = obj.data.level;
	render(depthBuffer, colorBuffer, voxelBuffer, obj, cam, CUTFalse, shadowMap, lightWorldViewProjection);
}

void render(unsigned int *depthBuffer, uchar4 *colorBuffer, VoxelData *voxelBuffer,
            Object3d obj, Camera cam, CUTBoolean shadowPass, float *shadowMap, Matrix lightWorldViewProjection)
{
	cudaThreadSynchronize();
	cudaMemcpyToSymbol("_travQueuePtr", &_h_endIndex, sizeof(_h_endIndex));
	do
	{		
		cudaMemcpyToSymbol("_startIndex", &_h_startIndex, sizeof(_h_startIndex));
		cudaMemcpyToSymbol("_endIndex", &_h_endIndex, sizeof(_h_endIndex));
		cudaMemcpyToSymbol("_level", &_h_level, sizeof(_h_level));

		traverse<<<(_h_endIndex - _h_startIndex) / TRAV_COUNT + 1, TRAV_COUNT>>>(
			obj.data.innerNodeCount,
			obj.data.d_innerNodes,
			obj.data.d_leaves,
			obj.data.dim,
			obj.transform, cam.pos, cam.view, cam.projection,
			obj.data.d_animation, obj.data.boneCount,
			depthBuffer, voxelBuffer
		);
		
		_h_startIndex = _h_endIndex;		
		cudaMemcpyFromSymbol(&_h_endIndex, "_travQueuePtr", sizeof(_h_endIndex));		
		++_h_level;
	}
	while (_h_endIndex - _h_startIndex > 0);
	
	cudaBindTexture((size_t*) 0, _depthBuffer, (void*) depthBuffer, _depthBufferDesc, (size_t) (glueGetWindowResolution() * sizeof(unsigned int)));
	if (shadowPass)
		drawShadowMap<<<_drawShadowNumBlocks, DRAW_SHADOW_COUNT>>>(depthBuffer, shadowMap, voxelBuffer);
	else
	{
		cudaBindTextureToArray(_diffuse, obj.data.diffuse.data, _mapDesc);
		cudaBindTextureToArray(_illum, obj.data.illum.data, _mapDesc);
		cudaBindTextureToArray(_spec, obj.data.spec.data, _mapDesc);
		cudaBindTextureToArray(_normal, obj.data.normal.data, _mapDesc);

		draw<<<_drawNumBlocks, DRAW_COUNT>>>(
			depthBuffer, colorBuffer, voxelBuffer, shadowMap, lightGetDir(), lightWorldViewProjection, lightGetDiffusePower()
		);

		cudaUnbindTexture(_diffuse);
		cudaUnbindTexture(_illum);
		cudaUnbindTexture(_spec);
		cudaUnbindTexture(_normal);
	}
	cudaUnbindTexture(_depthBuffer);
}

void clearBuffers(unsigned int *depthBuffer, uchar4 *colorBuffer, float *shadowMap,
                  unsigned short int jobCount, BFSJob *jobs, CUTBoolean shadowPass)
{
	unsigned long int threadIndex = blockDim.x * blockIdx.x + threadIdx.x;

	if (threadIndex < _windowResolution)
	{
		if (shadowPass)
		{
			depthBuffer[threadIndex] = INT_MAX_VALUE;
			colorBuffer[threadIndex] = make_uchar4(51, 51, 51, 0);
			shadowMap[threadIndex] = 1.f;
		}
		else
			depthBuffer[threadIndex] = INT_MAX_VALUE;

		if (threadIndex < jobCount)
		{
			_queue[threadIndex] = jobs[threadIndex];
		}
	}
}

void traverse(unsigned long int innerNodeCount,
              BFSInnerNode *innerNodes,
              BFSLeaf *leaves,
              float dimension,
              Matrix world, Vector3 camPos, Matrix view, Matrix projection,
              Matrix *animation, unsigned char boneCount,
              unsigned int *depthBuffer, VoxelData *voxelBuffer)
{
	unsigned long int index = blockDim.x * blockIdx.x + threadIdx.x + _startIndex;	
	short int x, y, z, w = 2;
	unsigned int depth;

	__shared__ unsigned int sharedTravQueuePtr;

	if (threadIdx.x == 0)
		sharedTravQueuePtr = 0;

	__syncthreads();

	if (index < _endIndex)
	{
		BFSJob job = _queue[index];
		BFSInnerNode node;
		node.childPtr = 0;
		node.mask = 0;
		
		if (job.index < innerNodeCount)
			node = innerNodes[job.index];
		else
			node.vd = leaves[job.index - innerNodeCount].vd;
			
		float gridCellDim = dimension / ((float)(1 << _level));
		float gridCellHalfDim = gridCellDim * 0.5f;
		
		float minCoord = -0.5f * dimension;
		Vector3 center = { fmaf(job.x, gridCellDim, minCoord + gridCellHalfDim),
					       fmaf(job.y, gridCellDim, minCoord + gridCellHalfDim),
						   fmaf(job.z, gridCellDim, minCoord + gridCellHalfDim) };		
		
		Vector3 skinnedCenter = ZERO;
		unsigned char involvedBones = 0;
        if (node.vd.boneWeights.x > 0.f)
		{
			skinnedCenter = d_vecMulS(d_vecMulM(center, animation[_frame*boneCount+node.vd.boneIndex0]),
                                      node.vd.boneWeights.x);
			++involvedBones;
		}
		if (node.vd.boneWeights.y > 0.f)
		{
			skinnedCenter = d_vecAddVec(skinnedCenter, 
                                        d_vecMulS(d_vecMulM(center, animation[_frame*boneCount+node.vd.boneIndex1]),
									    node.vd.boneWeights.y));
			++involvedBones;
		}
		if (node.vd.boneWeights.z > 0.f)
		{
			skinnedCenter = d_vecAddVec(skinnedCenter,
					                    d_vecMulS(d_vecMulM(center, animation[_frame*boneCount+node.vd.boneIndex2]),
						                node.vd.boneWeights.z));
			++involvedBones;
		}
		if (node.vd.boneWeights.w > 0.f)
		{
			skinnedCenter = d_vecAddVec(skinnedCenter, 
									    d_vecMulS(d_vecMulM(center, animation[_frame*boneCount+node.vd.boneIndex3]),
									    node.vd.boneWeights.w));
			++involvedBones;
		}
		center = skinnedCenter;
		Vector3 originalCenter = center;

		// resizing of voxel to counter stretching.
		gridCellHalfDim *= fminf(2.f, involvedBones);

		center = d_vecMulM(center, world);
		Vector3 eyeVec = d_vecNormalize(d_vecSubVec(camPos, center));

		center = d_vecMulM(center, view);
		Vector3 dimVec = { gridCellHalfDim, gridCellHalfDim, center.z };
		
		center = d_vecMulM(center, projection);
		dimVec = d_vecMulM(dimVec, projection);
							
		//viewing frustum + clockwise culling
		node.vd.normal = d_vecMulM(node.vd.normal, world);
		if (_travQueuePtr < QUEUE_LENGTH - 100000 &&
			-1.f <= center.x + dimVec.x && center.x - dimVec.x <= 1.f &&
			-1.f <= center.y + dimVec.x && center.y - dimVec.x <= 1.f &&
			 0.f <= center.z + dimVec.x && center.z - dimVec.x <= 1.f &&
			 (_level <= 8 || d_vecDot(node.vd.normal, eyeVec) >= -0.4f))
		{	
			center.x = (center.x + 1.f) * _windowWidth * 0.5f;
			center.y = _windowHeight - (center.y + 1.f) * _windowHeight * 0.5f;

			dimVec.x *= _windowWidth;
			dimVec.y *= _windowHeight;

			x = center.x;
			y = center.y;				

			x = max(x, 0);
			x = min(x, _windowWidth - 1);
			y = max(y, 0);
			y = min(y, _windowHeight - 1);		
								
			depth = center.z * INT_MAX_VALUE;
			index = x + y * _windowWidth;

			if ((dimVec.x > 1.f) && (z = d_getChildCountFromMask(node.mask)) != 0)
			{	
				index = atomicAdd(&sharedTravQueuePtr, z);
				w = 0;
			}
			else
			{
				atomicMin(depthBuffer + index, depth);
				w = 1;
			}
		}
		
		__syncthreads();
		if (threadIdx.x == 0)
			sharedTravQueuePtr = atomicAdd(&_travQueuePtr, sharedTravQueuePtr);

		__syncthreads();
			
		if (w == 0)
		{
			index += sharedTravQueuePtr;

#pragma unroll 8
			for (w = 0; w < 8; ++w)
			{
				if ((node.mask & (1u << w)) != 0)
				{
					x = 2 * job.x + (w & 1u);
					y = 2 * job.y + ((w & 2u) >> 1);
					z = 2 * job.z + ((w & 4u) >> 2);

					_queue[index++] = jobInit(node.childPtr++,
											  x, y, z);
				}
			}				
		}
		else if (w == 1 && depth == depthBuffer[index])
		{
			VoxelData vd = { node.vd.normal,
							 d_vecMulM(node.vd.tangent, world),
							 center.x,
							 center.y,
							 dimVec.x,
							 dimVec.y,
							 node.vd.texCoord,
							 eyeVec,
							 originalCenter };

			voxelBuffer[index] = vd;		
		}
	}
}

void draw(unsigned int *depthBuffer, uchar4 *colorBuffer, VoxelData *voxelBuffer, float *shadowMap,
          Vector3 light, Matrix lightWorldViewProjection, float diffusePower)
{
	unsigned long int index = blockIdx.x * blockDim.x + threadIdx.x, index2, minDepth = INT_MAX_VALUE, depth;
	int startIndex, curIndex, x, y;
	VoxelData vd, minVd;

	if (index < _windowResolution)
	{
		y = index / _windowWidth;
		x = index - y * _windowWidth;
		
		startIndex = index - _windowWidth - 1;
		curIndex = 0;

#pragma unroll 9
		for (int i = 0; i < 9; ++i)
		{
			index2 = min(max(startIndex + curIndex, 0), _windowResolution);
			if ((depth = tex1Dfetch(_depthBuffer, index2)) < minDepth)
			{		
				vd = voxelBuffer[index2];				
				if (fabsf(vd.pos.x - .5f - x) <= vd.pos.z)			
					if (fabsf(vd.pos.y - .5f - y) <= vd.pos.w)
					{
						minDepth = depth;	
						minVd = vd;
					}			
			}

			if (++curIndex == 3)
			{
				curIndex = 0;
				startIndex += _windowWidth;
			}
		}
		
		if (minDepth < INT_MAX_VALUE)
		{
			float4 color = make_float4(0.f, 0.f, 0.f, 0.f);			

			//shadow mapping
			minVd.center = d_vecMulM(minVd.center, lightWorldViewProjection);
			minVd.center.x = (minVd.center.x + 1.f) * _windowWidth * 0.5f;
			minVd.center.y = _windowHeight - (minVd.center.y + 1.f) * _windowHeight * 0.5f;

			x = minVd.center.x;
			y = minVd.center.y;				

			x = max(x, 0);
			x = min(x, _windowWidth - 1);
			y = max(y, 0);
			y = min(y, _windowHeight - 1);
			
			index2 = x + y * _windowWidth;			
			
			float4 tempf;
			if (minVd.center.z <= shadowMap[index2] + 0.01f) //not in shadow		
			{
				//normal mapping
				tempf = tex2D(_normal, minVd.texCoord.x, minVd.texCoord.y);
				Vector3 tempv = d_vecCross(minVd.normal, minVd.tangent);
				minVd.normal = d_vecNormalize(d_vecAddVec(d_vecMulS(minVd.normal, tempf.z),
											  d_vecAddVec(d_vecMulS(minVd.tangent, tempf.x),
														  d_vecMulS(tempv, tempf.y))));
				
				tempf = tex2D(_illum, minVd.texCoord.x, minVd.texCoord.y);				
				float intensity = fmaxf(0.f, d_vecDot(minVd.normal, light));				
				if (intensity > 0.f || diffusePower < 1.f)
				{					
					color = tex2D(_diffuse, minVd.texCoord.x, minVd.texCoord.y);
					color.x *= intensity * diffusePower + tempf.x + 1.0f - diffusePower;
					color.y *= intensity * diffusePower + tempf.y + 1.0f - diffusePower;
					color.z *= intensity * diffusePower + tempf.z + 1.0f - diffusePower;
				}

				tempv = d_vecNormalize(d_vecAddVec(d_vecMulS(minVd.eyeVec, 0.5f), d_vecMulS(light, 0.5f)));				
				intensity = powf(fmaxf(0.f, d_vecDot(tempv, minVd.normal)), 32.f); 
				if (intensity > 0.f)
				{
					tempf = tex2D(_spec, minVd.texCoord.x, minVd.texCoord.y);
					color.x += diffusePower * intensity * tempf.x;
					color.y += diffusePower * intensity * tempf.y;
					color.z += diffusePower * intensity * tempf.z;
				}
			}
			else
			{
				color = tex2D(_diffuse, minVd.texCoord.x, minVd.texCoord.y);
				tempf = tex2D(_illum, minVd.texCoord.x, minVd.texCoord.y);				
				color.x *= tempf.x + 1.0f - diffusePower;
				color.y *= tempf.y + 1.0f - diffusePower;
				color.z *= tempf.z + 1.0f - diffusePower;
			}
					
			color.x = fminf(1.f, color.x);
			color.y = fminf(1.f, color.y);
			color.z = fminf(1.f, color.z);
			
			colorBuffer[index] = make_uchar4(color.x * 255.f, color.y * 255.f, color.z * 255.f, 0);
		}
	}
}

static __global__ void drawShadowMap(unsigned int *depthBuffer, float *shadowMap, VoxelData *voxelBuffer)
{
	unsigned long int index = blockIdx.x * blockDim.x + threadIdx.x, index2, minDepth = INT_MAX_VALUE, depth;
	int startIndex, curIndex, x, y;
	VoxelData vd;

	if (index < _windowResolution)
	{
		y = index / _windowWidth;
		x = index - y * _windowWidth;
		
		startIndex = index - _windowWidth - 1;
		curIndex = 0;

#pragma unroll 9
		for (int i = 0; i < 9; ++i)
		{
			index2 = min(max(startIndex + curIndex, 0), _windowResolution);						
			if ((depth = tex1Dfetch(_depthBuffer, index2)) < minDepth)
			{		
				vd = voxelBuffer[index2];				
				if (fabsf(vd.pos.x - .5f - x) <= vd.pos.z)				
					if (fabsf(vd.pos.y - .5f - y) <= vd.pos.w)					
						minDepth = depth;				
			}

			if (++curIndex == 3)
			{
				curIndex = 0;
				startIndex += _windowWidth;
			}
		}
		
		if (minDepth < INT_MAX_VALUE)
		{
			float color = ((float)minDepth) / ((float)INT_MAX_VALUE);
			shadowMap[index] = color;
		}
	}
}

BFSJob jobInit(unsigned long int index,
               unsigned short int x,
               unsigned short int y,
               unsigned short int z)
{
	BFSJob result = { index, x, y, z };
	return result;
}

unsigned long int d_getChildCountFromMask(unsigned long int mask)
{
    return (1ul & mask) +
          ((2ul & mask) >> 1) +
          ((4ul & mask) >> 2) +
          ((8ul & mask) >> 3) +
          ((16ul & mask) >> 4) +
          ((32ul & mask) >> 5) +
          ((64ul & mask) >> 6) +
          ((128ul & mask) >> 7);
}

// Include the implementations of all math functions.
// CUDA requires that function declarations and definitions are
// in the same .cu file.
#include "math3d.cpp"