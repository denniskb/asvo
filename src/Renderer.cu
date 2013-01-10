#include "../inc/Renderer.h"

#include "../inc/bfsoctree_operations.h"
#include "../inc/light.h"

// Include the implementations of all math functions.
// CUDA requires that function declarations and definitions are
// in the same .cu file.
#include "math3d.cpp"

#define QUEUE_LENGTH 10000000
#define INT_MAX_VALUE 4294967295ul

static int _h_startIndex, _h_endIndex;
static int _h_level;

static __constant__ int _frame;

static __constant__ int _startIndex, _endIndex;
static __constant__ int _level;
static __device__ int _travQueuePtr;
static __device__ BFSJob _queue[QUEUE_LENGTH];

static texture<unsigned int, 1, cudaReadModeElementType> _depthBuffer;
static cudaChannelFormatDesc _depthBufferDesc;

static texture<uchar4, 2, cudaReadModeNormalizedFloat> _diffuse;
static texture<uchar4, 2, cudaReadModeNormalizedFloat> _illum;
static texture<uchar4, 2, cudaReadModeNormalizedFloat> _spec;
static texture<uchar4, 2, cudaReadModeNormalizedFloat> _normal;
static cudaChannelFormatDesc _mapDesc;

unsigned long int d_getChildCountFromMask( unsigned long int mask )
{
    return (   1ul & mask ) +
          ((   2ul & mask ) >> 1 ) +
          ((   4ul & mask ) >> 2 ) +
          ((   8ul & mask ) >> 3 ) +
          ((  16ul & mask ) >> 4 ) +
          ((  32ul & mask ) >> 5 ) +
          ((  64ul & mask ) >> 6 ) +
          (( 128ul & mask ) >> 7 );
}

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
static __device__ BFSJob jobInit
(
	unsigned long int index,
    unsigned short int x,
    unsigned short int y,
    unsigned short int z
)
{
	BFSJob result = { index, x, y, z };
	return result;
}

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
static __global__ void clearBuffers
(
	uchar4 * colorBuffer,
	int frameWidth, int frameHeight,

    unsigned short int jobCount,
	BFSJob * jobs,
	bool shadowPass
)
{
	unsigned long int threadIndex = blockDim.x * blockIdx.x + threadIdx.x;

	if (threadIndex < frameWidth * frameHeight)
	{
		if (shadowPass)
		{
			colorBuffer[threadIndex] = make_uchar4(51, 51, 51, 0);
		}

		if (threadIndex < jobCount)
		{
			_queue[threadIndex] = jobs[threadIndex];
		}
	}
}

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
static __global__ void traverse
(
	unsigned long int innerNodeCount,
    BFSInnerNode * innerNodes,
    BFSLeaf * leaves,
    float dimension,
    Matrix world, Vector3 camPos, Matrix view, Matrix projection,
    Matrix * animation, unsigned char boneCount,
    unsigned int * depthBuffer, VoxelData * voxelBuffer,
	int frameWidth, int frameHeight
)
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
			center.x = (center.x + 1.f) * frameWidth * 0.5f;
			center.y = frameHeight - (center.y + 1.f) * frameHeight * 0.5f;

			dimVec.x *= frameWidth;
			dimVec.y *= frameHeight;

			x = center.x;
			y = center.y;				

			x = max(x, 0);
			x = min(x, frameWidth - 1);
			y = max(y, 0);
			y = min(y, frameHeight - 1);		
								
			depth = center.z * INT_MAX_VALUE;
			index = x + y * frameWidth;

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
static __global__ void draw
(
	unsigned int * depthBuffer,
	uchar4 * colorBuffer,
	VoxelData * voxelBuffer,
	float * shadowMap,
	int frameWidth, int frameHeight,

    Vector3 light,
	Matrix lightWorldViewProjection,
	float diffusePower
)
{
	unsigned long int index = blockIdx.x * blockDim.x + threadIdx.x, index2, minDepth = INT_MAX_VALUE, depth;
	int startIndex, curIndex, x, y;
	VoxelData vd, minVd;

	if( index < frameWidth * frameHeight )
	{
		y = index / frameWidth;
		x = index - y * frameWidth;
		
		startIndex = index - frameWidth - 1;
		curIndex = 0;

#pragma unroll 9
		for (int i = 0; i < 9; ++i)
		{
			index2 = min(max(startIndex + curIndex, 0), frameWidth * frameHeight);
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
				startIndex += frameWidth;
			}
		}
		
		if (minDepth < INT_MAX_VALUE)
		{
			float4 color = make_float4(0.f, 0.f, 0.f, 0.f);			

			//shadow mapping
			minVd.center = d_vecMulM(minVd.center, lightWorldViewProjection);
			minVd.center.x = (minVd.center.x + 1.f) * frameWidth * 0.5f;
			minVd.center.y = frameHeight - (minVd.center.y + 1.f) * frameHeight * 0.5f;

			x = minVd.center.x;
			y = minVd.center.y;				

			x = max(x, 0);
			x = min(x, frameWidth - 1);
			y = max(y, 0);
			y = min(y, frameHeight - 1);
			
			index2 = x + y * frameWidth;			
			
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

/**
 * @see draw
 * Like draw but outputs a shadow map.
 *
 * @param depthBuffer The depth buffer.
 * @param shadowMap   The shadow map to output the data to.
 * @param voxelBuffer The voxel buffer.
 */
static __global__ void drawShadowMap
(
	unsigned int * depthBuffer,
	float * shadowMap,
	VoxelData * voxelBuffer,
	int frameWidth, int frameHeight
)
{
	unsigned long int index = blockIdx.x * blockDim.x + threadIdx.x, index2, minDepth = INT_MAX_VALUE, depth;
	int startIndex, curIndex, x, y;
	VoxelData vd;

	if (index < frameWidth * frameHeight)
	{
		y = index / frameWidth;
		x = index - y * frameWidth;
		
		startIndex = index - frameWidth - 1;
		curIndex = 0;

#pragma unroll 9
		for (int i = 0; i < 9; ++i)
		{
			index2 = min(max(startIndex + curIndex, 0), frameWidth * frameHeight);						
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
				startIndex += frameWidth;
			}
		}
		
		if (minDepth < INT_MAX_VALUE)
		{
			float color = ((float)minDepth) / ((float)INT_MAX_VALUE);
			shadowMap[index] = color;
		}
	}
}

Renderer::Renderer( int frameWidthInPixels, int frameHeightInPixels, bool shadowMapping ) :
	m_frameWidthInPixels( frameWidthInPixels ),
	m_frameHeightInPixels( frameHeightInPixels ),
	m_shadowMapping( shadowMapping )
{
	int frameResolution = frameWidthInPixels * frameHeightInPixels;

	m_depthBuffer.resize( frameResolution );
	m_voxelBuffer.resize( frameResolution );
	m_shadowMap.resize( frameResolution );

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



void Renderer::render
(
	Object3d & obj,
	Camera const & cam,

	uchar4 * outColorBuffer
)
{
	int frameResolution = m_frameWidthInPixels * m_frameHeightInPixels;

	int frame = BFSOctreeUpdate(&obj.data);
	cudaMemcpyToSymbol( _frame, &frame, sizeof(frame) );

	cudaThreadSynchronize();

	if( m_shadowMapping )
	{
		clearBuffers<<< nBlocks( frameResolution, nTHREADS_CLEAR_KERNEL ), nTHREADS_CLEAR_KERNEL >>>
		(
			outColorBuffer,
			m_frameWidthInPixels, m_frameHeightInPixels,
			obj.data.jobCount,
			obj.data.d_jobs,
			true
		);
		clearDepthBuffer();
		clearShadowMap();

		_h_startIndex = 0;
		_h_endIndex = obj.data.jobCount;
		_h_level = obj.data.level;
		rasterize
		( 
			obj, 
			lightGetCam(),
		
			true,

			outColorBuffer
		);
	}

	cudaThreadSynchronize();
	clearBuffers<<< nBlocks( frameResolution, nTHREADS_CLEAR_KERNEL ), nTHREADS_CLEAR_KERNEL >>>
	(
		outColorBuffer, 
		m_frameWidthInPixels, m_frameHeightInPixels,
		obj.data.jobCount, 
		obj.data.d_jobs, 
		false
	);
	clearDepthBuffer();

	_h_startIndex = 0;
	_h_endIndex = obj.data.jobCount;
	_h_level = obj.data.level;
	rasterize
	( 
		obj, 
		cam,
		
		false,

		outColorBuffer
	);
}



void Renderer::rasterize
(
	Object3d const & obj,
	Camera const & cam,

	bool shadowPass,
		
	uchar4 * outColorBuffer
)
{
	int frameResolution = m_frameWidthInPixels * m_frameHeightInPixels;

	cudaThreadSynchronize();
	cudaMemcpyToSymbol(_travQueuePtr, &_h_endIndex, sizeof(_h_endIndex));

	do
	{		
		cudaMemcpyToSymbol(_startIndex, &_h_startIndex, sizeof(_h_startIndex));
		cudaMemcpyToSymbol(_endIndex, &_h_endIndex, sizeof(_h_endIndex));
		cudaMemcpyToSymbol(_level, &_h_level, sizeof(_h_level));

		traverse<<< nBlocks( _h_endIndex - _h_startIndex, nTHREADS_TRAV_KERNEL ), nTHREADS_TRAV_KERNEL >>>
		(
			obj.data.innerNodeCount,
			obj.data.d_innerNodes,
			obj.data.d_leaves,
			obj.data.dim,
			obj.transform, cam.pos, cam.view, cam.projection,
			obj.data.d_animation, obj.data.boneCount,
			thrust::raw_pointer_cast( m_depthBuffer.data() ), thrust::raw_pointer_cast( m_voxelBuffer.data() ),
			m_frameWidthInPixels, m_frameHeightInPixels
		);
		
		_h_startIndex = _h_endIndex;		
		cudaMemcpyFromSymbol(&_h_endIndex, _travQueuePtr, sizeof(_h_endIndex));		
		++_h_level;
	}
	while (_h_endIndex - _h_startIndex > 0);
	
	cudaBindTexture((size_t*) 0, _depthBuffer, (void*) thrust::raw_pointer_cast( m_depthBuffer.data() ), _depthBufferDesc, (size_t) (frameResolution * sizeof(unsigned int)));
	if( shadowPass )
	{
		drawShadowMap<<< nBlocks( frameResolution, nTHREADS_DRAW_SHADOW_KERNEL ), nTHREADS_DRAW_SHADOW_KERNEL >>>
		(
			thrust::raw_pointer_cast( m_depthBuffer.data() ), 
			thrust::raw_pointer_cast( m_shadowMap.data() ), 
			thrust::raw_pointer_cast( m_voxelBuffer.data() ),
			m_frameWidthInPixels, m_frameHeightInPixels
		);
	}
	else
	{
		cudaBindTextureToArray(_diffuse, obj.data.diffuse.data, _mapDesc);
		cudaBindTextureToArray(_illum, obj.data.illum.data, _mapDesc);
		cudaBindTextureToArray(_spec, obj.data.spec.data, _mapDesc);
		cudaBindTextureToArray(_normal, obj.data.normal.data, _mapDesc);

		draw<<< nBlocks( frameResolution, nTHREADS_DRAW_KERNEL ), nTHREADS_DRAW_KERNEL >>>
		(
			thrust::raw_pointer_cast( m_depthBuffer.data() ),
			outColorBuffer,
			thrust::raw_pointer_cast( m_voxelBuffer.data() ),
			thrust::raw_pointer_cast( m_shadowMap.data() ),
			m_frameWidthInPixels, m_frameHeightInPixels,
			lightGetDir(),
			lightGetCam().viewProjection,
			lightGetDiffusePower()
		);

		cudaUnbindTexture( _diffuse );
		cudaUnbindTexture( _illum );
		cudaUnbindTexture( _spec );
		cudaUnbindTexture( _normal );
	}
	cudaUnbindTexture( _depthBuffer );
}



void Renderer::clearDepthBuffer()
{
	unsigned int const depthBufferClearValue = std::numeric_limits< unsigned int >::max();
	m_depthBuffer.assign( m_depthBuffer.size(), depthBufferClearValue );
}

void Renderer::clearShadowMap()
{
	float const shadowMapClearValue = 1;
	m_shadowMap.assign( m_shadowMap.size(), shadowMapClearValue );
}



// static
int Renderer::nBlocks( int nElements, int nThreadsPerBlock )
{
	int result = nElements / nThreadsPerBlock;
	return result + ( result * nThreadsPerBlock < nElements );
}