#include "../inc/Renderer.h"

#include <cstdint>

#include "../inc/BFSJob.cuh"
#include "../inc/bfsoctree_operations.h"
#include "../inc/Light.h"

// Include the implementations of all math functions.
// CUDA requires that function declarations and definitions are
// in the same .cu file.
#include "math3d.cpp"

/* Textures */
static texture< unsigned, cudaTextureType1D, cudaReadModeElementType > tDepthBuffer;
static texture< uchar4, cudaTextureType2D, cudaReadModeNormalizedFloat > tDiffuse;
static texture< uchar4, cudaTextureType2D, cudaReadModeNormalizedFloat > tIllum;
static texture< uchar4, cudaTextureType2D, cudaReadModeNormalizedFloat > tSpec;
static texture< uchar4, cudaTextureType2D, cudaReadModeNormalizedFloat > tNormal;

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
	int frameWidth, int frameHeight,
	int animationFrameIndex,
	int octreeLevel,
	// TODO: Rename
	int const * startIndex, int const * endIndex,
	int * travQueuePtr,
	BFSJob * jobQueue
)
{
	unsigned long int index = blockDim.x * blockIdx.x + threadIdx.x + ( * startIndex );	
	short int x, y, z, w = 2;
	unsigned int depth;

	__shared__ unsigned int sharedTravQueuePtr;

	if (threadIdx.x == 0)
		sharedTravQueuePtr = 0;

	__syncthreads();

	if( index < ( * endIndex ) )
	{
		BFSJob job = jobQueue[ index ];
		BFSInnerNode node;
		node.childPtr = 0;
		node.mask = 0;
		
		if (job.index < innerNodeCount)
			node = innerNodes[job.index];
		else
			node.vd = leaves[job.index - innerNodeCount].vd;
			
		float gridCellDim = dimension / ( (float) ( 1 << octreeLevel ) );
		float gridCellHalfDim = gridCellDim * 0.5f;
		
		float minCoord = -0.5f * dimension;
		Vector3 center = { fmaf(job.x, gridCellDim, minCoord + gridCellHalfDim),
					       fmaf(job.y, gridCellDim, minCoord + gridCellHalfDim),
						   fmaf(job.z, gridCellDim, minCoord + gridCellHalfDim) };		
		
		Vector3 skinnedCenter = ZERO;
		unsigned char involvedBones = 0;
        if (node.vd.boneWeights.x > 0.f)
		{
			skinnedCenter = d_vecMulS
			(
				d_vecMulM(center, animation[ animationFrameIndex * boneCount+node.vd.boneIndex0 ]),
				node.vd.boneWeights.x
			);
			++involvedBones;
		}
		if (node.vd.boneWeights.y > 0.f)
		{
			skinnedCenter = d_vecAddVec( skinnedCenter, d_vecMulS(
				d_vecMulM(center, animation[ animationFrameIndex * boneCount+node.vd.boneIndex1 ]), node.vd.boneWeights.y
			));
			++involvedBones;
		}
		if (node.vd.boneWeights.z > 0.f)
		{
			skinnedCenter = d_vecAddVec( skinnedCenter, d_vecMulS(
				d_vecMulM(center, animation[ animationFrameIndex * boneCount+node.vd.boneIndex2 ]), node.vd.boneWeights.z
			));
			++involvedBones;
		}
		if (node.vd.boneWeights.w > 0.f)
		{
			skinnedCenter = d_vecAddVec(skinnedCenter, d_vecMulS(
				d_vecMulM(center, animation[ animationFrameIndex * boneCount+node.vd.boneIndex3 ]), node.vd.boneWeights.w
			));
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
		// TODO: Add check whether adding the current node's children
		// to the job queue would exceed the queue's size limit
		if ( -1.f <= center.x + dimVec.x && center.x - dimVec.x <= 1.f &&
			-1.f <= center.y + dimVec.x && center.y - dimVec.x <= 1.f &&
			 0.f <= center.z + dimVec.x && center.z - dimVec.x <= 1.f &&
			 ( octreeLevel <= 8 || d_vecDot( node.vd.normal, eyeVec ) >= -0.4f ))
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
								
			depth = center.z * INT_MAX;
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
		if( threadIdx.x == 0 )
		{
			sharedTravQueuePtr = atomicAdd( travQueuePtr, sharedTravQueuePtr );
		}

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

					jobQueue[ index++ ] = dmake_BFSJob( node.childPtr++, x, y, z );
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
	unsigned index = blockIdx.x * blockDim.x + threadIdx.x, index2;
	unsigned minDepth = INT_MAX;
	unsigned depth;
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
			if ((depth = tex1Dfetch(tDepthBuffer, index2)) < minDepth)
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
		
		if (minDepth < INT_MAX)
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
				tempf = tex2D(tNormal, minVd.texCoord.x, minVd.texCoord.y);
				Vector3 tempv = d_vecCross(minVd.normal, minVd.tangent);
				minVd.normal = d_vecNormalize(d_vecAddVec(d_vecMulS(minVd.normal, tempf.z),
											  d_vecAddVec(d_vecMulS(minVd.tangent, tempf.x),
														  d_vecMulS(tempv, tempf.y))));
				
				tempf = tex2D(tIllum, minVd.texCoord.x, minVd.texCoord.y);				
				float intensity = fmaxf(0.f, d_vecDot(minVd.normal, light));				
				if (intensity > 0.f || diffusePower < 1.f)
				{					
					color = tex2D(tDiffuse, minVd.texCoord.x, minVd.texCoord.y);
					color.x *= intensity * diffusePower + tempf.x + 1.0f - diffusePower;
					color.y *= intensity * diffusePower + tempf.y + 1.0f - diffusePower;
					color.z *= intensity * diffusePower + tempf.z + 1.0f - diffusePower;
				}

				tempv = d_vecNormalize(d_vecAddVec(d_vecMulS(minVd.eyeVec, 0.5f), d_vecMulS(light, 0.5f)));				
				intensity = powf(fmaxf(0.f, d_vecDot(tempv, minVd.normal)), 32.f); 
				if (intensity > 0.f)
				{
					tempf = tex2D(tSpec, minVd.texCoord.x, minVd.texCoord.y);
					color.x += diffusePower * intensity * tempf.x;
					color.y += diffusePower * intensity * tempf.y;
					color.z += diffusePower * intensity * tempf.z;
				}
			}
			else
			{
				color = tex2D(tDiffuse, minVd.texCoord.x, minVd.texCoord.y);
				tempf = tex2D(tIllum, minVd.texCoord.x, minVd.texCoord.y);				
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
	unsigned int index = blockIdx.x * blockDim.x + threadIdx.x, index2, minDepth = INT_MAX, depth;
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
			if ((depth = tex1Dfetch(tDepthBuffer, index2)) < minDepth)
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
		
		if (minDepth < INT_MAX)
		{
			float color = ((float)minDepth) / ((float)INT_MAX);
			shadowMap[index] = color;
		}
	}
}

Renderer::Renderer( int frameWidthInPixels, int frameHeightInPixels, bool shadowMapping ) :
	m_frameWidth( frameWidthInPixels ),
	m_frameHeight( frameHeightInPixels ),
	m_shadowMapping( shadowMapping )
{
	// TODO: Dynamically resize queue if it gets too small instead of
	// using a big conservative value
	m_dJobQueue.resize( 10000000 );

	m_dDepthBuffer.resize( resolution() );
	m_dVoxelBuffer.resize( resolution() );
	m_dShadowMap.resize( resolution() );

	tDiffuse.normalized = true;
	tDiffuse.filterMode = cudaFilterModeLinear;
	tDiffuse.addressMode[0] = tDiffuse.addressMode[1] = cudaAddressModeWrap;

	tIllum.normalized = true;
	tIllum.filterMode = cudaFilterModeLinear;
	tIllum.addressMode[0] = tIllum.addressMode[1] = cudaAddressModeWrap;

	tSpec.normalized = true;
	tSpec.filterMode = cudaFilterModeLinear;
	tSpec.addressMode[0] = tSpec.addressMode[1] = cudaAddressModeWrap;

	tNormal.normalized = true;
	tNormal.filterMode = cudaFilterModeLinear;
	tNormal.addressMode[0] = tNormal.addressMode[1] = cudaAddressModeWrap;
}



void Renderer::render
(
	Object3d & obj,
	Camera const & cam,
	Light const & light,

	uchar4 * outColorBuffer
)
{
	int animationFrameIndex = BFSOctreeUpdate( & obj.data );

	if( m_shadowMapping )
	{
		fillJobQueue( thrust::raw_pointer_cast( obj.data.d_jobs->data() ), obj.data.jobCount );
		clearDepthBuffer();
		clearShadowMap();

		rasterize
		( 
			obj, 
			light.camera(),
			light,
			animationFrameIndex,
		
			true,

			outColorBuffer
		);
	}

	fillJobQueue( thrust::raw_pointer_cast( obj.data.d_jobs->data() ), obj.data.jobCount );
	clearColorBuffer( outColorBuffer );
	clearDepthBuffer();
	if( ! m_shadowMapping )
	{
		clearShadowMap();
	}

	rasterize
	( 
		obj, 
		cam,
		light,
		animationFrameIndex,
		
		false,

		outColorBuffer
	);
}



void Renderer::rasterize
(
	Object3d const & obj,
	Camera const & cam,
	Light const & light,
	int animationFrameIndex,

	bool shadowPass,
		
	uchar4 * outColorBuffer
)
{
	int hStartIndex = 0;
	int hEndIndex = obj.data.jobCount;

	// TODO: Extract into DeviceQueue class
	thrust::device_vector< int > dStartIndex( 1 );
	thrust::device_vector< int > dEndIndex( 1 );
	thrust::device_vector< int > dTravQueuePtr( 1 );

	dTravQueuePtr[ 0 ] = hEndIndex;

	int octreeLevel = obj.data.level;
	do
	{		
		dStartIndex[ 0 ] = hStartIndex;
		dEndIndex[ 0 ] = hEndIndex;

		traverse<<< nBlocks( hEndIndex - hStartIndex, nTHREADS_TRAV_KERNEL ), nTHREADS_TRAV_KERNEL >>>
		(
			obj.data.innerNodeCount,
			obj.data.d_innerNodes,
			obj.data.d_leaves,
			obj.data.dim,
			obj.transform, cam.position(), cam.viewMatrix(), cam.projectionMatrix(),
			obj.data.d_animation, obj.data.boneCount,
			thrust::raw_pointer_cast( m_dDepthBuffer.data() ), thrust::raw_pointer_cast( m_dVoxelBuffer.data() ),
			m_frameWidth, m_frameHeight,
			animationFrameIndex,
			octreeLevel,
			thrust::raw_pointer_cast( dStartIndex.data() ), thrust::raw_pointer_cast( dEndIndex.data() ),
			thrust::raw_pointer_cast( dTravQueuePtr.data() ),
			thrust::raw_pointer_cast( m_dJobQueue.data() )
		);
		
		hStartIndex = hEndIndex;		
		hEndIndex = dTravQueuePtr[ 0 ];
		octreeLevel++;
	}
	while( hEndIndex - hStartIndex > 0 );
	
	cudaBindTexture
	(
		(size_t *) 0,
		tDepthBuffer,
		(void *) thrust::raw_pointer_cast( m_dDepthBuffer.data() ),
		cudaCreateChannelDesc< unsigned >(),
		(size_t) ( resolution() * sizeof( unsigned int ) )
	);
	if( shadowPass )
	{
		drawShadowMap<<< nBlocks( resolution(), nTHREADS_DRAW_SHADOW_KERNEL ), nTHREADS_DRAW_SHADOW_KERNEL >>>
		(
			thrust::raw_pointer_cast( m_dDepthBuffer.data() ), 
			thrust::raw_pointer_cast( m_dShadowMap.data() ), 
			thrust::raw_pointer_cast( m_dVoxelBuffer.data() ),
			m_frameWidth, m_frameHeight
		);
	}
	else
	{
		cudaBindTextureToArray( tDiffuse, obj.data.diffuse->data(), cudaCreateChannelDesc< uchar4 >() );
		cudaBindTextureToArray( tIllum, obj.data.illum->data(), cudaCreateChannelDesc< uchar4 >() );
		cudaBindTextureToArray( tSpec, obj.data.spec->data(), cudaCreateChannelDesc< uchar4 >() );
		cudaBindTextureToArray( tNormal, obj.data.normal->data(), cudaCreateChannelDesc< uchar4 >() );

		draw<<< nBlocks( resolution(), nTHREADS_DRAW_KERNEL ), nTHREADS_DRAW_KERNEL >>>
		(
			thrust::raw_pointer_cast( m_dDepthBuffer.data() ),
			outColorBuffer,
			thrust::raw_pointer_cast( m_dVoxelBuffer.data() ),
			thrust::raw_pointer_cast( m_dShadowMap.data() ),
			m_frameWidth, m_frameHeight,
			light.direction(),
			light.camera().viewProjectionMatrix(),
			light.diffusePower()
		);

		cudaUnbindTexture( tDiffuse );
		cudaUnbindTexture( tIllum );
		cudaUnbindTexture( tSpec );
		cudaUnbindTexture( tNormal );
	}
	cudaUnbindTexture( tDepthBuffer );
}



void Renderer::clearColorBuffer( uchar4 * dpOutColorBuffer )
{
	uchar4 const colorBufferClearValue = make_uchar4( 51, 51, 51, 255 );
	thrust::device_ptr< uchar4 > wrappedPtr( dpOutColorBuffer );

	thrust::fill
	( 
		wrappedPtr,
		wrappedPtr + resolution(),
		colorBufferClearValue
	);
}

void Renderer::clearDepthBuffer()
{
	unsigned int const depthBufferClearValue = std::numeric_limits< unsigned int >::max();
	m_dDepthBuffer.assign( m_dDepthBuffer.size(), depthBufferClearValue );
}

void Renderer::clearShadowMap()
{
	float const shadowMapClearValue = 1;
	m_dShadowMap.assign( m_dShadowMap.size(), shadowMapClearValue );
}

void Renderer::fillJobQueue( BFSJob const * dpJobs, int jobCount )
{
	cudaMemcpy
	(
		thrust::raw_pointer_cast( m_dJobQueue.data() ),
		dpJobs,
		jobCount * sizeof( BFSJob ),
		cudaMemcpyDeviceToDevice
	);
}



int Renderer::resolution() const
{
	return m_frameWidth * m_frameHeight;
}



// static
int Renderer::nBlocks( int nElements, int nThreadsPerBlock )
{
	int result = nElements / nThreadsPerBlock;
	return result + ( result * nThreadsPerBlock < nElements );
}