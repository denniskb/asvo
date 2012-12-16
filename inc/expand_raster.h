/** \file */

#ifndef expand_raster_h

#define expand_raster_h

#include <cuda_runtime.h>
#include "voxeldata.h"
#include "object3d.h"
#include "camera.h"
#include "matrix.h"

/**
 * Invokes the rasterizer on the GPU. This function exists to separate
 * device code from host code. It is being called by the kernel-caller (host code) and
 * itself calls the actual kernels used for rendering an asvo.
 *
 * @param depthBuffer              The depth-buffer to use for rendering.
 * @param colorBuffer              The color-buffer to output the final image to.
 * @param voxelBuffer              The voxel-buffer to use for rendering.
 * @param obj                      The Object3d to be rendered.
 * @param cam                      The virtual camera to use for rendering.
 * @param shadowMap                A depth-buffer used to implement shadow mapping.
 * @param lightWorldViewProjection light matrix * obj world transform matrix * camera view matrix * camera projection matrix.
 */
void expandRasterInvoke(unsigned int *depthBuffer, uchar4 *colorBuffer, VoxelData *voxelBuffer,
                        Object3d obj, Camera cam, float *shadowMap, Matrix lightWorldViewProjection);

/**
 * Initializes the rasterizer. Does work that is
 * neccessary for rendering but that needs to be
 * set up only once.
 */
void expandRasterInit(void);

#endif