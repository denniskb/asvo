/** \file
 * This module is the bridge between the host and the device.
 * It allows the host code to call kernels without worrying about
 * execution configurations for example.
 */

#ifndef kernel_caller_h

#define kernel_caller_h

#include <cuda_runtime.h>
#include "object3d.h"

/**
 * Called by the GLUT framework. Executes the kernel which
 * renders an asvo once per frame.
 *
 * @param colorBuffer The color-buffer to render the asvo into.
 */
void kernelRun(uchar4 *colorBuffer);

/**
 * Sets runtime parameters needed for the kernel execution.
 *
 * @param obj The Object3d to be rendered.
 */
void kernelSetParams(Object3d obj);

/**
 * Cleans up initializations.
 */
void kernelCleanup(void);

#endif