/** \file
 * The Glue module is the bridge between the main program utilizing
 * cuda and the OpenGL part responsible for window creation, drawing, etc.
 */

#ifndef glue_h

#define glue_h

#include <cuda_runtime.h>
#include <helper_cuda.h>

#include "object3d.h"

class Light;
class Renderer;

/**
 * Initializes the Glue module. Creates a window, OpenGL context,
 * initializes cuda etc.
 * Original code by Rob Farber.
 *
 * @param screenWidth  The width, in pixels, of the display.
 * @param screenHeight The height, in pixels, of the display.
 * @param windowWidth  The width, in pixels, of the application window.
 * @param windowHeight The height, in pixels, of the application window.
 * @param argc         The argument count provided by the user when starting this application.
 * @param argv         An array of strings containing the arguments provided by the user when
                       starting this application.
 * @param runKernel    Callback-function that will be called by GLUT in order to
 *                     render an asvo into the provided color-buffer.
 *
 * @return true if the initialization was successful, false otherwise.
 */
bool glueInit
(
    unsigned short int windowWidth,
    unsigned short int windowHeight,
    int argc, char **argv,
    Renderer * pRenderer,
	Object3d obj,
	Light const & light
);

/**
 * Cleans up all initializations done in glueInit.
 * Original code by Rob Farber.
 */
void glueCleanup(void);

/**
 * Returns the window width.
 *
 * @return The width, in pixels, of the application window.
 */
unsigned short int glueGetWindowWidth(void);

/**
 * Returns the window height.
 *
 * @return The height, in pixels, of the application window.
 */
unsigned short int glueGetWindowHeight(void);

/**
 * Returns the window resolution.
 *
 * @return The resolution, in pixels, of the application window.
 */
unsigned long int glueGetWindowResolution(void);

/**
 * Returns the last frame time in milliseconds.
 *
 * @return The time, in milliseconds, elapsed since the last frame.
 */
double glueGetLastFrameTime(void);

/**
 * Returns the aspect ratio of the window.
 *
 * @return glueGetWindowWidth() / glueGetWindowHeight()
 */
float glueGetWindowRatio(void);

#endif