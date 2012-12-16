/** \file */

#ifndef camera_operations_h

#define camera_operations_h

#include "camera.h"
#include "vector3.h"

/**
 * Initializes the only camera in the application.
 *
 * @param pos       The camera's position.
 * @param lookAt    The point at which the camera looks.
 * @param fow       Field of vision in radians.
 * @param ratio     The aspect ratio (width / height).
 * @param nearPlane The distance to the near clipping plane.
 * @param farPlane The distance to the far clipping plane.
 */
void camInit(Vector3 pos, Vector3 lookAt,
             float fov, float ratio, float nearPlane, float farPlane);

/**
 * Returns the only camera in the application.
 *
 * @return The only camera in the application.
 */
Camera camGet(void);

/**
 * GLUT callback function that handles camera movement.
 * The parameters are provided by GLUT.
 */
void mouseFunc(int button, int state, int x, int y);

/**
 * GLUT callback function that handles camera movement.
 * The parameters are provided by GLUT.
 */
void motionFunc(int x, int y);

/**
 * Updates the camera based on user input and
 * the elapsed time since the last frame.
 *
 * @param lastFrameTime The time, in milliseconds, elapsed since the last frame.
 */
void camUpdate(int lastFrameTime);

#endif