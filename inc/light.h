/** \file
 * Allows manipulating the only light in the application.
 * It's a directional light.
 */

#ifndef light_h

#define light_h

#include "vector3.h"
#include "camera.h"

/**
 * Sets the light.
 *
 * @param light        The direction of the light.
 * @param diffusePower The diffuse intensity of the light between 0.0 and 1.0
 *                     The light's ambient intensity equals (1.0 - diffusePower).
 */
void lightSet(Vector3 light, float diffusePower);

/**
 * Returns the camera defined by the light.
 * Used for shadow mapping.
 *
 * @return The camera defined by the light.
 */
Camera lightGetCam(void);

/**
 * Returns the direction of the light.
 *
 * @return The direction of the light.
 */
Vector3 lightGetDir(void);

/**
 * Returns the diffuse intensity of the light.
 *
 * @return The diffuse intensity of the light.
 */
float lightGetDiffusePower(void);

/**
 * Returns the ambient intensity of the light.
 *
 * @return 1.0 - lightGetDiffusePower()
 */
float lightGetAmbientPower(void);

#endif