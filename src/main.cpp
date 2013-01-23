#include <Windows.h>

#include <memory>
#include <string>

#include <GL/freeglut.h>

#include <helper_cuda.h>

#include "../inc/glue.h"
#include "../inc/Light.h"
#include "../inc/math3d.h"
#include "../inc/object3d.h"
#include "../inc/object3d_operations.h"
#include "../inc/Renderer.h"
#include "../inc/vector3.h"

int main(int argc, char **argv)
{	
	// TODO: Replace hard-coded values
	int const frameWidthInPixels = 550;
	int const frameHeightInPixels = 800;
	bool const shadowMapping = true;

	std::unique_ptr< Renderer > pRenderer(
		new Renderer( frameWidthInPixels, frameHeightInPixels, shadowMapping )
	);

	std::string path( argv[ 0 ] );
	int lastSlash = path.find_last_of( "\\" );
	path.resize( lastSlash + 1 );
	path.append( "../../content/" );

	// Load an asvo from file.
	Object3d imrod = obj3dInit( BFSOctree
	(
		( path + "imrod.asvo" ).c_str(),
		( path + "diffuse.raw" ).c_str(),
		( path + "illum.raw" ).c_str(),
		( path + "spec.raw" ).c_str(),
		( path + "normal.raw" ).c_str()
	),
	true );

	Vector3 rotAxis = { 1.f, 0.f, 0.f };
	obj3dAssignTransform(&imrod, h_createRotation(rotAxis, -1.5707f));

	// Set up the light.
	Vector3 lightPosition = { -1.f, -0.5f, 0.5f };
	float lightDiffusePower = 0.8;
	Light light( lightPosition, lightDiffusePower );

	// Initialize the GLUT framework.
	if ( ! glueInit( frameWidthInPixels, frameHeightInPixels, argc, argv, pRenderer.get(), imrod, light ) )
	{
		return 1;
	}

	// Set up the camera.
	Vector3 position = { 0.f, 25.f, -100.f };
	Vector3 lookAt = { 0.f, 0.f, 0.f };	
	float fov = 1;
	Camera camera( position, lookAt, fov, glueGetWindowRatio(), 10, 200 );
	Camera::setGlobalCamera( camera );

	// Start the main render-and-update loop
	// FIXME: App crashes if we omit the following call
	glutMainLoop();	

	// Do cleanup work.
	glueCleanup();

	return 0;
}