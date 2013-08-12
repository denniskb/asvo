#define NOMINMAX
#include <Windows.h>

#include <memory>
#include <string>

#include <GL/freeglut.h>

#include <helper_cuda.h>
#include <vector_functions.h>
#include <vector_types.h>

#include "../inc/extended_helper_math.h"
#include "../inc/glue.h"
#include "../inc/Light.h"
#include "../inc/Object3D.h"
#include "../inc/Renderer.h"

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
	Object3D imrod( new BFSOctree
	(
		( path + "imrod.asvo" ).c_str(),
		( path + "diffuse.raw" ).c_str(),
		( path + "illum.raw" ).c_str(),
		( path + "spec.raw" ).c_str(),
		( path + "normal.raw" ).c_str()
	),
	true );

	float3 rotAxis = make_float3( 1.f, 0.f, 0.f );
	imrod.assignTransform( make_rotation( rotAxis, -1.5707f ) );

	// Set up the light.
	float3 lightPosition = make_float3( -1.f, -0.5f, 0.5f );
	float lightDiffusePower = 0.8;
	Light light( lightPosition, lightDiffusePower );

	// Initialize the GLUT framework.
	if ( ! glueInit( frameWidthInPixels, frameHeightInPixels, argc, argv, pRenderer.get(), imrod, light ) )
	{
		return 1;
	}

	// Set up the camera.
	float3 position = make_float3( 0.f, 25.f, -100.f );
	float3 lookAt = make_float3( 0.f, 0.f, 0.f );
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