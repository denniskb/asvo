#include <Windows.h>

#include <memory>
#include <string>

#include <GL/freeglut.h>

#include <helper_cuda.h>

#include "../inc/bfsoctree_operations.h"
#include "../inc/camera_operations.h"
#include "../inc/glue.h"
#include "../inc/light.h"
#include "../inc/math3d.h"
#include "../inc/object3d.h"
#include "../inc/object3d_operations.h"
#include "../inc/Renderer.h"
#include "../inc/vector3.h"

int main(int argc, char **argv)
{	
	std::string path( argv[ 0 ] );
	int lastSlash = path.find_last_of( "\\" );
	path.resize( lastSlash + 1 );
	path.append( "../../content/" );

	// Load an asvo from file.
	Object3d imrod = obj3dInit(BFSOctreeImport
	(
		( path + "imrod.asvo" ).c_str(),
		( path + "diffuse.raw" ).c_str(),
		( path + "illum.raw" ).c_str(),
		( path + "spec.raw" ).c_str(),
		( path + "normal.raw" ).c_str()
	), true);

	Vector3 rotAxis = { 1.f, 0.f, 0.f };
	obj3dAssignTransform(&imrod, h_createRotation(rotAxis, -1.5707f));
	// Copy the model to device memory.
	BFSOctreeCopyToDevice(&imrod.data);

	// TODO: Replace hard-coded values
	int const frameWidthInPixels = 550;
	int const frameHeightInPixels = 800;
	bool const shadowMapping = true;

	std::unique_ptr< Renderer > pRenderer(
		new Renderer( frameWidthInPixels, frameHeightInPixels, shadowMapping )
	);

	// Initialize the GLUT framework.
	if (!glueInit(frameWidthInPixels, frameHeightInPixels, argc, argv, pRenderer.get(), imrod))
	{
		return 1;
	}

	// Set up the camera.
	Vector3 pos = { 0.f, 25.f, -100.f };
	Vector3 lookAt = { 0.f, 0.f, 0.f };	
	camInit(pos, lookAt, 
			1.f, glueGetWindowRatio(), 10.f, 200.f);

	// Set up the light.
	Vector3 light = { -1.f, -0.5f, 0.5f };
	lightSet(light, 0.8f);	

	// Start the main render-and-update loop
	glutMainLoop();	

	// Do cleanup work.
	BFSOctreeCleanup(&imrod.data);
	glueCleanup();

	return 0;
}