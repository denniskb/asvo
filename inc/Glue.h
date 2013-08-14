#pragma once

#define NOMINMAX
#include <Windows.h>
#include <GL/glew.h>
#include <GL/GL.h>

#include <memory>

#include "Camera.h"
#include "Light.h"

class Object3D;
class Renderer;

/*
A thin wrapper around freeglut which covers initialization
of the CUDA device, creation of textures, render targets,
setting up the view port, etc.
*/
class Glue
{
public:
	
	/*
	Singleton. Makes the provided instance available globally.
	Used to query window dimensions and register freeglut event
	handlers (which need to be raw function pointers).
	@param glue does not take ownership of the pointer
	*/
	static void setGlobalInstance( Glue * glue );
	/*
	Behavior is undefined if called without a preceding call to
	setGlobalInstance!
	*/
	static Glue const & globalInstance();

	/*
	Initializes freeglut, OpenGL and CUDA and sets everything up
	for rendering (textures, render targets, etc.)
	@param renderer takes ownership of the pointer
	@param model takes ownership of the pointer
	
	@param outSuccess returns true if the initialization was successful,
	false otherwise (in this case the app should be aborted)

	@precond windowWidth > 0
	@precond windowHeight > 0
	@precond renderer != nullptr
	@precond model != nullptr
	*/
	Glue
	( 
		int argc, char ** argv,
		int windowWidth, int windowHeight,
		Renderer * renderer,
		Object3D * model,
		Light light,
		Camera camera,

		bool & outSuccess
	);
	/*
	Cleans up all initialized resources.
	*/
	~Glue();

	void startGlutMainLoop() const;

	int windowWidth() const;
	int windowHeight() const;
	int windowResolution() const;
	// windowWidth() / windowHeight()
	double windowAspectRatio() const;
	
	double lastFrameTimeInMilliSeconds() const;

private:

	static Glue * m_globalInstance;

	/*
	Registered as glutDisplayFunc.
	Calls display.
	If called without a preceding call to setGlobalInstance, the
	behavior is undefined!
	*/
	static void displayFunc();
	/*
	Main render function. Binds buffers and calls Renderer::render
	*/
	void display();

	/*
	Registered as glutMouseFunc.
	Calls Camera::handleMouseButtonPress
	*/
	static void mouseFunc( int button, int state, int x, int y );
	/*
	Registered as glutMotionFunc.
	Calls Camera::handleMouseMovement
	*/
	static void motionFunc( int newX, int newY );

	int m_windowWidth;
	int m_windowHeight;

	std::unique_ptr< Renderer > m_renderer;
	std::unique_ptr< Object3D > m_model;

	Light m_light;
	Camera m_camera;

	GLuint m_pbo;
	GLuint m_texture;

	double m_lastFrameTimeInMilliSeconds;
};