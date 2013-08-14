/*
The majority of this code is copied from Rob Farber's article
"CUDA, Supercomputing for the Masses" (http://www.drdobbs.com/parallel/cuda-supercomputing-for-the-masses-part/207200659)
*/

#include "../inc/Glue.h"

#define NOMINMAX
#include <Windows.h>

#include <cuda_gl_interop.h>

#include <GL/freeglut.h>
#include <GL/glew.h>

#include <cassert>

#include "../inc/Light.h"
#include "../inc/Object3D.h"
#include "../inc/Renderer.h"



// static 
Glue * Glue::m_globalInstance = nullptr;

// static
bool Glue::init
( 
	int argc, char ** argv,
	int windowWidth, int windowHeight,
	Renderer * renderer,
	Object3D * model,
	Light light,
	Camera camera
)
{
	// preconds
	assert( windowWidth > 0 );
	assert( windowHeight > 0 );
	assert( renderer != nullptr );
	assert( model != nullptr );

	bool result = false;

	if( nullptr == m_globalInstance )
	{
		m_globalInstance = new Glue
		(
			argc, argv,
			windowWidth, windowHeight,
			renderer,
			model,
			light,
			camera,

			result
		);
	}

	return result;
}

// static
void Glue::cleanUp()
{
	if( m_globalInstance != nullptr )
	{
		delete m_globalInstance;
	}
}



// static
void Glue::startGlutMainLoop()
{
	glutMainLoop();
}



Glue::Glue
( 
	int argc, char ** argv,
	int windowWidth, int windowHeight,
	Renderer * renderer,
	Object3D * model,
	Light light,
	Camera camera,

	bool & outSuccess
) :
	m_windowWidth( windowWidth ),
	m_windowHeight( windowHeight ),
	m_renderer( renderer ),
	m_model( model ),
	m_light( light ),
	m_camera( camera ),
	m_lastFrameTimeInMilliseconds( 0 )
{
	// preconds
	assert( windowWidth > 0 );
	assert( windowHeight > 0 );
	assert( renderer != nullptr );
	assert( model != nullptr );

	// Initialize freeglut and OpenGL
	glutInit( & argc, argv );
	glutInitDisplayMode( GLUT_RGBA | GLUT_DOUBLE );
	glutInitWindowSize( windowWidth, windowHeight );
	glutInitWindowPosition( 50, 50 );
	glutCreateWindow( "asvo@cuda" );	
	glutDisplayFunc( displayFunc );
	glutMouseFunc( mouseFunc );
	glutMotionFunc( motionFunc );

	glewInit();
	if( ! glewIsSupported( "GL_VERSION_2_0" ) )
	{
		fprintf( stderr, "ERROR: Support for necessary OpenGL extensions missing." );
		outSuccess = false;
		return;
	}  
	
	glViewport( 0, 0, windowWidth, windowHeight );
	glClearColor( 1, 1, 1, 0 );
	glDisable( GL_DEPTH_TEST );

	glMatrixMode( GL_MODELVIEW );
	glLoadIdentity();

	glMatrixMode( GL_PROJECTION );
	glLoadIdentity();

	glOrtho( 0, 1, 0, 1, 0, 1 );

	// Initialize CUDA
	cudaGLSetGLDevice( 0 );

	// Create PBO
	glGenBuffers( 1, & m_pbo );
	glBindBuffer( GL_PIXEL_UNPACK_BUFFER, m_pbo );
	glBufferData
	(
		GL_PIXEL_UNPACK_BUFFER, 
		windowResolution() * 4 * sizeof( GLubyte ), 
		nullptr, 
		GL_DYNAMIC_COPY
	);
	cudaGLRegisterBufferObject( m_pbo );

	// Create texture to render into and display on the screen
	glEnable( GL_TEXTURE_2D );
	glGenTextures( 1, & m_texture );
	glBindTexture( GL_TEXTURE_2D, m_texture );
	glTexImage2D
	( 
		GL_TEXTURE_2D, 
		0, 
		GL_RGBA8, 
		windowWidth, windowHeight, 
		0,
		GL_BGRA,GL_UNSIGNED_BYTE, 
		nullptr
	);
	//                                         !!!
	glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST );
	glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST );

	outSuccess = true;
}

Glue::~Glue()
{
	// delete pbo
    glBindBuffer( GL_ARRAY_BUFFER, m_pbo );
    glDeleteBuffers( 1, & m_pbo );
	cudaGLUnregisterBufferObject( m_pbo );

	// delete texture
	glDeleteTextures( 1, & m_texture );
}



// static 
void Glue::displayFunc()
{
	m_globalInstance->display();
}

void Glue::display()
{
	LARGE_INTEGER start;
	QueryPerformanceCounter( & start );
	uchar4 * dptr = nullptr;

	m_camera.update
	( 
		m_lastFrameTimeInMilliseconds,
		m_windowWidth,
		m_windowHeight
	);

	cudaGLMapBufferObject( (void**) & dptr, m_pbo );

	int animationFrameIndex = m_model->data()->update( m_lastFrameTimeInMilliseconds );
	m_renderer->render
	(
		* m_model,
		m_camera,
		m_light,
		animationFrameIndex,

		dptr
	);

	cudaGLUnmapBufferObject( m_pbo );
	glBindBuffer( GL_PIXEL_UNPACK_BUFFER, m_pbo );
	glBindTexture( GL_TEXTURE_2D, m_texture );

	glTexSubImage2D
	( 
		GL_TEXTURE_2D, 
		0, 0, 0, 
		m_windowWidth, m_windowHeight,
		GL_RGBA, GL_UNSIGNED_BYTE, nullptr
	);

	glBegin( GL_QUADS );
	glTexCoord2f( 0, 1 ); glVertex3f( 0, 0, 0 );
	glTexCoord2f( 0, 0 ); glVertex3f( 0, 1, 0 );
	glTexCoord2f( 1, 0 ); glVertex3f( 1, 1, 0 );
	glTexCoord2f( 1, 1 ); glVertex3f( 1, 0, 0 );
	glEnd();	

	glutSwapBuffers();
	glutPostRedisplay();

	LARGE_INTEGER end;
	QueryPerformanceCounter( & end );

	LARGE_INTEGER freq;
	QueryPerformanceFrequency( & freq );

	m_lastFrameTimeInMilliseconds = ( end.QuadPart - start.QuadPart ) / ( (double) freq.QuadPart ) * 1000.0;

	char title[ 64 ];
	sprintf( title, "asvo@cuda - %.1f fps", 1000.0 / m_lastFrameTimeInMilliseconds );
	glutSetWindowTitle( title );
}



// static 
void Glue::mouseFunc( int button, int state, int x, int y )
{
	m_globalInstance->m_camera.handleMouseButtonPress( button, state, x, y );
}

// static 
void Glue::motionFunc( int newX, int newY )
{
	m_globalInstance->m_camera.handleMouseMovement( newX, newY );
}



int Glue::windowResolution() const
{
	return m_windowWidth * m_windowHeight;
}

double Glue::windowAspectRatio() const
{
	return ( (double) m_windowWidth / m_windowHeight );
}