#include "../inc/glue.h"

#include <Windows.h>

#include <ctime>

#include <GL/glew.h>
#include <GL/freeglut.h>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <helper_cuda.h>

#include "../inc/camera_operations.h"
#include "../inc/light.h"
#include "../inc/Renderer.h"

static int _argc;
static char **_argv;
static unsigned short int _windowWidth;
static unsigned short int _windowHeight;
static double _msFrameTime = 33;
static GLuint _pbo;
static GLuint _texture;
static Renderer * _pRenderer;
static Object3d _obj;

/*
 * Creates a window and sets up the viewport.
 * Original code by Rob Farber.
 */
static bool initGL(void);

/*
 * Dummy for the display function called by glut. For now,
 * all it does, is map the texture to device memory, call the passed cuda kernel,
 * unmap the texture and draw the results on a quad.
 * Original code by Rob Farber.
 */
static void displayFuncDummy(void);

/*
 * Initializes Open GL context.
 * Original code by Rob Farber.
 */
static void initCuda(void);

/*
 * Creates the pixel buffer object used to draw to the screen.
 * Original code by Rob Farber.
 */
static void createPBO(void);

/*
 * Creates the texture used to hold the color information.
 * Original code by Rob Farber.
 */
static void createTexture(void);

/*
 * Deletes the pbo.
 * Orignial code by Rob Farber.
 */
static void deletePBO(void);

/*
 * Deletes the texture.
 * Original code by Rob Farber.
 */
static void deleteTexture(void);

bool glueInit
(
	unsigned short int windowWidth,
    unsigned short int windowHeight,
    int argc, char **argv,
    Renderer * pRenderer,
	Object3d obj
)
{
	_argc = argc;
	_argv = argv;
	_windowWidth = windowWidth;
	_windowHeight = windowHeight;
	_pRenderer = pRenderer;
	_obj = obj;

	if (!initGL())
		return false;

	initCuda();

	createPBO();
	createTexture();

	return true;
}

void glueCleanup()
{
	deletePBO();
	deleteTexture();
	cudaThreadExit();
}

unsigned short int glueGetWindowWidth(void)
{
	return _windowWidth;
}

unsigned short int glueGetWindowHeight(void)
{
	return _windowHeight;
}

unsigned long int glueGetWindowResolution(void)
{
	static const unsigned long int windowRes = _windowWidth * _windowHeight;
	return windowRes;
}

double glueGetLastFrameTime(void)
{
	return _msFrameTime;
}

float glueGetWindowRatio(void)
{
	static const float ratio = _windowWidth / (float)_windowHeight;
	return ratio;
}

void deleteTexture(void)
{
    glDeleteTextures(1, &_texture);
    _texture = NULL;
}

static bool initGL(void)
{
	glutInit(&_argc, _argv);
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
	glutInitWindowSize(_windowWidth, _windowHeight);
	glutInitWindowPosition( 50, 50 );
	glutCreateWindow("asvo@cuda");	
	glutDisplayFunc(displayFuncDummy);
	glutMouseFunc(mouseFunc);
	glutMotionFunc(motionFunc);

	glewInit();
	if (! glewIsSupported( "GL_VERSION_2_0 " ) ) {
		fprintf(stderr, "ERROR: Support for necessary OpenGL extensions missing.");
		return false;
	}  
	
	glViewport(0, 0, _windowHeight, _windowHeight);
	glClearColor(1., 1., 1., 0.);
	glDisable(GL_DEPTH_TEST);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();

	glOrtho(0.0f, 1.0f, 0.0f, 1.0f, 0.0f, 1.0f);
  
	return true;
}

static void displayFuncDummy(void)
{	
	LARGE_INTEGER start;
	QueryPerformanceCounter( & start );
	uchar4 *dptr = NULL;

	camUpdate(glueGetLastFrameTime());

	// map OpenGL buffer object for writing from CUDA on a single GPU
	// no data is moved (Win & Linux). When mapped to CUDA, OpenGL
	// should not use this buffer
	cudaGLMapBufferObject((void**)&dptr, _pbo);

	// execute the kernel
	if( _pRenderer )
	{
		_pRenderer->rasterize
		(
			_obj,
			camGet(),
			
			dptr
		);
	}

	// unmap buffer object
	cudaGLUnmapBufferObject(_pbo);

	// Create a texture from the buffer
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, _pbo);

	// bind texture from PBO
	glBindTexture(GL_TEXTURE_2D, _texture);

	// Note: glTexSubImage2D will perform a format conversion if the
	// buffer is a different format from the texture. We created the
	// texture with format GL_RGBA8. In glTexSubImage2D we specified
	// GL_BGRA and GL_UNSIGNED_INT. This is a fast-path combination

	// Note: NULL indicates the data resides in device memory
	glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, _windowWidth, _windowHeight, 
					GL_RGBA, GL_UNSIGNED_BYTE, NULL);

	// Draw a single Quad with texture coordinates for each vertex.
	glBegin(GL_QUADS);
	glTexCoord2f(0.0f, 1.0f); glVertex3f(0.0f, 0.0f, 0.0f);
	glTexCoord2f(0.0f, 0.0f); glVertex3f(0.0f, 1.0f, 0.0f);
	glTexCoord2f(1.0f, 0.0f); glVertex3f(1.0f, 1.0f, 0.0f);
	glTexCoord2f(1.0f, 1.0f); glVertex3f(1.0f, 0.0f, 0.0f);
	glEnd();	

	// Don't forget to swap the buffers!
	glutSwapBuffers();
	glutPostRedisplay();

	LARGE_INTEGER end;
	QueryPerformanceCounter( & end );

	LARGE_INTEGER freq;
	QueryPerformanceFrequency( & freq );

	_msFrameTime = ( end.QuadPart - start.QuadPart ) / ( (double) freq.QuadPart ) * 1000;

	char title[100];
	sprintf( title, "asvo@cuda - %.1f fps", 1000.0 / _msFrameTime );
	glutSetWindowTitle( title );
}

/*
 * Initializes Open GL context.
 */
static void initCuda(void)
{
	cudaGLSetGLDevice( 0 );
}

static void createPBO(void)
{  
	// Generate a buffer ID called a PBO (Pixel Buffer Object)
	glGenBuffers(1, &_pbo);
	// Make this the current UNPACK buffer (OpenGL is state-based)
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, _pbo);
	// Allocate data for the buffer. 4-channel 8-bit image
	glBufferData(GL_PIXEL_UNPACK_BUFFER, _windowWidth * _windowHeight * 4 * sizeof(GLubyte), NULL, GL_DYNAMIC_COPY);
	cudaGLRegisterBufferObject(_pbo);
}

void createTexture(void)
{
	// Enable Texturing
	glEnable(GL_TEXTURE_2D);

	// Generate a texture identifier
	glGenTextures(1, &_texture);

	// Make this the current texture (remember that GL is state-based)
	glBindTexture(GL_TEXTURE_2D, _texture);

	// Allocate the texture memory. The last parameter is NULL since we only
	// want to allocate memory, not initialize it
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, _windowWidth, _windowHeight, 0,
 				 GL_BGRA,GL_UNSIGNED_BYTE, NULL);

	// Must set the filter mode, GL_LINEAR enables interpolation when scaling
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	// Note: GL_TEXTURE_RECTANGLE_ARB may be used instead of
	// GL_TEXTURE_2D for improved performance if linear interpolation is
	// not desired. Replace GL_LINEAR with GL_NEAREST in the
	// glTexParameteri() call
}

void deletePBO(void)
{  
    // unregister this buffer object with CUDA
    cudaGLUnregisterBufferObject(_pbo);
    
    glBindBuffer(GL_ARRAY_BUFFER, _pbo);
    glDeleteBuffers(1, &_pbo);
    
    _pbo = NULL;
}