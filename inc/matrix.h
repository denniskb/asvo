#ifndef matrix_h

#define matrix_h

/**
 * Represents a 4x4 matrix.
 */
typedef struct
{	//row * col
	float m11; float m12; float m13; float m14;
	float m21; float m22; float m23; float m24;
	float m31; float m32; float m33; float m34;
	float m41; float m42; float m43; float m44;
} Matrix;

#endif