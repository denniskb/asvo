#ifndef bfsinnernode_h

#define bfsinnernode_h

#include "visualdata.h"

/**
 * Represents an inner node of a BFSOctree.
 */
typedef struct
{
	VisualData vd;
	unsigned long int mask;
	unsigned long int childPtr;
} BFSInnerNode;

#endif