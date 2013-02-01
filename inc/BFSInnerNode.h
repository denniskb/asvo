#pragma once

#include "VisualData.h"

/**
 * Represents an inner node of a BFSOctree.
 */
class BFSInnerNode
{
public:

	VisualData vd;
	unsigned long int mask;
	unsigned long int childPtr;
};