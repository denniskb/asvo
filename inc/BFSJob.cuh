#pragma once

#include "BFSJob.h"

#ifndef __CUDACC__
#error "Compiling device code with CL"
#endif

__device__
BFSJob dmake_BFSJob( int index, short x, short y, short z )
{
	BFSJob result = { index, x, y, z };
	return result;
}