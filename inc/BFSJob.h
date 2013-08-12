#pragma once

#include <host_defines.h>

struct BFSJob
{
	int index;
	short x;
	short y;
	short z;
};

inline __host__ __device__ BFSJob make_BFSJob( int index, short x, short y, short z )
{
	BFSJob result = { index, x, y, z };
	return result;
}