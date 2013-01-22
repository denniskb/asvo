#include "../inc/BFSJob.h"

BFSJob make_BFSJob( int index, short x, short y, short z )
{
	BFSJob result = { index, x, y, z };
	return result;
}