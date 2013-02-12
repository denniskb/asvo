#include "../inc/BFSInnerNode.h"

void BFSInnerNode::deserialize( FILE * inFile )
{
	vd.deserialize( inFile );
		
	fread( & mask,     4, 1, inFile );
	fread( & childPtr, 4, 1, inFile );
}