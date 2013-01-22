#pragma once

class BFSJob
{
public:

	int index;
	short x;
	short y;
	short z;
};

// TODO: Find clean and robust way to share code between host and device
BFSJob make_BFSJob( int index, short x, short y, short z );