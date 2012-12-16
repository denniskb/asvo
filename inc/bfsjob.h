#ifndef bfsjob_h

#define bfsjob_h

/**
 * Stores all data neccessary to process a single voxel.
 * The renderer stores BFSJobs inside a queue and every job is
 * processed by a thread.
 *
 * @see traverse
 */
typedef struct
{
	unsigned long int index;
	unsigned short int x;
	unsigned short int y;
	unsigned short int z;
} BFSJob;

#endif