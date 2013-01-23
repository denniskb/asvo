/** \file */

#ifndef bfsoctree_operations_h

#define bfsoctree_operations_h

#include "bfsoctree.h"
#include "util.h"

/**
 * Imports a BFSOctree from a binary file.
 *
 * @param path    The path to the .asvo file containing the BFSOctree.
 * @param diffuse The path to the diffuse texture (raw rgb format).
 * @param illum   The path to the illumination texture (raw rgb format).
 * @param spec    The path to the specular texture (raw rgb format).
 * @param normal  The path to the normal texture (raw rgb format).
 *
 * @return The BFSOctree stored in the file at path.
 */
BFSOctree BFSOctreeImport(char const * path, char const * diffuse, char const * illum, char const * spec, char const * normal);

/**
 * Does cleanup work like releasing allocated memory.
 *
 * @param octree The BFSOctree to be cleaned up.
 */
void BFSOctreeCleanup(BFSOctree *octree);

/**
 * Returns the child count encoded by a child-mask.
 *
 * @param mask A child-mask encoding the child nodes of a voxel.
 *
 * @return The child count encoded by mask.
 */
unsigned long int $$(getChildCountFromMask)(unsigned long int mask);

/**
 * Updates the running animation of an octree.
 *
 * @param octree The BFSOctree to be animated.
 *
 * @return The new frame number after the animation has been updated.
 */
unsigned short int BFSOctreeUpdate(BFSOctree *octree);

#endif