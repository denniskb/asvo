/** \file
 * A collection of some helpful utilities.
 */

#ifndef util_h

#define util_h

/**
 * The following two macro pairs are closely related.
 * They allow to declare, define and call functions for both hosts and devices without
 * typing several declarations, definitions and call instructions.
 *
 * $$(funcName) and $(funcName) will translate to appropriate
 * function declarations and calls depending on whether the currently
 * compiled code is host code or device code.
 *
 * Host functions are prefixed with "h_", while
 * device functions are prefixed with "d_".
 *
 * This was very handy for writing the math lib, since
 * all math functions shared the same implementation.
 */
#ifdef __CUDACC__

#define $$(funcDef) static __device__ d_##funcDef
#define $(funcCall) d_##funcCall

#else

#define $$(funcDef) h_##funcDef
#define $(funcCall) h_##funcCall

#endif

#endif