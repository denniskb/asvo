using System.Threading;
using Microsoft.Xna.Framework;
using Microsoft.Xna.Framework.Graphics;

using asvo.multithreading;

namespace asvo
{
    namespace datastructures
    {
        /// <summary>
        /// Represents a depth buffer that is accessable by multiple threads.
        /// Internally multiple depth buffers are stored so that each can
        /// be accessed by its own thread and no conflicts occur.
        /// These depth buffers are merged back into one (by computing
        /// max(depthBuffer0, ..., depthBufferN) for every element).
        /// </summary>
        internal class SharedDepthBuffer
        {
            public readonly float[][] _elements;
            public readonly float[] _maxDims;

            /// <summary>
            /// Constructs a new shared depth buffer with the size of elementCount
            /// </summary>
            /// <param name="elementCount">The size of the buffer.</param>
            public SharedDepthBuffer(int elementCount)
            {
                _elements = new float[JobCenter.getWorkerCount()][];
                for (int i = 0; i < JobCenter.getWorkerCount(); ++i)
                    _elements[i] = new float[elementCount];

                _maxDims = new float[JobCenter.getWorkerCount()];
            }

            /// <summary>
            /// Zeroes the depth buffer.
            /// </summary>
            /// <param name="threadIndex">Index of the calling thread, starts at 0</param>
            public void zeroOut(int threadIndex)
            {
                for (int i = 0; i < _elements[threadIndex].Length; ++i)
                    _elements[threadIndex][i] = 1.0f;

                _maxDims[threadIndex] = 0.0f;
            }

            /// <summary>
            /// Merges the results from the different threads into one array. Additionally,
            /// an array of 2D surfaces (<paramref name="colorBuffer"/>) is provided. Every
            /// element of this array was accessed by a different thread. The winning color at
            /// every pixel is derived from the winning (the smallest) depth at every pixel.
            /// </summary>
            /// <param name="threadIndex">Index of the calling thread, starts at 0.</param>
            /// <param name="colorBuffer">An array of 2D surfaces.</param>
            public void merge(int threadIndex, Color[][] colorBuffer)
            {
                int start = (_elements[threadIndex].Length * threadIndex) / JobCenter.getWorkerCount();
                int end = (_elements[threadIndex].Length * (threadIndex + 1)) / JobCenter.getWorkerCount();

                float minDepth;
                int minColor = 0;
                for (int i = start; i < end; ++i)
                {
                    minDepth = 1.0f;
                    for (int j = 0; j < JobCenter.getWorkerCount(); ++j)
                    {
                        if (_elements[j][i] < minDepth)
                        {
                            minDepth = _elements[j][i];
                            minColor = j;
                        }
                    }
                    _elements[0][i] = minDepth;
                    colorBuffer[0][i] = colorBuffer[minColor][i];
                }

                if (threadIndex == 0)
                {
                    float maxDim = 0.0f;
                    for (int i = 0; i < JobCenter.getWorkerCount(); ++i)
                    {
                        if (_maxDims[i] > maxDim)
                            maxDim = _maxDims[i];
                    }
                    _maxDims[0] = maxDim;
                }
            }
        }
    }
}
