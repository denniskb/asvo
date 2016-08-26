using Microsoft.Xna.Framework;
using System.IO;
using System.Collections.Generic;
using asvo.tools;

using asvo.world3D;

namespace asvo
{
    namespace datastructures
    {
        /// <summary>
        /// Represents a compressed octree. The nodes are stored sequentially in
        /// memory as if they were traversed via BFS. Empty nodes are omitted completely.
        /// 
        /// A node consists of two parts: "Positional data" and "visual data". The
        /// positional data determiens the octree hierarchy whereas the visual data is used for
        /// rendering.
        /// 
        /// The positional data consits of a "child mask" (one byte in which every bit represents a
        /// child of the current node) and a "child pointer" that stores the index of the first
        /// child node within the BFSOctree. Thus the mask allows to compute the
        /// child count of the current node as well as the children's positions.
        /// <code>
        /// node: (mask|child-ptr)
        /// 
        /// index: 0          1               2                 3                   ...
        /// root node    first child    second child    first child's first child   ...
        /// (1001000|1)  (00111011|3)   (00010000|8)    (00110100|36)               ...
        /// </code>
        /// See also <see cref="DynamicOctree"/> for more infos on the
        /// general octree structure used in this lib.
        /// </summary>
        internal class BFSOctree
        {            
            public readonly float dimension;
            public readonly BFSInnerNode[] _innerNodes;
            public readonly BFSLeaf[] _leaves;
            public Matrix[,] _animation; //frames x bones
            public uint frameCount;
            public uint boneCount;

            /// <summary>
            /// Creates a new BFSOctree out of a DynamicOctree.
            /// </summary>
            /// <param name="octree">The DynamicOctree to create this BFSOctree out of.</param>
            public BFSOctree(DynamicOctree octree)
            {               
                dimension = octree.dimension;

                uint leafCount = octree.getLeafCount();
                _innerNodes = new BFSInnerNode[octree.getNodeCount() - leafCount];
                _leaves = new BFSLeaf[leafCount];

                convert(octree);               
            }

            /// <summary>
            /// Converts a DynamicOctree into this BFSOctree.
            /// </summary>
            /// <param name="node">The DynamicOctree to convert into this BFSOctree.</param>
            private void convert(DynamicOctreeNode node)
            {
                Queue<DynamicOctreeNode> queue = new Queue<DynamicOctreeNode>();

                queue.Enqueue(node);
                node = null;
                uint offset = 1;
                int index = 0;

                while (queue.Count > 0)
                {
                    DynamicOctreeNode currentNode = queue.Dequeue();

                    uint mask = 0;
                    if (currentNode.hasChildren())
                    {
                        DynamicOctreeNode child = currentNode.getFirstChild();
                        for (byte i = 0; i < currentNode.getChildCount(); ++i)
                        {
                            queue.Enqueue(child);
                            
                            mask |= (byte)(1u << child.position);
                            child = child.getNextNode();
                        }
                    }                    

                    if (index < _innerNodes.Length)
                        _innerNodes[index++] = new BFSInnerNode(mask, offset, currentNode.vd);
                    else
                        _leaves[index++ - _innerNodes.Length] = new BFSLeaf(currentNode.vd);

                    offset += currentNode.getChildCount();
                }
                queue.Clear();
            }
         
            /// <summary>
            /// Exports this BFSOctree into a binary file format.
            /// Output-format:
            /// <code>
            /// bytes:              type/format:        interpretation:
            /// =======================================================
            /// 4                   IEEE 32bit float    dimension of the octree (which is a cube).
            /// 4                   uint32              inner node count.
            /// 4                   uint32              leaf node count.
            /// #inner-nodes *      BFSInnerNode        the inner nodes of the octree.     
            /// sizeof(inner-node)    
            /// #leaves *           BFSLeaf             the leaf nodes of the octree.
            /// sizeof(leaf-node)
            /// 4                   uint32              frame count of the animation.
            /// 4                   uint32              bone count of the octree.
            /// #frames *           Matrix              transformation matrices for every frame.
            /// #bones
            /// </code>
            /// See also <see cref="VisualData.export"/>, <see cref="BFSInnerNode.export"/> and
            /// <see cref="BFSLeaf.export"/> for how the composed data types are being exported.
            /// </summary>
            /// <param name="path">The relative filepath to store the exported file at.</param>
            public void export(string path)
            {
                BinaryWriter writer = new BinaryWriter(new FileStream(path, FileMode.Create));

                writer.Write(dimension);
                writer.Write(_innerNodes.Length);
                writer.Write(_leaves.Length);                
                                
                for (int i = 0; i < _innerNodes.Length; ++i)
                    _innerNodes[i].export(writer);

                for (int i = 0; i < _leaves.Length; ++i)
                    _leaves[i].export(writer);            

                writer.Write(frameCount);
                writer.Write(boneCount);

                for (int i = 0; i < frameCount; ++i)
                    for (int j = 0; j < boneCount; ++j)                    
                        Math3DHelper.export(_animation[i, j], writer);                   
                
                writer.Close();
            }

            /// <summary>
            /// Computes the child count from a given child mask.
            /// </summary>
            /// <param name="mask">A child mask.</param>
            /// <returns>The child count encoded in <paramref name="mask"/>.</returns>
            public static byte getChildCount(byte mask)
            {
                return (byte)((mask & 1) +
                      ((mask & 2) >> 1) +
                      ((mask & 4) >> 2) +
                      ((mask & 8) >> 3) +
                      ((mask & 16) >> 4) +
                      ((mask & 32) >> 5) +
                      ((mask & 64) >> 6) +
                      ((mask & 128) >> 7));
            }
        }

        /// <summary>
        /// Represents an inner node of a BFSOctree.
        /// </summary>
        internal struct BFSInnerNode
        {
            public readonly VisualData visualData;
            public readonly uint childMask;
            public readonly uint childIndex;            

            /// <summary>
            /// Creates a new BFSInnerNode.
            /// </summary>
            /// <param name="childMask">child-mask</param>
            /// <param name="childIndex">child-pointer</param>
            /// <param name="visualData">visual data</param>
            public BFSInnerNode(uint childMask, uint childIndex, VisualData visualData)
            {
                this.childMask = childMask;
                this.childIndex = childIndex;
                this.visualData = visualData;
            }

            /// <summary>
            /// Writes the contents of this BFSInnerNode to a
            /// binary stream using the provided BinaryWriter.
            /// Output-format:
            /// <code>
            /// bytes:              type/format:    interpretation:
            /// ===================================================
            /// sizeof(VisualData)  VisualData      visual data
            /// 4                   uint32          child mask (only lower 8 bits matter)
            /// 4                   uint32          child index
            /// </code>
            /// See also <see cref="VisualData.export"/> for how the
            /// composed data types are being exported.
            /// </summary>
            /// <param name="writer">The BinaryWriter used to write to the stream.</param>
            public void export(BinaryWriter writer)
            {
                visualData.export(writer);
                writer.Write(childMask);
                writer.Write(childIndex);
            }
        }

        /// <summary>
        /// Represents a leaf node of a BFSOctree. A leaf doesn't contain any
        /// positional data, but only visual data.
        /// </summary>
        internal struct BFSLeaf
        {
            public readonly VisualData visualData;

            /// <summary>
            /// Constructs a new leaf node.
            /// </summary>
            /// <param name="visualData">visual data</param>
            public BFSLeaf(VisualData visualData)
            {
                this.visualData = visualData;
            }

            /// <summary>
            /// Writes the contents of this BFSLeaf to a
            /// binary stream using the provided BinaryWriter.
            /// Output-format:
            /// <code>
            /// bytes:              type/format:    interpretation:
            /// ===================================================
            /// sizeof(VisualData)  VisualData      visual data
            /// </code>
            /// See also <see cref="VisualData.export"/> for how the
            /// composed data types are being exported.
            /// </summary>
            /// <param name="writer">The BinaryWriter used to write to the stream.</param>
            public void export(BinaryWriter writer)
            {
                visualData.export(writer);
            }
        }

        /// <summary>
        /// A container for all visual data stored inside a node.
        /// </summary>
        internal struct VisualData
        {
            public Vector3 normal;
            public byte boneIndex0;
            public byte boneIndex1;
            public byte boneIndex2;
            public byte boneIndex3;
            public Vector4 boneWeights;
            public Vector2 texCoords;
            public Vector3 tangent;

            /// <summary>
            /// Writes the contents of this struct to a
            /// binary stream using the provided BinaryWriter.
            /// Output-format:
            /// <code>
            /// bytes:          type/format:    interpretation:
            /// ===============================================
            /// sizeof(Vector3) Vector3         normal
            /// sizeof(Vector3) Vector3         tangent
            /// sizeof(Vecotr2) Vector2         texture coordinates
            /// 1               uchar           first bone index
            /// 1               uchar           second bone index
            /// 1               uchar           third bone index
            /// 1               uchar           fourth bone index
            /// sizeof(Vector4) Vector4         bone weights
            /// </code>
            /// See also <see cref="Math3DHelper.export(Vector2,BinaryWriter)"/>,
            /// <see cref="Math3DHelper.export(Vector3,BinaryWriter)"/> and
            /// <see cref="Math3DHelper.export(Vector4,BinaryWriter)"/> for how the
            /// composed data types are being exported.
            /// </summary>
            /// <param name="writer">The BinaryWriter used to write to the stream.</param>
            public void export(BinaryWriter writer)
            {
                Math3DHelper.export(normal, writer);
                Math3DHelper.export(tangent, writer);
                Math3DHelper.export(texCoords, writer);
                
                writer.Write(boneIndex0);
                writer.Write(boneIndex1);
                writer.Write(boneIndex2);
                writer.Write(boneIndex3);

                Math3DHelper.export(boneWeights, writer);
            }
        }
    }
}
