using Microsoft.Xna.Framework;
using asvo.world3D;
using System.Collections.Generic;

namespace asvo {
    namespace datastructures
    {
        /// <summary>
        /// Represents a tree data structure in which every node has up to
        /// 8 child nodes.
        /// Convention:
        /// <code>
        ///   6---7
        ///  /   /|
        /// 2---3 5
        /// |   |/
        /// 0---1
        /// </code>
        /// with:
        /// <code>
        /// y z
        /// |/
        /// 0-x (left handed coordinate system)
        /// </code>
        /// This is a dynamic octree that can be altered at runtime and
        /// is intended for construction of 3D models.
        /// 
        /// Every node has a pointer to its first child and a pointer to the
        /// next child of his parent node after it:
        /// <code>
        /// parent->nil
        ///    |
        ///  node->parent's second child->parent's third child->nil
        ///    |                                     |
        ///  node's first child->  ...    parent's third child's
        ///   ...                               first child
        /// </code>
        /// Since the octree is sparse, nodes store the
        /// relative position to their parents.
        /// Every DynamicOctree is a cube.
        /// </summary>
        internal class DynamicOctree : DynamicOctreeNode
        {
            public readonly float dimension;

            /// <summary>
            /// Creates a new DynamicOctree with the specified dimension.
            /// </summary>
            /// <param name="dimension">Dimension of the bounding cube spanned by this octree.</param>          
            public DynamicOctree(float dimension) : base(0)
            {
                this.dimension = dimension;
            }           
        }

        /// <summary>
        /// Represents an node of a DynamicOctree.
        /// </summary>
        internal class DynamicOctreeNode
        {            
            private byte childCount;
            public readonly byte position;
            private DynamicOctreeNode firstChild;
            private DynamicOctreeNode nextNode;
            public VisualData vd;
            public float dist;         

            /// <summary>
            /// Creates a new empty node with no child nodes.
            /// </summary>
            /// <param name="position">The position of this node relative to
            /// its parent. <seealso cref="DynamicOctree"/></param>
            public DynamicOctreeNode(byte position)
            { 
                childCount = 0;
                this.position = position;
                firstChild = null;
                nextNode = null;
                dist = float.MaxValue;           
            }            

            /// <summary>
            /// Returns whether this node has child nodes or not.
            /// </summary>
            /// <returns>true if this node has child nodes, false otherwise.</returns>
            public bool hasChildren()
            {
                return childCount > 0;
            }

            /// <summary>
            /// Returns the child count of this node.
            /// </summary>
            /// <returns>Child count of this node.</returns>
            public byte getChildCount()
            {
                return childCount;
            }          

            /// <summary>
            /// Adds a child with the specified node to this
            /// node, iff there is no child node with position <paramref name="withPos"/> yet.
            /// In there is already such a node, it is returned.
            /// </summary>
            /// <param name="withPos">The position of the child node relative to this node.</param>
            /// <returns>The child of this node with relative position <paramref name="withPos"/>.</returns>
            public DynamicOctreeNode addChild(byte withPos)
            {
                if (childCount == 0)
                { 
                    firstChild = new DynamicOctreeNode(withPos);
                    ++childCount;
                    return firstChild;
                }
                else
                {
                    if (withPos < firstChild.position)
                    {
                        DynamicOctreeNode insert = new DynamicOctreeNode(withPos);
                        insert.nextNode = firstChild;
                        firstChild = insert;
                        ++childCount;
                        return insert;
                    }
                    else if (withPos == firstChild.position)
                        return firstChild;

                    DynamicOctreeNode currentNode = firstChild.nextNode;
                    DynamicOctreeNode previousNode = firstChild;
                    for (int i = 1; i < childCount; ++i)
                    {
                        if (withPos < currentNode.position)
                        {
                            DynamicOctreeNode insert = new DynamicOctreeNode(withPos);
                            previousNode.nextNode = insert;
                            insert.nextNode = currentNode;                           
                            ++childCount;
                            return insert;
                        }
                        else if (withPos == currentNode.position)
                        {
                            return currentNode;
                        }
                        else
                        {
                            currentNode = currentNode.nextNode;
                            previousNode = previousNode.nextNode;
                        }
                    }

                    previousNode.nextNode = new DynamicOctreeNode(withPos);
                    ++childCount;
                    return previousNode.nextNode;
                }
            }

            /// <summary>
            /// Returns the first child of this node.
            /// </summary>
            /// <returns>The first child of this node. Can be null.</returns>
            public DynamicOctreeNode getFirstChild()
            {
                return firstChild;
            }

            /// <summary>
            /// Returns the next child of this node's parent after this node.
            /// </summary>
            /// <returns>The next child of this node's parent after this node. Can be null.</returns>
            public DynamicOctreeNode getNextNode()
            {
                return nextNode;
            }

            /// <summary>
            /// Returns the total number of this node's children + 1.
            /// </summary>
            /// <returns>#children + 1</returns>
            public int getNodeCount()
            {
                int result = 1;

                DynamicOctreeNode child = firstChild;
                for (int i = 0; i < childCount; ++i)
                {
                    result += child.getNodeCount();
                    child = child.nextNode;
                }

                return result;
            }           

            /// <summary>
            /// Returns the total number of leaf nodes beneath this node.
            /// </summary>
            /// <returns>The total number of leaf nodes beneath this node.</returns>
            public uint getLeafCount()
            {
                uint result = childCount == 0 ? 1u : 0u;

                DynamicOctreeNode child = firstChild;
                for (int i = 0; i < childCount; ++i)
                {
                    result += child.getLeafCount();
                    child = child.nextNode;
                }

                return result;
            }

            /// <summary>
            /// Returns the total number of pre-leaf nodes beneath this node.
            /// A pre-leaf is a node, whose children are leaves.
            /// </summary>
            /// <returns>The total number of pre-leaf nodes beneath this node.</returns>
            public uint getPreLeafCount()
            {
                uint result = (childCount > 0 && firstChild.childCount == 0) ? 1u : 0u;

                DynamicOctreeNode child = firstChild;
                for (int i = 0; i < childCount; ++i)
                {
                    result += child.getPreLeafCount();
                    child = child.nextNode;
                }

                return result;
            }
        }
    }
}
