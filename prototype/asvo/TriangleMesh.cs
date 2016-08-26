using Microsoft.Xna.Framework;
using Microsoft.Xna.Framework.Content;
using Microsoft.Xna.Framework.Graphics;
using System.IO;
using System;
using System.Collections.Generic;

using XNAnimation;
using XNAnimation.Controllers;
using XNAnimation.Pipeline;

using asvo.datastructures;
using asvo.tools;

namespace asvo
{
    namespace world3D
    {
        /// <summary>
        /// Used for loading .fbx files.
        /// </summary>
        internal struct FBXVertexFormat
        {
            public Vector3 position;
            public byte boneIndex0;
            public byte boneIndex1;
            public byte boneIndex2;
            public byte boneIndex3;
            public Vector4 boneWeights;
            public Vector3 normal;
            public Vector2 texCoord;
        }

        /// <summary>
        /// Represents a standard triangle mesh consisting of
        /// vertices and faces.
        /// A vertex stores
        /// - a positions
        /// - bone weights and indices
        /// - a normal
        /// - texture coordinates
        ///         
        /// TriangleMeshes are loaded from 3dsmax .fbx files. The
        /// following restrictions apply to *accepted* .fbx files:
        /// - The mesh has to have a material assigned to it.
        /// - The mesh has to be skinned with at least 1 bone.
        /// - The file has to contain exactly one animation called "Take 001" in
        ///   which every(!) bone of the mesh is animated.
        /// - The mesh has to contain >= 2^16 vertices and indices.
        ///   
        /// A TriangleMesh can also convert itself to a DynamicOctree.
        /// </summary>
        internal class TriangleMesh
        {
            private FBXVertexFormat[] _vertices;
            private int[] _indices;
            private SkinnedModel _model;
            private static float[] boneWeights;
            private static byte[] boneIndices;

            /// <summary>
            /// Constructs a triangle mesh from a .fbx file.
            /// See <see cref="TriangleMesh"/> on which restrictions apply to it.
            /// </summary>
            /// <param name="fileName">Path to the .fbx file.</param>
            /// <param name="content">ContentManager used to load .fbx files.</param>
            public TriangleMesh(string fileName, ContentManager content)
            {
                if (boneWeights == null)
                    boneWeights = new float[256];

                if (boneIndices == null)
                    boneIndices = new byte[256];

                createFromFBX(fileName, content);               
            }

            /// <summary>
            /// Reads a .fbx file and creates the triangle mesh out of it.
            /// </summary>
            /// <param name="fileName">Path to the .fbx file.</param>
            /// <param name="content">ContentManager used to load .fbx files.</param>
            private void createFromFBX(string fileName, ContentManager content)
            {
                _model = content.Load<SkinnedModel>(fileName);
                _vertices = new FBXVertexFormat[_model.Meshes[0].Vertices.VertexCount];
                _indices = new int[_model.Meshes[0].Indices.IndexCount];
                _model.Meshes[0].Vertices.GetData<FBXVertexFormat>(_vertices);
                _model.Meshes[0].Indices.GetData<int>(_indices);                           
               
                for (int i = 0; i < _vertices.Length; ++i)
                {
                    _vertices[i].normal.Normalize();
                }
            }            

            /// <summary>
            /// Converts this triangle mesh into a DynamicOctree.
            /// !Important: Although nodes can have as few as 1 and as many as 8
            /// children, all leavs have the same depth. This is an important property that
            /// allows the created octree to be rendred on a CUDA implementation of asvo.
            /// </summary>
            /// <param name="maxLevel">The maximum node level/depth of any node in
            /// the created DynamicOctree.</param>
            /// <returns>A DynamicOctree representing this triangle mesh.</returns>
            private DynamicOctree toDynamicOctree(byte maxLevel)
            {
                DynamicOctree result;
                Vector3 min, max, center;
                float minX, minY, minZ, maxX, maxY, maxZ;
                double dimension, halfDimension;
                minX = minY = minZ = float.MaxValue;
                maxX = maxY = maxZ = float.MinValue;

                for (int i = 0; i < _vertices.Length; ++i)
                {
                    if (_vertices[i].position.X < minX)
                        minX = _vertices[i].position.X;
                    if (_vertices[i].position.X > maxX)
                        maxX = _vertices[i].position.X;

                    if (_vertices[i].position.Y < minY)
                        minY = _vertices[i].position.Y;
                    if (_vertices[i].position.Y > maxY)
                        maxY = _vertices[i].position.Y;

                    if (_vertices[i].position.Z < minZ)
                        minZ = _vertices[i].position.Z;
                    if (_vertices[i].position.Z > maxZ)
                        maxZ = _vertices[i].position.Z;
                }

                dimension = ((double)maxX) - ((double)minX);
                if ((((double)maxY) - ((double)minY)) > dimension)
                    dimension = ((double)maxY) - ((double)minY);
                if ((((double)maxZ) - ((double)minZ)) > dimension)
                    dimension = ((double)maxZ) - ((double)minZ);

                halfDimension = dimension * 0.5;
                center = new Vector3((maxX - minX) * 0.5f + minX,
                                     (maxY - minY) * 0.5f + minY,
                                     (maxZ - minZ) * 0.5f + minZ);

                min = center - new Vector3((float)halfDimension);
                max = center + new Vector3((float)halfDimension);

                result = new DynamicOctree(max.X - min.X);

                Vector3[] triangle = new Vector3[3];                
                for (int i = 0; i < _indices.Length - 2; i += 3)
                {
                    minX = minY = minZ = float.MaxValue;
                    maxX = maxY = maxZ = float.MinValue;

                    for (int j = 0; j < 3; ++j)
                    {
                        triangle[j] = _vertices[_indices[i + j]].position;

                        if (triangle[j].X < minX)
                            minX = triangle[j].X;
                        if (triangle[j].X > maxX)
                            maxX = triangle[j].X;

                        if (triangle[j].Y < minY)
                            minY = triangle[j].Y;
                        if (triangle[j].Y > maxY)
                            maxY = triangle[j].Y;

                        if (triangle[j].Z < minZ)
                            minZ = triangle[j].Z;
                        if (triangle[j].Z > maxZ)
                            maxZ = triangle[j].Z;
                    }

                    traverse(0,
                             0, 0, 0,
                             min, dimension,
                             i,
                             new Vector3(minX, minY, minZ),
                             new Vector3(maxX, maxY, maxZ),
                             0, maxLevel,
                             result);
                }

                _vertices = null;
                _indices = null;                

                return result;
            }

            /// <summary>
            /// Converts this trianlge mesh into a BFSOctree.
            /// </summary>
            /// <param name="maxLevel">The maximum node level/depth of any node in
            /// the created DynamicOctree.</param>
            /// <returns>A BFSOctree representing this triangle mesh.</returns>
            public BFSOctree toBFSOctree(byte maxLevel)
            {
                System.Collections.Generic.IEnumerator<AnimationClip> enum1;
                System.Collections.Generic.IEnumerator<AnimationChannel> enum2;
                enum1 = _model.AnimationClips.Values.GetEnumerator();
                enum1.MoveNext();
                enum2 = enum1.Current.Channels.Values.GetEnumerator();
                enum2.MoveNext();
                Matrix[,] animation = new Matrix[enum2.Current.Count, enum1.Current.Channels.Count];

                AnimationClip clip;
                _model.AnimationClips.TryGetValue("Take 001", out clip);
                AnimationController c = new AnimationController(_model.SkeletonBones);                
                c.StartClip(clip);
             
                for (long i = 0; i < enum2.Current.Count; ++i)
                {
                    c.Update(new TimeSpan(clip.Duration.Ticks / enum2.Current.Count), Matrix.Identity);
                    for (int j = 0; j < enum1.Current.Channels.Count; ++j)
                        animation[i, j] = c.SkinnedBoneTransforms[j];
                }

                _model = null;

                BFSOctree result = new BFSOctree(toDynamicOctree(maxLevel));
                result._animation = animation;
                result.frameCount = (uint)enum2.Current.Count;
                result.boneCount = (uint)enum1.Current.Channels.Count;
                                             
                return result;
            }

            /// <summary>
            /// Helper method used by <see cref="TriangleMesh.toDynamicOctree"/>.
            /// It constructs the DynamicOctree recursively. It is supplied a
            /// triangle and adds all nodes to the octree that intersect the triangle.
            /// </summary>
            /// <param name="pos">Position of the current node relative to its parent.
            /// See <see cref="DynamicOctree"/></param>
            /// <param name="x">x-coordinate within the virtual uniform grid spanned by the octree.</param>
            /// <param name="y">y-coordinate within the virtual uniform grid spanned by the octree.</param>
            /// <param name="z">z-coordinate within the virtual uniform grid spanned by the octree.</param>
            /// <param name="octreeMin">Minimum vector of the octree.</param>
            /// <param name="gridDimension">Dimension of the octree (a cube).</param>
            /// <param name="index">Index of the first triangle vertex.</param>
            /// <param name="triBBmin">Minimum vector of the triangle bounding box.</param>
            /// <param name="triBBmax">Maximum vector of the triangle bounding box.</param>
            /// <param name="level">Current tree level/depth.</param>
            /// <param name="maxLevel">The maximum node level/depth of any node in
            /// the created DynamicOctree.</param>
            /// <param name="parent">Parent node of the current node.</param>
            private void traverse(byte pos,
                                  ushort x, ushort y, ushort z,
                                  Vector3 octreeMin, double gridDimension,                            
                                  int index,
                                  Vector3 triBBmin, Vector3 triBBmax,                  
                                  byte level, byte maxLevel,
                                  DynamicOctreeNode parent)
            {
                if (level == 0)
                {
                    for (byte i = 0; i < 8; ++i)
                    {                        
                        traverse(i,
                                 (ushort)(x * 2 + (i & 1)),
                                 (ushort)(y * 2 + ((i & 2) >> 1)),
                                 (ushort)(z * 2 + ((i & 4) >> 2)),
                                 octreeMin, gridDimension,    
                                 index,
                                 triBBmin, triBBmax,
                                 (byte)(level + 1), maxLevel,
                                 parent);
                    }
                }
                else
                {
                    if (Math3DHelper.intersects(x, y, z, level, gridDimension,
                                                octreeMin,
                                                _vertices[_indices[index]].position,
                                                _vertices[_indices[index + 1]].position,
                                                _vertices[_indices[index + 2]].position,
                                                triBBmin, triBBmax))
                    {
                        DynamicOctreeNode newParent = parent.addChild(pos);                    
                                               
                        float gridCellDim = (float)(gridDimension / (1u << level));
                        float gridCellHalfDim = gridCellDim * .5f;
                        Vector3 center = new Vector3(x * gridCellDim + gridCellHalfDim,
                                                     y * gridCellDim + gridCellHalfDim,
                                                     z * gridCellDim + gridCellHalfDim) +
                                                     octreeMin;

                        float triArea = Vector3.Cross(_vertices[_indices[index + 1]].position -
                                                      _vertices[_indices[index]].position,
                                                      _vertices[_indices[index + 2]].position -
                                                      _vertices[_indices[index]].position).Length() * 0.5f;

                        float u = Vector3.Cross(_vertices[_indices[index + 1]].position -
                                                center,
                                                _vertices[_indices[index + 2]].position -
                                                center).Length() * 0.5f;

                        float v = Vector3.Cross(_vertices[_indices[index]].position -
                                                center,
                                                _vertices[_indices[index + 2]].position -
                                                center).Length() * 0.5f;

                        float w = Vector3.Cross(_vertices[_indices[index]].position -
                                                center,
                                                _vertices[_indices[index + 1]].position -
                                                center).Length() * 0.5f;

                        float sum;
                        if ((sum = u + v + w) < newParent.dist)
                        {
                            newParent.dist = sum;
                            triArea = 1.0f / triArea;
                            u *= triArea;
                            v *= triArea;
                            w *= triArea;
                            sum = 1.0f / (u + v + w);
                            u *= sum;
                            v *= sum;
                            w *= sum;
                            newParent.vd.normal = u * _vertices[_indices[index]].normal +
                                                  v * _vertices[_indices[index + 1]].normal +
                                                  w * _vertices[_indices[index + 2]].normal;
                            newParent.vd.normal.Normalize();

                            newParent.vd.texCoords = u * _vertices[_indices[index]].texCoord +
                                                     v * _vertices[_indices[index + 1]].texCoord +
                                                     w * _vertices[_indices[index + 2]].texCoord; 

                            weight(newParent, _vertices, _indices, index, u, v, w);

                            Vector3 e1 = _vertices[_indices[index + 1]].position - _vertices[_indices[index]].position;
                            Vector3 e2 = _vertices[_indices[index + 2]].position - _vertices[_indices[index]].position;
                            Vector2 euv1 = _vertices[_indices[index + 1]].texCoord - _vertices[_indices[index]].texCoord;
                            Vector2 euv2 = _vertices[_indices[index + 2]].texCoord - _vertices[_indices[index]].texCoord;

                            float cp = euv1.Y * euv2.X - euv1.X * euv2.Y;
                            if (cp != 0.0f)
                            {
                                newParent.vd.tangent = (e1 * (-euv2.Y) + e2 * euv1.Y) / cp;
                                newParent.vd.tangent.Normalize();
                            }
                        }                                                          

                        if (level < maxLevel)
                        {
                            for (byte i = 0; i < 8; ++i)
                            {
                                traverse(i,
                                         (ushort)(x * 2 + (i & 1)),
                                         (ushort)(y * 2 + ((i & 2) >> 1)),
                                         (ushort)(z * 2 + ((i & 4) >> 2)),
                                         octreeMin, gridDimension,
                                         index,
                                         triBBmin, triBBmax,
                                         (byte)(level + 1), maxLevel,
                                         newParent);
                            }
                        }
                    }
                }
            }

            /// <summary>
            /// Helper function used to assign bone weights to voxels.
            /// Since a voxel is approximated by a triangle, in the worst case,
            /// 12 bone weights could affect it. This function determines the 4 most
            /// influential bones for a voxel and assigns them to it.
            /// </summary>
            /// <param name="node">The node to assign bone weights to.</param>
            /// <param name="vertices">Vertex buffer of the triangle mesh.</param>
            /// <param name="indices">Index buffer of the triangle mesh.</param>
            /// <param name="index">Index of the first vertex of the triangle.</param>
            /// <param name="u">Barycentric coordinate of the triangle's first vertex.</param>
            /// <param name="v">Barycentric coordinate of the triangle's second vertex.</param>
            /// <param name="w">Barycentric coordinate of the triangle's third vertex.</param>
            private static void weight(DynamicOctreeNode node,
                                       FBXVertexFormat[] vertices,
                                       int[] indices, int index,
                                       float u, float v, float w)
            {
                for (int i = 0; i < 256; ++i)
                {
                    boneWeights[i] = 0.0f;
                    boneIndices[i] = (byte) i;
                }
                
                boneWeights[vertices[indices[index]].boneIndex0] += vertices[indices[index]].boneWeights.X * u;
                boneWeights[vertices[indices[index]].boneIndex1] += vertices[indices[index]].boneWeights.Y * u;
                boneWeights[vertices[indices[index]].boneIndex2] += vertices[indices[index]].boneWeights.Z * u;
                boneWeights[vertices[indices[index]].boneIndex3] += vertices[indices[index]].boneWeights.W * u;

                boneWeights[vertices[indices[index + 1]].boneIndex0] += vertices[indices[index + 1]].boneWeights.X * v;
                boneWeights[vertices[indices[index + 1]].boneIndex1] += vertices[indices[index + 1]].boneWeights.Y * v;
                boneWeights[vertices[indices[index + 1]].boneIndex2] += vertices[indices[index + 1]].boneWeights.Z * v;
                boneWeights[vertices[indices[index + 1]].boneIndex3] += vertices[indices[index + 1]].boneWeights.W * v;

                boneWeights[vertices[indices[index + 2]].boneIndex0] += vertices[indices[index + 2]].boneWeights.X * w;
                boneWeights[vertices[indices[index + 2]].boneIndex1] += vertices[indices[index + 2]].boneWeights.Y * w;
                boneWeights[vertices[indices[index + 2]].boneIndex2] += vertices[indices[index + 2]].boneWeights.Z * w;
                boneWeights[vertices[indices[index + 2]].boneIndex3] += vertices[indices[index + 2]].boneWeights.W * w;
                
                Array.Sort(boneWeights, boneIndices);
                float sum = 1.0f / (boneWeights[255] + boneWeights[254] + boneWeights[253] + boneWeights[252]);

                node.vd.boneIndex0 = boneIndices[255];
                node.vd.boneIndex1 = boneIndices[254];
                node.vd.boneIndex2 = boneIndices[253];
                node.vd.boneIndex3 = boneIndices[252];

                node.vd.boneWeights.X = boneWeights[255] * sum;
                node.vd.boneWeights.Y = boneWeights[254] * sum;
                node.vd.boneWeights.Z = boneWeights[253] * sum;
                node.vd.boneWeights.W = boneWeights[252] * sum;
            }
        }
    }
}
