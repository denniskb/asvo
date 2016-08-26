using asvo.datastructures;
using asvo.world3D;
using asvo.tools;

using Microsoft.Xna.Framework;
using Microsoft.Xna.Framework.Graphics;
using Microsoft.Xna.Framework.Content;
using System.Diagnostics;
using System.IO;
using System.Threading;

using asvo.multithreading;

namespace asvo {
    namespace renderers
    {
        /// <summary>
        /// Represents a software rasterizer that can render BFSOctrees.
        /// It supports phong shading and can render models consisting of
        /// 200k - 750k voxels in realtime on multi-core systems.
        /// The rasterizer is used for testing purposes.
        /// 
        /// If a voxel covers many pixels, the rasterizer doesn't color all of them for
        /// performance reasons. It only colors the pixel nearest to the voxel center.
        /// This information is encoded in textures which are sent to the GPU, which then
        /// fills all pixels of the image (via shaders).
        /// </summary>
        internal class Rasterizer
        {
            public readonly int horRes, vertRes, resolution;
            public readonly Color[][] _colorBuffer;
            public readonly SharedDepthBuffer _depthBuffer;            
            private readonly Color fillColor;
            
            private VertexBuffer _vb;
            private Texture2D _colorTexture, _depthTexture;
            private Effect _effect;
            private VertexDeclaration _vd;

            private readonly double pixelWidth, pixelHeight;          

            /// <summary>
            /// Creates a new Rasterizer.
            /// </summary>
            /// <param name="horRes">Horizontal screen resolution.</param>
            /// <param name="vertRes">Vertixal screen resolution.</param>
            public Rasterizer(int horRes, int vertRes)
            {
                this.horRes = horRes;
                this.vertRes = vertRes;
                resolution = horRes * vertRes;
                fillColor = Color.White;

                _colorBuffer = new Color[JobCenter.getWorkerCount()][];
                for (int i = 0; i < JobCenter.getWorkerCount(); ++i)
                    _colorBuffer[i] = new Color[resolution];
               
                _depthBuffer = new SharedDepthBuffer(resolution);               

                pixelWidth = 1.0 / horRes;
                pixelHeight = 1.0 / vertRes;                
            }

            /// <summary>
            /// Called by Game1. Initializes textures, buffers, etc.
            /// </summary>
            public void loadContent(GraphicsDevice device, ContentManager cm)
            {
                _vd = new VertexDeclaration(4 * sizeof(float), VertexFormat.VertexElements);

                _vb = new VertexBuffer(device, _vd, 6, BufferUsage.WriteOnly);
                VertexFormat[] data = new VertexFormat[6];
                data[0].position.X = -1.0f;
                data[0].position.Y = 1.0f;
                data[0].texCoord.X = 0.0f;
                data[0].texCoord.Y = 0.0f;

                data[1].position.X = 1.0f;
                data[1].position.Y = 1.0f;
                data[1].texCoord.X = 1.0f;
                data[1].texCoord.Y = 0.0f;

                data[2].position.X = 1.0f;
                data[2].position.Y = -1.0f;
                data[2].texCoord.X = 1.0f;
                data[2].texCoord.Y = 1.0f;



                data[3].position.X = -1.0f;
                data[3].position.Y = 1.0f;
                data[3].texCoord.X = 0.0f;
                data[3].texCoord.Y = 0.0f;

                data[4].position.X = 1.0f;
                data[4].position.Y = -1.0f;
                data[4].texCoord.X = 1.0f;
                data[4].texCoord.Y = 1.0f;

                data[5].position.X = -1.0f;
                data[5].position.Y = -1.0f;
                data[5].texCoord.X = 0.0f;
                data[5].texCoord.Y = 1.0f;

                _vb.SetData<VertexFormat>(data, 0, 6);                

                _colorTexture = new Texture2D(device, horRes, vertRes,
                                             false, SurfaceFormat.Color);
                
                _depthTexture = new Texture2D(device, horRes, vertRes,
                                             false, SurfaceFormat.Single);                

                cm.RootDirectory = "Content";
                _effect = cm.Load<Effect>("simple");
                _effect.Parameters["pixelWidth"].SetValue((float)pixelWidth);
                _effect.Parameters["pixelHeight"].SetValue((float)pixelHeight);               
                _effect.Parameters["fillColor"].SetValue(fillColor.ToVector4());
                _effect.Parameters["horRes"].SetValue(horRes);
                _effect.Parameters["vertRes"].SetValue(vertRes);
            }

            /// <summary>
            /// Called by Game1, draws the final image. This happens independent of
            /// a call to <see cref="Rasterizer.render"/>.
            /// If this method is called without calling <see cref="Rasterizer.render"/> prior to it,
            /// an empty screen appears.
            /// </summary>     
            public void draw(GraphicsDevice device)
            {
                _colorTexture.SetData<Color>(_colorBuffer[0]);
                _depthTexture.SetData<float>(_depthBuffer._elements[0]);

                _effect.CurrentTechnique = _effect.Techniques["Simplest"];                
                _effect.Parameters["colorTexture"].SetValue(_colorTexture);
                _effect.Parameters["depthTexture"].SetValue(_depthTexture);
                _effect.Parameters["maxDim"].SetValue(_depthBuffer._maxDims[0]);
                         
                foreach (EffectPass pass in _effect.CurrentTechnique.Passes)
                {
                    pass.Apply();
                    device.SetVertexBuffer(_vb);
                    device.DrawPrimitives(PrimitiveType.TriangleList, 0, 2);
                }

                device.Textures[0] = null;
                device.Textures[1] = null;
            }

            /// <summary>
            /// Vertex format used to describe the rendering quad this rasterizer uses.
            /// </summary>
            private struct VertexFormat
            {
                public Vector2 position;
                public Vector2 texCoord;

                public static VertexElement[] VertexElements = {
                    new VertexElement(0, VertexElementFormat.Vector2, VertexElementUsage.Position, 0),
                    new VertexElement(8, VertexElementFormat.Vector2, VertexElementUsage.TextureCoordinate, 0)         
                };

                public static int sizeInBytes = 16;
            }

            /// <summary>
            /// Renders the sparse voxel octree stored in <paramref name="obj"/>.
            /// 
            /// - Viewing frustum culling
            /// - Level of detail
            /// </summary>
            /// <param name="object3D">The object to be rendered.</param>
            /// <param name="fromPerspective">The camera this model shall be rendered from.</param>
            /// <param name="threadIndex">The index of the calling thread, starting by 0.</param>
            public void render(Object3D object3D, Camera fromPerspective, int threadIndex)
            {
                _depthBuffer.zeroOut(threadIndex);
                float dimension = -0.5f * object3D.getData().dimension;
                Vector3 octreeMin = new Vector3(dimension, dimension, dimension);

                traverse(object3D,
                         0,
                         0, 0, 0,                         
                         fromPerspective.getProjectionMatrix(),
                         object3D.getTransformation() *
                         fromPerspective.getViewMatrix(),
                         object3D.getTransformation() *
                         fromPerspective.getViewProjectionMatrix(),
                         0,
                         0.0f,
                         threadIndex,
                         octreeMin);                
            }

            /// <summary>
            /// Traverses all octree nodes and draws them. Used by <see cref="Rasterizer.render"/>.
            /// </summary>
            private void traverse(Object3D obj,
                                  uint offset,
                                  int x, int y, int z,
                                  Matrix projection,
                                  Matrix worldView,
                                  Matrix worldViewProjection,
                                  byte level,
                                  float halfDim,
                                  int threadIndex,
                                  Vector3 octreeMin)
            {
                if (level == 2 && offset % JobCenter.getWorkerCount() != threadIndex)
                    return;            

                int gridResolution = 1 << level;
                float gridDimension = obj.getData().dimension;
             
                Vector3 xyzTimesGridDim = new Vector3(x, y, z) * gridDimension;
                float gridResolutionMul = 1.0f / gridResolution;
                Vector3 min = octreeMin + xyzTimesGridDim * gridResolutionMul;

                xyzTimesGridDim += new Vector3(gridDimension);
                Vector3 max = octreeMin + xyzTimesGridDim * gridResolutionMul;

                float dimension = max.X - min.X;
                bool subDivide = false;
                Vector3 center = Vector3.Lerp(min, max, 0.5f);
                Vector3 center2 = center + new Vector3(0, 0, dimension);

                VisualData vd = offset >= obj.getData()._innerNodes.Length ?
                    obj.getData()._leaves[offset - obj.getData()._innerNodes.Length].visualData :
                    obj.getData()._innerNodes[offset].visualData;
                
                Vector3 posIn, posOut;
                posOut = Vector3.Zero;
                if (vd.boneWeights.X > 0.0f)
                {
                    posIn = center;
                    Math3DHelper.mul(ref posIn, ref obj.getData()._animation[obj.frame, vd.boneIndex0]);
                    posOut += vd.boneWeights.X * posIn;
                }
                if (vd.boneWeights.Y > 0.0f)
                {
                    posIn = center;
                    Math3DHelper.mul(ref posIn, ref obj.getData()._animation[obj.frame, vd.boneIndex1]);
                    posOut += vd.boneWeights.Y * posIn;
                }
                if (vd.boneWeights.Z > 0.0f)
                {
                    posIn = center;
                    Math3DHelper.mul(ref posIn, ref obj.getData()._animation[obj.frame, vd.boneIndex2]);
                    posOut += vd.boneWeights.Z * posIn;
                }
                if (vd.boneWeights.W > 0.0f)
                {
                    posIn = center;
                    Math3DHelper.mul(ref posIn, ref obj.getData()._animation[obj.frame, vd.boneIndex3]);
                    posOut += vd.boneWeights.W * posIn;
                }
                
                center = posOut;                             
                
                if (offset >= obj.getData()._innerNodes.Length)
                {                    
                    Math3DHelper.mul(ref center, ref worldViewProjection);
                    halfDim *= 0.5f * dimension;                    
                } 
                else
                {                    
                    Math3DHelper.mul(ref center, ref worldView);

                    Vector3 dimVec = new Vector3(0.5f * dimension, 0, center.Z);

                    Math3DHelper.mul(ref center, ref projection);

                    Math3DHelper.mul(ref dimVec, ref projection);                   
                    halfDim = dimVec.X;

                    subDivide = halfDim > pixelWidth;
                }     
             
                if (subDivide)
                {
                    float minusOneminusHalfDim = -1.0f - halfDim;
                    float onePlusHalfDim = 1.0f + halfDim;

                    if (center.X < minusOneminusHalfDim || center.X > onePlusHalfDim ||
                        center.Y < minusOneminusHalfDim || center.Y > onePlusHalfDim ||
                        center.Z < minusOneminusHalfDim + 1.0f || center.Z > onePlusHalfDim)
                        return;

                    Vector3 halfVec = 0.5f * (max - min);

                    byte childIndex = 0; uint mask, index = 0;
                    mask = obj.getData()._innerNodes[offset].childMask;
                    index = obj.getData()._innerNodes[offset].childIndex;
                    

                    for (int i = 0; i < 8; ++i)
                    {
                        if ((mask & (byte)(1u << i)) != 0)
                        {                    
                            traverse(obj,
                                     index + childIndex,
                                     x * 2 +  (i & 1),
                                     y * 2 + ((i & 2) >> 1),
                                     z * 2 + ((i & 4) >> 2),
                                     projection,
                                     worldView,
                                     worldViewProjection,
                                     (byte)(level + 1),
                                     halfDim,
                                     threadIndex,
                                     octreeMin);

                            ++childIndex;
                        }
                    }
                }
                else
                {
                    center.X = center.X * 0.5f + 0.5f;
                    center.Y = 1.0f - (center.Y * 0.5f + 0.5f);
                                        
                    int centerX, centerY;
                    centerX = (int)Math3DHelper.round(center.X * horRes);
                    centerY = (int)Math3DHelper.round(center.Y * vertRes);

                    if (centerX < 0 || centerX >= horRes ||
                        centerY < 0 || centerY >= vertRes)
                        return;

                    int index = centerY * horRes + centerX;

                    Vector3 normal;
                    if (offset < obj.getData()._innerNodes.Length)                    
                        normal = obj.getData()._innerNodes[offset].visualData.normal;                    
                    else                    
                        normal = obj.getData()._leaves[offset - obj.getData()._innerNodes.Length].visualData.normal;                

                    Matrix rotMatrix = obj.getRotation();
                    Math3DHelper.mul(ref normal, ref rotMatrix);          

                    byte color = (byte) MathHelper.Max(0, Vector3.Dot(normal, Vector3.UnitY) * 255);

                    if (center.Z < _depthBuffer._elements[threadIndex][index])
                    {
                        if (halfDim > _depthBuffer._maxDims[threadIndex])
                            _depthBuffer._maxDims[threadIndex] = halfDim;

                        _colorBuffer[threadIndex][index].R = color;
                        _colorBuffer[threadIndex][index].G = color;
                        _colorBuffer[threadIndex][index].B = color;

                        _depthBuffer._elements[threadIndex][index] = center.Z;  
                    }               
                }
            }
        }
    }
}
