/*
 * SkinnedModelMesh.cs
 * Author: Rodrigo 'r2d2rigo' Díaz
 * Copyright (c) 2010 Rodrigo 'r2d2rigo' Díaz. All rights reserved.
 *
 * THIS SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 * 
 */
using System;
using System.Collections.Generic;
using Microsoft.Xna.Framework;
using Microsoft.Xna.Framework.Content;
using Microsoft.Xna.Framework.Graphics;
using XNAnimation.Effects;

namespace XNAnimation
{
    public class SkinnedModelMesh
    {
        internal int numVertices;
        internal int numTriangles;
        internal VertexBuffer vertices;
        internal IndexBuffer indices;
        internal SkinnedModelBasicEffect effect;

        public VertexBuffer Vertices { get { return vertices; } }

        public IndexBuffer Indices { get { return indices; } }

        public SkinnedModelBasicEffect Effect
        {
            get
            {
                return effect;
            }
        }

        internal SkinnedModelMesh()
        {
        }

        internal static SkinnedModelMesh Read(ContentReader input)
        {
            SkinnedModelMesh skinnedModelPart = new SkinnedModelMesh();

            skinnedModelPart.numVertices = input.ReadInt32();
            skinnedModelPart.numTriangles = input.ReadInt32();
            skinnedModelPart.vertices = input.ReadObject<VertexBuffer>();
            skinnedModelPart.indices = input.ReadObject<IndexBuffer>();
            skinnedModelPart.effect = input.ReadObject<SkinnedModelBasicEffect>();

            return skinnedModelPart;
        }

        public void Draw()
        {
            GraphicsDevice graphicsDevice = vertices.GraphicsDevice;
            graphicsDevice.SetVertexBuffer(vertices);
            graphicsDevice.Indices = indices;

            for (int i = 0; i < effect.CurrentTechnique.Passes.Count; i++)
            {
                effect.CurrentTechnique.Passes[i].Apply();
                graphicsDevice.DrawIndexedPrimitives(PrimitiveType.TriangleList, 0, 0, numVertices, 0, numTriangles);
            }
        }
    }
}