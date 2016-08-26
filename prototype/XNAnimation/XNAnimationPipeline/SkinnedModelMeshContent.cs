/*
 * SkinnedModelMeshContent.cs
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
using Microsoft.Xna.Framework;
using Microsoft.Xna.Framework.Content.Pipeline.Graphics;
using Microsoft.Xna.Framework.Content.Pipeline.Processors;
using Microsoft.Xna.Framework.Content.Pipeline.Serialization.Compiler;

namespace XNAnimationPipeline
{
    public class SkinnedModelMeshContent
    {
        private int numVertices;
        private int numTriangles;
        private VertexBufferContent vertices;
        private IndexCollection indices;
        private MaterialContent material;

        internal SkinnedModelMeshContent(int numVertices, int numTriangles, VertexBufferContent vertices,
            IndexCollection indices, MaterialContent material)
        {
            this.numVertices = numVertices;
            this.numTriangles = numTriangles;
            this.vertices = vertices;
            this.indices = indices;
            this.material = material;
        }

        internal void Write(ContentWriter output)
        {
            output.Write(numVertices);
            output.Write(numTriangles);
            output.WriteObject<VertexBufferContent>(vertices);
            output.WriteObject<IndexCollection>(indices);
            output.WriteObject<MaterialContent>(material);
        }
    }
}
