/*
 * SkinnedModelContent.cs
 * Authors:  Bruno Evangelista
 *           Rodrigo 'r2d2rigo' Díaz
 * Copyright (c) 2008 Bruno Evangelista. All rights reserved.
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
using System.Collections.Generic;
using Microsoft.Xna.Framework.Content;
using Microsoft.Xna.Framework.Content.Pipeline.Graphics;
using Microsoft.Xna.Framework.Content.Pipeline.Processors;
using Microsoft.Xna.Framework.Content.Pipeline.Serialization.Compiler;

namespace XNAnimationPipeline
{
    public class SkinnedModelContent 
    {
        [ContentSerializer(ElementName = "Meshes")]
        private readonly SkinnedModelMeshContentCollection meshes;
             
        [ContentSerializer(ElementName = "Skeleton")]
        private readonly SkinnedModelBoneContentCollection skeleton;

        [ContentSerializer(ElementName = "AnimationClipDictionary")]
        private readonly AnimationClipContentDictionary animationClips;

        internal SkinnedModelContent(SkinnedModelMeshContentCollection meshes, SkinnedModelBoneContentCollection skeleton,
            AnimationClipContentDictionary animationClips)
        {
            this.meshes = meshes;
            this.skeleton = skeleton;
            this.animationClips = animationClips;
        }

        internal void Write(ContentWriter output)
        {
            WriteMeshes(output);
            WriteBones(output);
            WriteAnimationClips(output);
        }
        
        private void WriteMeshes(ContentWriter output)
        {
            output.Write(meshes.Count);
            for (int i = 0; i < meshes.Count; i++)
            {
                output.WriteObject<SkinnedModelMeshContent>(meshes[i]);
            }
        }
        
        private void WriteBones(ContentWriter output)
        {
            output.Write(skeleton.Count);
            foreach (SkinnedModelBoneContent bone in skeleton)
            {
                output.WriteSharedResource(bone);
            }
        }

        private void WriteAnimationClips(ContentWriter output)
        {
            output.Write(animationClips.Count);
            foreach (AnimationClipContent animationClipContent in animationClips.Values)
            { 
                output.WriteSharedResource(animationClipContent);
            }
        }
    }
}