/*
 * SkinnedModelBoneContent.cs
 * Author: Bruno Evangelista
 * Copyright (c) 2008 Bruno Evangelista. All rights reserved.
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
using Microsoft.Xna.Framework.Content.Pipeline.Serialization.Compiler;
using XNAnimation;

namespace XNAnimationPipeline
{
    public class SkinnedModelBoneContent
    {
        private readonly ushort index;
        private readonly string name;
        private SkinnedModelBoneContent parent;
        private SkinnedModelBoneContentCollection children;

        private readonly Pose bindPose;
        private readonly Matrix inverseBindPoseTransform;

        #region Properties

        public string Name
        {
            get { return name; }
        }

        public ushort Index
        {
            get { return index; }
        }

        public SkinnedModelBoneContent Parent
        {
            get { return parent; }
            internal set { parent = value; }
        }

        public SkinnedModelBoneContentCollection Children
        {
            get { return children; }
            internal set { children = value; }
        }

        public Pose BindPose
        {
            get { return bindPose; }
        }

        public Matrix InverseBindPoseTransform
        {
            get { return inverseBindPoseTransform; }
        }

        #endregion

        internal SkinnedModelBoneContent(ushort index, string name, Pose bindPose,
            Matrix inverseBindPoseTransform)
        {
            this.index = index;
            this.name = name;
            this.bindPose = bindPose;
            this.inverseBindPoseTransform = inverseBindPoseTransform;
        }

        internal void Write(ContentWriter output)
        {
            output.Write(index);
            output.Write(name);

            // Write bind pose
            output.Write(bindPose.Translation);
            output.Write(bindPose.Orientation);
            output.Write(bindPose.Scale);

            output.Write(inverseBindPoseTransform);

            // Write parent and children
            output.WriteSharedResource(parent);
            output.Write(children.Count);
            foreach (SkinnedModelBoneContent childBone in children)
                output.WriteSharedResource(childBone);
        }
    }
}