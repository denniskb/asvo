/*
 * AnimationClipContent.cs
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
using System;
using System.Collections.Generic;
using Microsoft.Xna.Framework.Content.Pipeline.Serialization.Compiler;

namespace XNAnimationPipeline
{
    public class AnimationClipContent
    {
        private readonly string name;
        private readonly TimeSpan duration;
        private readonly AnimationChannelContentDictionary channels;

        #region Properties

        public string Name
        {
            get { return name; }
        }

        public TimeSpan Duration
        {
            get { return duration; }
        }

        public AnimationChannelContentDictionary Channels
        {
            get { return channels; }
        }

        #endregion

        internal AnimationClipContent(string name, AnimationChannelContentDictionary channels,
            TimeSpan duration)
        {
            this.name = name;
            this.channels = channels;
            this.duration = duration;
        }

        internal void Write(ContentWriter output)
        {
            output.Write(name);
            output.WriteObject<TimeSpan>(duration);

            // Write animation clip channels
            output.Write(channels.Count);
            foreach (KeyValuePair<string, AnimationChannelContent> pair in channels)
            {
                output.Write(pair.Key);
                AnimationChannelContent animationChannel = pair.Value;

                // Write the animation channel keyframes
                output.Write(animationChannel.Count);
                foreach (AnimationKeyframeContent keyframe in animationChannel)
                {
                    output.WriteObject<TimeSpan>(keyframe.Time);

                    // Write pose
                    output.Write(keyframe.Pose.Translation);
                    output.Write(keyframe.Pose.Orientation);
                    output.Write(keyframe.Pose.Scale);
                }
            }
        }
    }
}