/*
 * AnimationClip.cs
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
using Microsoft.Xna.Framework.Content;

namespace XNAnimation
{
    public class AnimationClip
    {
        private readonly string name;
        private readonly TimeSpan duration;
        private readonly AnimationChannelDictionary channels;

        #region Properties

        public string Name
        {
            get { return name; }
        }

        public TimeSpan Duration
        {
            get { return duration; }
        }

        public AnimationChannelDictionary Channels
        {
            get { return channels; }
        }

        #endregion

        internal AnimationClip(string name, TimeSpan duration, AnimationChannelDictionary channels)
        {
            this.name = name;
            this.duration = duration;
            this.channels = channels;
        }

        internal static AnimationClip Read(ContentReader input)
        {
            string animationName = input.ReadString();
            TimeSpan animationDuration = input.ReadObject<TimeSpan>();

            // Read animation clip channels
            Dictionary<string, AnimationChannel> animationChannelDictionary =
                new Dictionary<string, AnimationChannel>();

            int numAnimationChannels = input.ReadInt32();
            for (int i = 0; i < numAnimationChannels; i++)
            {
                string channelName = input.ReadString();

                // Read animation channel keyframes
                int numChannelKeyframes = input.ReadInt32();
                List<AnimationChannelKeyframe> keyframeList =
                    new List<AnimationChannelKeyframe>(numChannelKeyframes);

                for (int j = 0; j < numChannelKeyframes; j++)
                {
                    TimeSpan keyframeTime = input.ReadObject<TimeSpan>();

                    // Read keyframe pose
                    Pose keyframePose;
                    keyframePose.Translation = input.ReadVector3();
                    keyframePose.Orientation = input.ReadQuaternion();
                    keyframePose.Scale = input.ReadVector3();

                    keyframeList.Add(new AnimationChannelKeyframe(keyframeTime, keyframePose));
                }

                AnimationChannel animationChannel = new AnimationChannel(keyframeList);

                // Add the animation channel to the dictionary
                animationChannelDictionary.Add(channelName, animationChannel);
            }

            return
                new AnimationClip(animationName, animationDuration,
                    new AnimationChannelDictionary(animationChannelDictionary));
        }
    }
}