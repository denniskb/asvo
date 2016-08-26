/*
 * AnimationChannelKeyframe.cs
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

namespace XNAnimation
{
    public class AnimationChannelKeyframe
    {
        private readonly TimeSpan time;
        private readonly Pose pose;

        #region Properties

        public TimeSpan Time
        {
            get { return time; }
        }

        public Pose Pose
        {
            get { return pose; }
        }

        #endregion

        internal AnimationChannelKeyframe(TimeSpan time, Pose pose)
        {
            this.time = time;
            this.pose = pose;
        }
    }
}