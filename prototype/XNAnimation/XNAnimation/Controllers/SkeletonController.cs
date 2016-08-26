/*
 * SkeletonController.cs
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
namespace XNAnimation.Controllers
{
    /// <summary>
    /// Controls the pose of each bone in a skeleton. Allows custom skeleton poses to be blended 
    /// with other <see cref="T:XNAnimation.Controllers.IBlendable" /> objects.
    /// </summary>
    public class SkeletonController : ISkeletonController, IBlendable
    {
        private SkinnedModelBoneDictionary skeletonDictionary;
        private Pose[] localBonePoses;
        private float blendWeight;

        #region Properties

        /// <inheritdoc />
        public Pose[] LocalBonePoses
        {
            get { return localBonePoses; }
        }

        /// <inheritdoc />
        public float BlendWeight
        {
            get { return blendWeight; }
            set { blendWeight = value; }
        }

        #endregion

        /// <summary>Initializes a new instance of the 
        /// <see cref="T:XNAnimation.Controllers.SkeletonController"></see>
        /// class.
        /// </summary>
        /// <param name="skeletonDictionary"></param>
        public SkeletonController(SkinnedModelBoneDictionary skeletonDictionary)
        {
            this.skeletonDictionary = skeletonDictionary;
            localBonePoses = new Pose[skeletonDictionary.Count];

            blendWeight = 1.0f;
        }

        /// <inheritdoc />
        public void SetBonePose(string channelName, ref Pose pose)
        {
            localBonePoses[skeletonDictionary[channelName].Index] = pose;
        }

        /// <inheritdoc />
        public void SetBonePose(string channelName, Pose pose)
        {
            localBonePoses[skeletonDictionary[channelName].Index] = pose;
        }
    }
}