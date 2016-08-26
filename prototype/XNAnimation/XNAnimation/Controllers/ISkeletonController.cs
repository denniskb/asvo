/*
 * ISkeletonController.cs
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
    /// Defines an interface for an skeleton controller.
    /// </summary>
    public interface ISkeletonController
    {
        /// <summary>
        /// Gets the local pose of all skeleton's bones in depth-first order.
        /// </summary>
        Pose[] LocalBonePoses { get; }

        /// <summary>
        /// Sets a custom pose for an skeleton's bone.
        /// </summary>
        /// <param name="channelName">The name of the bone.</param>
        /// <param name="pose">The custom pose to be set.</param>
        void SetBonePose(string channelName, ref Pose pose);

        /// <summary>
        /// Sets a custom pose for an skeleton's bone.
        /// </summary>
        /// <param name="channelName">The name of the bone.</param>
        /// <param name="pose">The custom pose to be set.</param>
        void SetBonePose(string channelName, Pose pose);
    }
}