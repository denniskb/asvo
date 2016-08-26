/*
 * IAnimationController.cs
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
using Microsoft.Xna.Framework;

namespace XNAnimation.Controllers
{
    /// <summary>
    /// Specifies how translations, orientations and scales are interpolated between keyframes.
    /// </summary>
    public enum InterpolationMode
    {
        /// <summary>
        /// Does not use interpolation.
        /// </summary>
        None,

        /// <summary>
        /// Linear interpolation. Supported on translations and scales.
        /// </summary>
        Linear,

        /// <summary>
        /// Cubic interpolation. Supported on translations and scales.
        /// </summary>
        Cubic,

        /// <summary>
        /// Spherical interpolation. Only supported on orientations.
        /// </summary>
        Spherical
    } ;

    /// <summary>
    /// Specifies how an animation clip is played.
    /// </summary>
    public enum PlaybackMode
    {
        /// <summary>
        /// Plays the animation clip in the forward way.
        /// </summary>
        Forward,

        /// <summary>
        /// Plays the animation clip in the backward way.
        /// </summary>
        Backward
    } ;

    /// <summary>
    /// Defines an interface for an animation controller.
    /// </summary>
    public interface IAnimationController
    {
        /// <summary>
        /// Gets the animation clip being played.
        /// </summary>
        AnimationClip AnimationClip { get; }

        /// <summary>
        /// Gets os sets the current animation playback time.
        /// </summary>
        TimeSpan Time { get; set; }

        /// <summary>
        /// Gets os sets the animation playback speed.
        /// </summary>
        float Speed { get; set; }

        /// <summary>
        /// Enables animation looping.
        /// </summary>
        bool LoopEnabled { get; set; }

        /// <summary>
        /// Gets os sets the animation playback mode.
        /// </summary>
        PlaybackMode PlaybackMode { get; set; }

        /// <summary>
        /// Gets os sets how translations are interpolated between animation keyframes.
        /// Supports linear and cubic interpolation.
        /// </summary>
        InterpolationMode TranslationInterpolation { get; set; }

        /// <summary>
        /// Gets os sets how orientations are interpolated between animation keyframes.
        /// Supports linear and spherical interpolation.
        /// </summary>
        InterpolationMode OrientationInterpolation { get; set; }

        /// <summary>
        /// Gets os sets how scales are interpolated between animation keyframes.
        /// Supports linear and cubic interpolation.
        /// </summary>
        InterpolationMode ScaleInterpolation { get; set; }

        /// <summary>
        /// Returns whether the animation has finished.
        /// </summary>
        bool HasFinished { get; }

        /// <summary>
        /// Returns whether the animation is playing.
        /// </summary>
        bool IsPlaying { get; }

        /// <summary>
        /// Gets the local pose of all skeleton's bones in depth-first order.
        /// </summary>
        Pose[] LocalBonePoses { get; }

        /// <summary>
        /// Gets the final transformation of all skeleton's bonse in depth-first order.
        /// This transformation is used to transfom the model's mesh vertices.
        /// </summary>
        Matrix[] SkinnedBoneTransforms { get; }

        /// <summary>
        /// Starts the playback of an animation clip from the beginning.
        /// </summary>
        /// <param name="animationClip">The animation clip to be played.</param>
        void StartClip(AnimationClip animationClip);

        /// <summary>
        /// Plays an animation clip.
        /// </summary>
        /// <param name="animationClip">The animation clip to be played.</param>
        void PlayClip(AnimationClip animationClip);

        /// <summary>
        /// Interpolates linearly between two animation clips, fading out the current 
        /// animation clip and fading in a new one.
        /// </summary>
        /// <param name="animationClip">The animation clip to be faded in.</param>
        /// <param name="fadeTime">Time used to fade in and out the animation clips.</param>
        void CrossFade(AnimationClip animationClip, TimeSpan fadeTime);

        /// <summary>
        /// Interpolates between two animation clips, fading out the current animation clip 
        /// and fading in a new one.
        /// </summary>
        /// <param name="animationClip">The animation clip to be faded in.</param>
        /// <param name="fadeTime">Time used to fade in and out the animation clips.</param>
        /// <param name="translationInterpolation">How translations are interpolated between animation clips.</param>
        /// <param name="orientationInterpolation">How orientations are interpolated between animation clips.</param>
        /// <param name="scaleInterpolation">How scales are interpolated between animation clips.</param>
        void CrossFade(AnimationClip animationClip, TimeSpan fadeTime,
            InterpolationMode translationInterpolation, InterpolationMode orientationInterpolation,
            InterpolationMode scaleInterpolation);

        /// <summary>
        /// Updates the animation clip time and calculates the new skeleton's bone pose.
        /// </summary>
        /// <param name="elapsedTime">Time elapsed since the last update.</param>
        /// <param name="parent">The parent bone for the current skeleton's root bone.</param>
        void Update(TimeSpan elapsedTime, Matrix parent);
    }
}