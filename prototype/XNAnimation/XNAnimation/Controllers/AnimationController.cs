/*
 * AnimationController.cs
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
    /// Controls how animations are played and interpolated.
    /// </summary>
    public class AnimationController : IAnimationController, IBlendable
    {
        private SkinnedModelBoneCollection skeleton;
        private Pose[] localBonePoses;
        private Matrix[] skinnedBoneTransforms;
        private Quaternion[] boneControllers;

        private AnimationClip animationClip;
        private TimeSpan time;
        private float speed;
        private bool loopEnabled;
        private PlaybackMode playbackMode;

        private float blendWeight;

        // Interpolation mode fields
        private InterpolationMode translationInterpolation;
        private InterpolationMode orientationInterpolation;
        private InterpolationMode scaleInterpolation;

        // CrossFade fields
        private bool crossFadeEnabled;
        private AnimationClip crossFadeAnimationClip;
        private float crossFadeInterpolationAmount;
        private TimeSpan crossFadeTime;
        private TimeSpan crossFadeElapsedTime;
        private InterpolationMode crossFadeTranslationInterpolation;
        private InterpolationMode crossFadeOrientationInterpolation;
        private InterpolationMode crossFadeScaleInterpolation;

        private bool hasFinished;
        private bool isPlaying;

        #region Properties

        /// <inheritdoc />
        public AnimationClip AnimationClip
        {
            get { return animationClip; }
        }

        /// <inheritdoc />
        public TimeSpan Time
        {
            get { return time; }
            set { time = value; }
        }

        /// <inheritdoc />
        public float Speed
        {
            get { return speed; }
            set
            {
                if (speed < 0)
                    throw new ArgumentException("Speed must be a positive value");

                speed = value;
            }
        }

        /// <inheritdoc />
        public bool LoopEnabled
        {
            get { return loopEnabled; }
            set
            {
                loopEnabled = value;

                if (hasFinished && loopEnabled)
                    hasFinished = false;
            }
        }

        /// <inheritdoc />
        public PlaybackMode PlaybackMode
        {
            get { return playbackMode; }
            set { playbackMode = value; }
        }

        /// <inheritdoc />
        public InterpolationMode TranslationInterpolation
        {
            get { return translationInterpolation; }
            set { translationInterpolation = value; }
        }

        /// <inheritdoc />
        public InterpolationMode OrientationInterpolation
        {
            get { return orientationInterpolation; }
            set { orientationInterpolation = value; }
        }

        /// <inheritdoc />
        public InterpolationMode ScaleInterpolation
        {
            get { return scaleInterpolation; }
            set { scaleInterpolation = value; }
        }

        /// <inheritdoc />
        public bool HasFinished
        {
            get { return hasFinished; }
        }

        /// <inheritdoc />
        public bool IsPlaying
        {
            get { return isPlaying; }
        }

        /// <inheritdoc />
        public Pose[] LocalBonePoses
        {
            get { return localBonePoses; }
        }

        /// <inheritdoc />
        public Matrix[] SkinnedBoneTransforms
        {
            get { return skinnedBoneTransforms; }
        }

        /// <inheritdoc />
        public float BlendWeight
        {
            get { return blendWeight; }
            set { blendWeight = value; }
        }

        #endregion

        /// <summary>Initializes a new instance of the 
        /// <see cref="T:XNAnimation.Controllers.AnimationController" />
        /// class.
        /// </summary>
        /// <param name="skeleton">The skeleton of the model to be animated</param>
        public AnimationController(SkinnedModelBoneCollection skeleton)
        {
            this.skeleton = skeleton;
            localBonePoses = new Pose[skeleton.Count];
            skinnedBoneTransforms = new Matrix[skeleton.Count];
            skeleton[0].CopyBindPoseTo(localBonePoses);
            boneControllers = new Quaternion[skeleton.Count];

            time = TimeSpan.Zero;
            speed = 1.0f;
            loopEnabled = true;
            playbackMode = PlaybackMode.Forward;

            blendWeight = 1.0f;

            translationInterpolation = InterpolationMode.None;
            orientationInterpolation = InterpolationMode.None;
            scaleInterpolation = InterpolationMode.None;

            crossFadeEnabled = false;
            crossFadeInterpolationAmount = 0.0f;
            crossFadeTime = TimeSpan.Zero;
            crossFadeElapsedTime = TimeSpan.Zero;

            hasFinished = false;
            isPlaying = false;
        }

        /// <inheritdoc />
        public void StartClip(AnimationClip animationClip)
        {
            this.animationClip = animationClip;
            hasFinished = false;
            isPlaying = true;

            time = TimeSpan.Zero;
            skeleton[0].CopyBindPoseTo(localBonePoses);
        }

        /// <inheritdoc />
        public void PlayClip(AnimationClip animationClip)
        {
            this.animationClip = animationClip;

            if (time < animationClip.Duration)
            {
                hasFinished = false;
                isPlaying = true;
            }
        }

        /// <inheritdoc />
        public void CrossFade(AnimationClip animationClip, TimeSpan fadeTime)
        {
            CrossFade(animationClip, fadeTime, InterpolationMode.Linear, InterpolationMode.Linear,
                InterpolationMode.Linear);
        }

        /// <inheritdoc />
        public void CrossFade(AnimationClip animationClip, TimeSpan fadeTime,
            InterpolationMode translationInterpolation, InterpolationMode orientationInterpolation,
            InterpolationMode scaleInterpolation)
        {
            if (crossFadeEnabled)   
            {
                crossFadeInterpolationAmount = 0;
                crossFadeTime = TimeSpan.Zero;
                crossFadeElapsedTime = TimeSpan.Zero;

                StartClip(crossFadeAnimationClip);
            }

            crossFadeAnimationClip = animationClip;
            crossFadeTime = fadeTime;
            crossFadeElapsedTime = TimeSpan.Zero;

            crossFadeTranslationInterpolation = translationInterpolation;
            crossFadeOrientationInterpolation = orientationInterpolation;
            crossFadeScaleInterpolation = scaleInterpolation;

            crossFadeEnabled = true;
        }

        public Matrix GetBoneAbsoluteTransform(int boneId)
        {
            return Matrix.Invert(skeleton[boneId].InverseBindPoseTransform) * SkinnedBoneTransforms[boneId];
        }

        public Matrix GetBoneAbsoluteTransform(string boneName)
        {
            int id = 0;
            for (int i = 0; i < skeleton.Count; i++)
            {
                if (skeleton[i].Name == boneName)
                {
                    id = i;
                    break;
                }
            }

            return Matrix.Invert(skeleton[id].InverseBindPoseTransform) * SkinnedBoneTransforms[id];
        }

        public void SetBoneController(int bone, Quaternion angles)
        {
            boneControllers[bone] = angles;
        }

        public void SetBoneController(string boneName, Quaternion angles)
        {
            boneControllers[skeleton.GetBoneId(boneName)] = angles;
        }


        /*
        public bool GetCurrentKeyframeTransform(string animationChannelName, out Matrix transform)
        {
            if (animationClip == null)
                throw new InvalidOperationException("animationClip");

            // Check if this animation has the desired channel
            AnimationChannel animationChannel;
            if (!animationClip.Channels.TryGetValue(animationChannelName, out animationChannel))
            {
                //keyframeTransform = Matrix.Identity;
                transform = Matrix.Identity;
                return false;
            }

            if (interpolationMode == InterpolationMode.None)
            {
                AnimationChannelKeyframe keyframe = animationChannel.GetKeyframeByTime(time);
                transform = keyframe.Transform;
            }
            else
            {
                int keyframeIndex = animationChannel.GetKeyframeIndexByTime(time);
                int nextKeyframeIndex = (keyframeIndex + 1) % animationChannel.Count;

                AnimationChannelKeyframe pose1 = animationChannel[keyframeIndex];
                AnimationChannelKeyframe pose2 = animationChannel[nextKeyframeIndex];

                long timeBetweenPoses;
                // Looping
                if (keyframeIndex == (animationChannel.Count - 1) && nextKeyframeIndex == 0)
                    if (loopEnabled)
                        timeBetweenPoses = animationClip.Duration.Ticks - pose1.Time.Ticks;
                    else
                        timeBetweenPoses = 0;
                else
                    timeBetweenPoses = pose2.Time.Ticks - pose1.Time.Ticks;

                if (timeBetweenPoses > 0)
                {
                    long elapsedPoseTime = time.Ticks - pose1.Time.Ticks;
                    float lerpFactor = elapsedPoseTime / (float) timeBetweenPoses;

                    if (interpolationMode == InterpolationMode.Linear)
                    {
                        //Matrix lerpMatrix;
                        //Matrix.Subtract(ref pose2.Transform, ref pose1.Transform, out lerpMatrix);
                        //Matrix.Multiply(ref lerpMatrix, lerpFactor, out lerpMatrix);
                        transform = pose1.Transform +
                            (pose2.Transform - pose1.Transform) * lerpFactor;
                        //Matrix.Lerp(ref pose1.Transform, ref pose2.Transform, lerpFactor);
                    }
                        //else if (interpolationMode == InterpolationMode.Spherical)
                    else
                    {
                        Vector3 temp;
                        Vector3 pose1Translation, pose2Translation;
                        Quaternion pose1Quaternion, pose2Quaternion;
                        pose1.Transform.Decompose(out temp, out pose1Quaternion, out pose1Translation);
                        pose2.Transform.Decompose(out temp, out pose2Quaternion, out pose2Translation);

                        Vector3 translation =
                            Vector3.SmoothStep(pose1Translation, pose2Translation, lerpFactor);
                        Quaternion rotation =
                            Quaternion.Slerp(pose1Quaternion, pose2Quaternion, lerpFactor);

                        Matrix.
                        Matrix slerpMatrix = Matrix.CreateFromQuaternion(rotation);
                        slerpMatrix.Translation = translation;
                        transform = slerpMatrix;
                    }
                }
                else
                    transform = pose1.Transform;
            }

            return true;
        }
        */

        /// <inheritdoc />
        public void Update(TimeSpan elapsedTime, Matrix parent)
        {
            if (hasFinished)
                return;

            // Scale the elapsed time
            TimeSpan scaledElapsedTime = TimeSpan.FromTicks((long) (elapsedTime.Ticks * speed));

            if (animationClip != null)
            {
                UpdateAnimationTime(scaledElapsedTime);
                if (crossFadeEnabled)
                    UpdateCrossFadeTime(scaledElapsedTime);

                UpdateChannelPoses();
            }

            UpdateAbsoluteBoneTransforms(ref parent);
        }

        /// <summary>
        /// Updates the CrossFade time
        /// </summary>
        /// <param name="elapsedTime">Time elapsed since the last update.</param>
        private void UpdateCrossFadeTime(TimeSpan elapsedTime)
        {
            crossFadeElapsedTime += elapsedTime;

            if (crossFadeElapsedTime > crossFadeTime)
            {
                crossFadeEnabled = false;
                crossFadeInterpolationAmount = 0;
                crossFadeTime = TimeSpan.Zero;
                crossFadeElapsedTime = TimeSpan.Zero;

                StartClip(crossFadeAnimationClip);
            }
            else
                crossFadeInterpolationAmount = crossFadeElapsedTime.Ticks / (float) crossFadeTime.Ticks;
        }

        /// <summary>
        /// Updates the animation clip time.
        /// </summary>
        /// <param name="elapsedTime">Time elapsed since the last update.</param>
        private void UpdateAnimationTime(TimeSpan elapsedTime)
        {
            // Ajust controller time
            if (playbackMode == PlaybackMode.Forward)
                time += elapsedTime;
            else
                time -= elapsedTime;

            // Animation finished
            if (time < TimeSpan.Zero || time > animationClip.Duration)
            {
                if (loopEnabled)
                {
                    if (time > animationClip.Duration)
                        while (time > animationClip.Duration)
                            time -= animationClip.Duration;
                    else
                        while (time < TimeSpan.Zero)
                            time += animationClip.Duration;

                    // Copy bind pose on animation restart
                    skeleton[0].CopyBindPoseTo(localBonePoses);
                }
                else
                {
                    if (time > animationClip.Duration)
                        time = animationClip.Duration;
                    else
                        time = TimeSpan.Zero;

                    isPlaying = false;
                    hasFinished = true;
                }
            }
        }

        /// <summary>
        /// Updates the pose of all skeleton's bones.
        /// </summary>
        private void UpdateChannelPoses()
        {
            AnimationChannel animationChannel;
            Pose channelPose;

            for (int i = 0; i < localBonePoses.Length; i++)
            {
                // Search for the current channel in the current animation clip
                string animationChannelName = skeleton[i].Name;
                if (animationClip.Channels.TryGetValue(animationChannelName, out animationChannel))
                {
                    InterpolateChannelPose(animationChannel, time, out channelPose);
                    localBonePoses[i] = channelPose;
                }

                // If CrossFade is enabled blend this channel in two animation clips
                if (crossFadeEnabled)
                {
                    // Search for the current channel in the cross fade clip
                    if (
                        crossFadeAnimationClip.Channels.TryGetValue(animationChannelName,
                            out animationChannel))
                    {
                        InterpolateChannelPose(animationChannel, TimeSpan.Zero, out channelPose);
                    }
                    else
                        channelPose = skeleton[i].BindPose;

                    // Interpolate each channel with the cross fade animation
                    localBonePoses[i] =
                        Pose.Interpolate(localBonePoses[i], channelPose, crossFadeInterpolationAmount,
                            crossFadeTranslationInterpolation, crossFadeOrientationInterpolation,
                            crossFadeScaleInterpolation);
                }
            }
        }

        /// <summary>
        /// Retrieves and interpolates the pose of an animation channel.
        /// </summary>
        /// <param name="animationChannel">Name of the animation channel.</param>
        /// <param name="animationTime">Current animation clip time.</param>
        /// <param name="outPose">The output interpolated pose.</param>
        private void InterpolateChannelPose(AnimationChannel animationChannel, TimeSpan animationTime,
            out Pose outPose)
        {
            if (translationInterpolation == InterpolationMode.None &&
                orientationInterpolation == InterpolationMode.None &&
                    scaleInterpolation == InterpolationMode.None)
            {
                int keyframeIndex = animationChannel.GetKeyframeIndexByTime(animationTime);
                outPose = animationChannel[keyframeIndex].Pose;
            }
            else
            {
                int keyframeIndex = animationChannel.GetKeyframeIndexByTime(animationTime);
                int nextKeyframeIndex = (keyframeIndex + 1) % animationChannel.Count;

                AnimationChannelKeyframe keyframe1 = animationChannel[keyframeIndex];
                AnimationChannelKeyframe keyframe2 = animationChannel[nextKeyframeIndex];

                // Calculate the time between the keyframes considering loop
                long keyframeDuration;
                if (keyframeIndex == (animationChannel.Count - 1))
                    keyframeDuration = animationClip.Duration.Ticks - keyframe1.Time.Ticks;
                else
                    keyframeDuration = keyframe2.Time.Ticks - keyframe1.Time.Ticks;

                // Interpolate when duration higher than zero
                if (keyframeDuration > 0)
                {
                    long elapsedKeyframeTime = animationTime.Ticks - keyframe1.Time.Ticks;
                    float lerpFactor = elapsedKeyframeTime / (float) keyframeDuration;

                    outPose =
                        Pose.Interpolate(keyframe1.Pose, keyframe2.Pose, lerpFactor,
                            translationInterpolation, orientationInterpolation, scaleInterpolation);
                }
                // Otherwise don't interpolate
                else
                    outPose = keyframe1.Pose;
            }
        }

        /// <summary>
        /// Calculates the final configuration of all skeleton's bones used to transform
        /// the model's mesh.
        /// </summary>
        /// <param name="parent"></param>
        private void UpdateAbsoluteBoneTransforms(ref Matrix parent)
        {
            Matrix poseTransform;

            // Calculate the pose matrix
            poseTransform = Matrix.CreateFromQuaternion(localBonePoses[0].Orientation);

            poseTransform.Translation = localBonePoses[0].Translation;

            /*
            if (localBonePoses[0].Orientation == Quaternion.Identity)
            {
                System.Console.WriteLine("Pose 0");
            }
            */

            // Scale vectors
            poseTransform.M11 *= localBonePoses[0].Scale.X;
            poseTransform.M21 *= localBonePoses[0].Scale.X;
            poseTransform.M31 *= localBonePoses[0].Scale.X;
            poseTransform.M12 *= localBonePoses[0].Scale.Y;
            poseTransform.M22 *= localBonePoses[0].Scale.Y;
            poseTransform.M32 *= localBonePoses[0].Scale.Y;
            poseTransform.M13 *= localBonePoses[0].Scale.Z;
            poseTransform.M23 *= localBonePoses[0].Scale.Z;
            poseTransform.M33 *= localBonePoses[0].Scale.Z;

            // TODO Use and test the scale
            //poseTransform.Scale = localBonePoses[0].Scale;

            // Calculate the absolute bone transform
            skinnedBoneTransforms[0] = poseTransform * parent;
            for (int i = 1; i < skinnedBoneTransforms.Length; i++)
            {
                // Calculate the pose matrix
                poseTransform = Matrix.CreateFromQuaternion(localBonePoses[i].Orientation);
                // Add custom bone rotation
                poseTransform *= Matrix.CreateFromQuaternion(boneControllers[i]);

                poseTransform.Translation = localBonePoses[i].Translation;

                /*
                if (localBonePoses[i].Orientation == Quaternion.Identity)
                {
                    System.Console.WriteLine("Pose " + i);
                }
                */

                // Scale vectors
                poseTransform.M11 *= localBonePoses[i].Scale.X;
                poseTransform.M21 *= localBonePoses[i].Scale.X;
                poseTransform.M31 *= localBonePoses[i].Scale.X;
                poseTransform.M12 *= localBonePoses[i].Scale.Y;
                poseTransform.M22 *= localBonePoses[i].Scale.Y;
                poseTransform.M32 *= localBonePoses[i].Scale.Y;
                poseTransform.M13 *= localBonePoses[i].Scale.Z;
                poseTransform.M23 *= localBonePoses[i].Scale.Z;
                poseTransform.M33 *= localBonePoses[i].Scale.Z;

                int parentIndex = skeleton[i].Parent.Index;
                skinnedBoneTransforms[i] = poseTransform * skinnedBoneTransforms[parentIndex];
            }

            // Calculate final bone transform
            for (int i = 0; i < skinnedBoneTransforms.Length; i++)
            {
                skinnedBoneTransforms[i] = skeleton[i].InverseBindPoseTransform *
                    skinnedBoneTransforms[i];
            }
        }
    }
}