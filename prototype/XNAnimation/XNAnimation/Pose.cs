/*
 * Pose.cs
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
using XNAnimation.Controllers;
using System.Globalization;

namespace XNAnimation
{
    public struct Pose : IEquatable<Pose>
    {
        public Vector3 Translation;
        public Quaternion Orientation;
        public Vector3 Scale;

        private static readonly Pose _identity;

        #region Properties

        public static Pose Identity
        {
            get { return _identity; }
        }

        #endregion

        static Pose()
        {
            _identity.Orientation = Quaternion.Identity;
            _identity.Translation = Vector3.Zero;
            _identity.Scale = Vector3.Zero;
        }

        /*
        public Pose(Vector3 translation, Quaternion orientation, Vector3 scale)
        {
            this.Translation = translation;
            this.Orientation = orientation;
            this.Scale = scale;
        }
        */

        public static Pose Interpolate(Pose pose1, Pose pose2, float amount,
            InterpolationMode translationInterpolation, InterpolationMode orientationInterpolation,
            InterpolationMode scaleInterpolation)
        {
            Pose resultPose;

            if (amount < 0 || amount > 1)
                throw new ArgumentException("Amount must be between 0.0 and 1.0 inclusive.");

            switch (translationInterpolation)
            {
                case InterpolationMode.None:
                    resultPose.Translation = pose1.Translation;
                    break;

                case InterpolationMode.Linear:
                    Vector3.Lerp(ref pose1.Translation, ref pose2.Translation, amount, 
                        out resultPose.Translation);
                    break;

                case InterpolationMode.Cubic:
                    Vector3.SmoothStep(ref pose1.Translation, ref pose2.Translation, amount,
                        out resultPose.Translation);
                    break;

                default:
                    throw new ArgumentException("Translation interpolation method not supported");
            }

            switch (orientationInterpolation)
            {
                case InterpolationMode.None:
                    resultPose.Orientation = pose1.Orientation;
                    break;

                case InterpolationMode.Linear:
                        Quaternion.Lerp(ref pose1.Orientation, ref pose2.Orientation, amount,
                            out resultPose.Orientation);
                    break;

                case InterpolationMode.Spherical:
                        Quaternion.Slerp(ref pose1.Orientation, ref pose2.Orientation, amount,
                            out resultPose.Orientation);
                    break;

                default:
                    throw new ArgumentException("Orientation interpolation method not supported");
            }

            switch (scaleInterpolation)
            {
                case InterpolationMode.None:
                    resultPose.Scale = pose1.Scale;
                    break;

                case InterpolationMode.Linear:
                    Vector3.Lerp(ref pose1.Scale, ref pose2.Scale, amount,
                        out resultPose.Scale);
                    break;

                case InterpolationMode.Cubic:
                    Vector3.SmoothStep(ref pose1.Scale, ref pose2.Scale, amount,
                        out resultPose.Scale);
                    break;

                default:
                    throw new ArgumentException("Scale interpolation method not supported");
            }

            return resultPose;
        }

        #region IEquatable<Pose> Members

        public override int GetHashCode()
        {
            return (Translation.GetHashCode() + Orientation.GetHashCode() + Scale.GetHashCode());
        }
        
        public override string ToString()
        {
            CultureInfo currentCulture = CultureInfo.CurrentCulture;
            return string.Format(currentCulture, 
                "{{Translation:{0}\n Orientation:{1}\n Scale:{2}\n}}", new object[]
                { Translation.ToString(), Orientation.ToString(), Scale.ToString() });
        }

        public bool Equals(Pose other)
        {
            return (Translation == other.Translation && 
                Orientation == other.Orientation && 
                Scale == other.Scale);
        }

        public override bool Equals(object obj)
        {
            bool result = false;
            if (obj is Pose)
            {
                result = this.Equals((Pose)obj);
            }

            return result;
        }

        public static bool operator == (Pose pose1, Pose pose2)
        {
            return (pose1.Translation == pose2.Translation &&
                pose1.Orientation == pose2.Orientation &&
                pose1.Scale == pose2.Scale);
        }

        public static bool operator !=(Pose pose1, Pose pose2)
        {
            return (pose1.Translation != pose2.Translation ||
                pose1.Orientation != pose2.Orientation ||
                pose1.Scale != pose2.Scale);
        }

        #endregion
    }
}