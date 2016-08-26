/*
 * IBlendController.cs
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

namespace XNAnimation.Controllers
{
    /// <summary>
    /// Specify how animation clips are blended.
    /// </summary>
    public enum BlendMode
    {
        /// <summary>
        /// Blends animation clips interpolation between them.
        /// </summary>
        Interpolate,

        /// <summary>
        /// Blends animation clips additively.
        /// </summary>
        Additive
    }

    /// <summary>
    /// Enumerates the available blend layers.
    /// </summary>
    public enum BlendLayer
    {
        /// <summary>
        /// First blend layer.
        /// </summary>
        One,

        /// <summary>
        /// Second blend layer.
        /// </summary>
        Two,

        /// <summary>
        /// Third blend layer.
        /// </summary>
        Three,

        /// <summary>
        /// Fourth blend layer.
        /// </summary>
        Four
    }

    /// <summary>
    /// Defines an interface for an animation blend controller.
    /// </summary>
    public interface IBlendController : IBlendable
    {
        /// <summary>
        /// Gets or sets the animation blend mode.
        /// </summary>
        BlendMode BlendMode { get; set; }

        InterpolationMode TranslationInterpolation { get; set; }
        InterpolationMode OrientationInterpolation { get; set; }
        InterpolationMode ScaleInterpolation { get; set; }

        void SetBlendLayer(IBlendable blendable, BlendLayer blendLayer);

        // FadeLayer
        void SetWeightInterpolation(float value, float desiredValue, TimeSpan time, BlendLayer blendLayer);

        void UpdateBlend();
    }
}