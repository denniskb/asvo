/*
 * PointLight.cs
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
using Microsoft.Xna.Framework.Graphics;

namespace XNAnimation.Effects
{
    public class PointLight : IPointLight
    {
        private EffectParameter positionParam;
        private EffectParameter colorParam;

        #region Properties

        public Vector3 Position
        {
            get { return positionParam.GetValueVector3(); }
            set { positionParam.SetValue(value); }
        }

        public Vector3 Color
        {
            get { return colorParam.GetValueVector3(); }
            set { colorParam.SetValue(value); }
        }

        #endregion

        internal PointLight(EffectParameter lightStuctParameter)
        {
            CacheEffectParams(lightStuctParameter);
        }

        private void CacheEffectParams(EffectParameter lightStuctParameter)
        {
            positionParam = lightStuctParameter.StructureMembers["position"];
            colorParam = lightStuctParameter.StructureMembers["color"];
        }
    }
}