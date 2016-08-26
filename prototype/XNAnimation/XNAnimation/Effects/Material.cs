/*
 * IMaterial.cs
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
    public class Material : IMaterial
    {
        private EffectParameter emissiveColorParam;
        private EffectParameter diffuseColorParam;
        private EffectParameter specularColorParam;
        private EffectParameter specularPowerParam;

        #region Properties

        public Vector3 EmissiveColor
        {
            get { return emissiveColorParam.GetValueVector3(); }
            set { emissiveColorParam.SetValue(value); }
        }

        public Vector3 DiffuseColor
        {
            get { return diffuseColorParam.GetValueVector3(); }
            set { diffuseColorParam.SetValue(value); }
        }

        public Vector3 SpecularColor
        {
            get { return specularColorParam.GetValueVector3(); }
            set { specularColorParam.SetValue(value); }
        }

        public float SpecularPower
        {
            get { return specularPowerParam.GetValueSingle(); }
            set { specularPowerParam.SetValue(value); }
        }

        #endregion

        internal Material(EffectParameter materialStructParameter)
        {
            CacheEffectParams(materialStructParameter);
        }

        private void CacheEffectParams(EffectParameter materialStructParameter)
        {
            emissiveColorParam = materialStructParameter.StructureMembers["emissiveColor"];
            diffuseColorParam = materialStructParameter.StructureMembers["diffuseColor"];
            specularColorParam = materialStructParameter.StructureMembers["specularColor"];
            specularPowerParam = materialStructParameter.StructureMembers["specularPower"];
        }
    }
}