/*
 * SkinnedModelBasicEffect.cs
 * Authors:  Bruno Evangelista
 *           Rodrigo 'r2d2rigo' Díaz
 * Copyright (c) 2008 Bruno Evangelista. All rights reserved.
 * Copyright (c) 2010 Rodrigo 'r2d2rigo' Díaz. All rights reserved.
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
using Microsoft.Xna.Framework;
using Microsoft.Xna.Framework.Content;
using Microsoft.Xna.Framework.Graphics;

namespace XNAnimation.Effects
{
    /// <summary>
    /// Specify how many light sorces will be used
    /// </summary>
    public enum EnabledLights
    {
        One = 0,
        Two,
        Four,
        Six,
        Eight
    } ;

    /// <summary>
    /// Represents an effect that supports skinning, normal, specular and diffuse textures, 
    /// and phong lighting with multiple point light sources. This effect can use shader model 2.0
    /// or 3.0, according to the number of lights enabled.
    /// </summary>
    public class SkinnedModelBasicEffect : Effect
    {
        public static int MaxSupportedBones = 80;
        public static int MaxSupportedLights = 8;

        // Cached parameters
        private Matrix viewMatrix;
        private Matrix projectionMatrix;

        // Cached configurations
        private bool lightEnabled;
        private bool normalMapEnabled;

        // Effect parameters - Matrices
        private EffectParameter bonesParam;
        private EffectParameter worldParam;
        private EffectParameter viewInverseParam;
        private EffectParameter viewProjectionParam;

        // Configurations
        private EffectParameter diffuseMapEnabledParam;
        private EffectParameter specularMapEnabledParam;

        // Material
        private Material material;
        private EffectParameter diffuseMapParam;
        private EffectParameter normalMapParam;
        private EffectParameter specularMapParam;

        // Lights
        private EffectParameter ambientLightColorParam;
        private EnabledLights enabledLights;
        private PointLightCollection pointLightCollection;

        //private EffectTechnique[] lightTechniques;
        //private EffectTechnique[] normalLightTechniques;

        #region Properties

        public Matrix World
        {
            get { return worldParam.GetValueMatrix(); }
            set { worldParam.SetValue(value); }
        }

        /// <summary>
        /// Gets or sets the bone matrices of the skeleton.
        /// </summary>
        public Matrix[] Bones
        {
            get { return bonesParam.GetValueMatrixArray(MaxSupportedBones); }
            set { bonesParam.SetValue(value); }
        }

        /// <summary>
        /// Gets or sets the camera view matrix.
        /// </summary>
        public Matrix View
        {
            get { return viewMatrix; }
            set
            {
                viewMatrix = value;
                viewInverseParam.SetValue(Matrix.Invert(value));

                viewProjectionParam.SetValue(viewMatrix * projectionMatrix);
            }
        }

        /// <summary>
        /// Gets or sets the camera projection matrix.
        /// </summary>
        public Matrix Projection
        {
            get { return projectionMatrix; }
            set
            {
                projectionMatrix = value;
                viewProjectionParam.SetValue(viewMatrix * projectionMatrix);
            }
        }

        /// <summary>
        /// Enables diffuse texture.
        /// </summary>
        public bool DiffuseMapEnabled
        {
            get { return diffuseMapEnabledParam.GetValueBoolean(); }
            set { diffuseMapEnabledParam.SetValue(value); }
        }

        /// <summary>
        /// Gets or sets diffuse texture.
        /// </summary>
        public Texture2D DiffuseMap
        {
            get { return diffuseMapParam.GetValueTexture2D(); }
            set { diffuseMapParam.SetValue(value); }
        }

        /// <summary>
        /// Enables normal map texture.
        /// </summary>
        public bool SpecularMapEnabled
        {
            get { return specularMapEnabledParam.GetValueBoolean(); }
            set { specularMapEnabledParam.SetValue(value); }
        }

        /// <summary>
        /// Gets or sets normal map texture.
        /// </summary>
        public Texture2D SpecularMap
        {
            get { return specularMapParam.GetValueTexture2D(); }
            set { specularMapParam.SetValue(value); }
        }

        /// <summary>
        /// Enables normal map texture.
        /// </summary>
        public bool NormalMapEnabled
        {
            get { return normalMapEnabled; }
            set
            {
                normalMapEnabled = value;
                UpdateTechnique();
            }
        }

        /// <summary>
        /// Gets or sets normal map texture.
        /// </summary>
        public Texture2D NormalMap
        {
            get { return normalMapParam.GetValueTexture2D(); }
            set { normalMapParam.SetValue(value); }
        }


        /// <summary>
        /// Enables lighting.
        /// </summary>
        public bool LightEnabled
        {
            get { return lightEnabled; }
            set
            {
                lightEnabled = value;
                UpdateTechnique();
            }
        }

        /// <summary>
        /// Gets or sets the number of enabled light sources.
        /// </summary>
        public EnabledLights EnabledLights
        {
            get { return enabledLights; }
            set
            {
                enabledLights = value;
                UpdateTechnique();
            }
        }

        /// <summary>
        /// Gets the material used for lighting.
        /// </summary>
        public Material Material
        {
            get { return material; }
        }

        /// <summary>
        /// Gets or sets the ambient light color
        /// </summary>
        public Vector3 AmbientLightColor
        {
            get { return ambientLightColorParam.GetValueVector3(); }
            set { ambientLightColorParam.SetValue(value); }
        }

        /// <summary>
        /// Gets the point light sources. 
        /// </summary>
        public PointLightCollection PointLights
        {
            get { return pointLightCollection; }
        }

        #endregion

        /// <summary>Initializes a new instance of the 
        /// <see cref="T:XNAnimation.Effects.SkinnedModelBasicEffect" />
        /// class.
        /// </summary>
        /// <param name="graphicsDevice">The graphics device that will create the effect.</param>
        /// <param name="clone">The effect to clone.</param>
        protected SkinnedModelBasicEffect(SkinnedModelBasicEffect clone)
            : base(clone)
        {
            CacheEffectParams();
        }

        /// <summary>Initializes a new instance of the 
        /// <see cref="T:XNAnimation.Effects.SkinnedModelBasicEffect" />
        /// class.
        /// </summary>
        /// <param name="graphicsDevice">The graphics device that will create the effect.</param>
        /// <param name="effectPool">Specifies a pool of resources to share between effects.</param>
        public SkinnedModelBasicEffect(Effect cloneSource)
            : base(cloneSource)
        {
            CacheEffectParams();
            InitializeEffectParams();
        }

        private void CacheEffectParams()
        {
            // Matrices
            worldParam = Parameters["matW"];
            viewInverseParam = Parameters["matVI"];
            viewProjectionParam = Parameters["matVP"];
            bonesParam = Parameters["matBones"];

            // Configurations
            diffuseMapEnabledParam = Parameters["diffuseMapEnabled"];
            specularMapEnabledParam = Parameters["specularMapEnabled"];

            // Material
            EffectParameter materialStructParam = Parameters["material"];
            material = new Material(materialStructParam);
            diffuseMapParam = Parameters["diffuseMap0"];
            normalMapParam = Parameters["normalMap0"];
            specularMapParam = Parameters["specularMap0"];

            // Ambient light
            ambientLightColorParam = Parameters["ambientLightColor"];

            // Lights
            EffectParameter pointLightStructParam = Parameters["lights"];
            List<PointLight> pointLights = new List<PointLight>(MaxSupportedLights);
            for (int i = 0; i < MaxSupportedLights; i++)
                pointLights.Add(new PointLight(pointLightStructParam.Elements[i]));
            pointLightCollection = new PointLightCollection(pointLights);
        }

        private void InitializeEffectParams()
        {
            World = Matrix.Identity;
            View = Matrix.Identity;
            Projection = Matrix.Identity;
            for (int i = 0; i < MaxSupportedBones; i++)
                bonesParam.Elements[i].SetValue(Matrix.Identity);

            LightEnabled = false;
            EnabledLights = EnabledLights.One;
            DiffuseMapEnabled = false;
            NormalMapEnabled = false;
            SpecularMapEnabled = false;

            Material.EmissiveColor = Vector3.Zero;
            Material.DiffuseColor = Vector3.One;
            Material.SpecularColor = Vector3.One;
            Material.SpecularPower = 16;

            for (int i = 0; i < MaxSupportedLights; i++)
            {
                PointLights[i].Position = Vector3.Zero;
                PointLights[i].Color = Vector3.Zero;
            }
        }

        private void UpdateTechnique()
        {
            // TODO Optimize this pre-caching the techniques
            if (lightEnabled)
            {
                switch (enabledLights)
                {
                    case EnabledLights.One:
                        CurrentTechnique = (normalMapEnabled)
                            ? Techniques["AnimatedModel_OneLightWithNormal"]
                            : Techniques["AnimatedModel_OneLight"];
                        break;

                    case EnabledLights.Two:
                        CurrentTechnique = (normalMapEnabled)
                            ? Techniques["AnimatedModel_TwoLightWithNormal"]
                            : Techniques["AnimatedModel_TwoLight"];
                        break;

                    case EnabledLights.Four:
                        CurrentTechnique = (normalMapEnabled)
                            ? Techniques["AnimatedModel_FourLightWithNormal"]
                            : Techniques["AnimatedModel_FourLight"];
                        break;

                    case EnabledLights.Six:
                        CurrentTechnique = (normalMapEnabled)
                            ? Techniques["AnimatedModel_SixLightWithNormal"]
                            : Techniques["AnimatedModel_SixLight"];
                        break;

                    case EnabledLights.Eight:
                        CurrentTechnique = (normalMapEnabled)
                            ? Techniques["AnimatedModel_EightLightWithNormal"]
                            : Techniques["AnimatedModel_EightLight"];
                        break;
                }
            }
            else
            {
                CurrentTechnique = Techniques["AnimatedModel_NoLight"];
            }
        }

        internal static SkinnedModelBasicEffect Read(ContentReader input)
        {
            Effect effect = input.ReadObject<Effect>();

            SkinnedModelBasicEffect basicEffect = new SkinnedModelBasicEffect(effect);

            basicEffect.material.EmissiveColor = input.ReadVector3();
            basicEffect.material.DiffuseColor = input.ReadVector3();
            basicEffect.material.SpecularColor = input.ReadVector3();
            basicEffect.material.SpecularPower = input.ReadSingle();

            basicEffect.DiffuseMapEnabled = input.ReadBoolean();
            basicEffect.NormalMapEnabled = input.ReadBoolean();
            basicEffect.SpecularMapEnabled = input.ReadBoolean();

            basicEffect.DiffuseMap = input.ReadExternalReference<Texture2D>();
            basicEffect.NormalMap = input.ReadExternalReference<Texture2D>();
            basicEffect.SpecularMap = input.ReadExternalReference<Texture2D>();

            basicEffect.lightEnabled = false;
            basicEffect.enabledLights = EnabledLights.One;

            return basicEffect;
        }
    }
}