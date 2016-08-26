/*
 * SkinnedModelMaterialProcessor.cs
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
using System.ComponentModel;
using System.IO;
using Microsoft.Xna.Framework;
using Microsoft.Xna.Framework.Content.Pipeline;
using Microsoft.Xna.Framework.Content.Pipeline.Graphics;
using Microsoft.Xna.Framework.Content.Pipeline.Processors;
using XNAnimationPipeline.Effects;

namespace XNAnimationPipeline.Pipeline
{
    [ContentProcessor(DisplayName = "Model Material - XNAnimation"), DesignTimeVisible(false)]
    internal class SkinnedModelMaterialProcessor : MaterialProcessor
    {
        public static readonly string DiffuseMapKey = "Texture";
        public static readonly string NormalMapKey = "Bump0";
        public static readonly string SpecularMapKey = "Specular0";

        public override MaterialContent Process(MaterialContent input, ContentProcessorContext context)
        {
            MaterialContent processedMaterial;

            if (context.TargetProfile == Microsoft.Xna.Framework.Graphics.GraphicsProfile.HiDef)
            {
                processedMaterial = new SkinnedModelMaterialContent();
            }
            else
            {
                processedMaterial = new SkinnedMaterialContent();
            }

            ProcessTextures(input, processedMaterial, context);
            
            // Processes surface material
            ProcessMaterial(input, processedMaterial, context);

            return processedMaterial;
        }

        private void ProcessTextures(MaterialContent input, MaterialContent output,
            ContentProcessorContext context)
        {
            foreach (KeyValuePair<string, ExternalReference<TextureContent>> keyValue in input.Textures)
            {
                ProcessTexture(keyValue.Key, keyValue.Value, output, context);
            }
        }

        protected virtual void ProcessTexture(string key, ExternalReference<TextureContent> texture,
            MaterialContent output, ContentProcessorContext context)
        {
            if (output is SkinnedModelMaterialContent)
            {
                SkinnedModelMaterialContent skinnedModelMaterial = output as SkinnedModelMaterialContent;
                if (key.Equals(DiffuseMapKey))
                {
                    skinnedModelMaterial.DiffuseMapEnabled = true;
                    skinnedModelMaterial.DiffuseMapContent = base.BuildTexture(key, texture, context);
                }
                else if (key.Equals(NormalMapKey))
                {
                    skinnedModelMaterial.NormalMapEnabled = true;
                    skinnedModelMaterial.NormalMapContent = base.BuildTexture(key, texture, context);
                }
                else if (key.Equals(SpecularMapKey))
                {
                    skinnedModelMaterial.SpecularMapEnabled = true;
                    skinnedModelMaterial.SpecularMapContent = base.BuildTexture(key, texture, context);
                }
            }
            else if (output is SkinnedMaterialContent)
            {
                SkinnedMaterialContent skinnedModelMaterial = output as SkinnedMaterialContent;
                if (key.Equals(DiffuseMapKey))
                {
                    skinnedModelMaterial.Texture = base.BuildTexture(key, texture, context);
                    context.Logger.LogWarning(null, null, "built {0}", skinnedModelMaterial.Texture.Filename);

                }
            }
        }

        protected virtual void ProcessMaterial(MaterialContent input, 
            MaterialContent output, ContentProcessorContext context)
        {
            BasicMaterialContent basicMaterial = input as BasicMaterialContent;

            if (basicMaterial != null)
            {
                if (output is SkinnedModelMaterialContent)
                {
                    SkinnedModelMaterialContent skinnedModelMaterial = output as SkinnedModelMaterialContent;

                    skinnedModelMaterial.EmissiveColor = basicMaterial.EmissiveColor.GetValueOrDefault(
                        Vector3.Zero);
                    skinnedModelMaterial.DiffuseColor = basicMaterial.DiffuseColor.GetValueOrDefault(
                        Vector3.One);
                    skinnedModelMaterial.SpecularColor = basicMaterial.SpecularColor.GetValueOrDefault(
                        Vector3.One);
                    skinnedModelMaterial.SpecularPower = basicMaterial.SpecularPower.GetValueOrDefault(
                        16);

                    EffectContent effectContent = new EffectContent();
                    effectContent.EffectCode = Resources.SkinnedModelEffect;

                    EffectProcessor effectProcessor = new EffectProcessor();
                    skinnedModelMaterial.SkinnedEffectContent = effectProcessor.Process(effectContent, context);
                }
                else if (output is SkinnedMaterialContent)
                {
                    SkinnedMaterialContent skinnedModelMaterial = output as SkinnedMaterialContent;

                    skinnedModelMaterial.EmissiveColor = basicMaterial.EmissiveColor.GetValueOrDefault(
                        Vector3.Zero);
                    skinnedModelMaterial.DiffuseColor = basicMaterial.DiffuseColor.GetValueOrDefault(
                        Vector3.One);
                    skinnedModelMaterial.SpecularColor = basicMaterial.SpecularColor.GetValueOrDefault(
                        Vector3.One);
                    skinnedModelMaterial.SpecularPower = basicMaterial.SpecularPower.GetValueOrDefault(
                        16);
                }
            }
        }
    }
}