/*
 * SkinnedModel.cs
 * Authors: Bruno Evangelista
 *          Rodrigo 'r2d2rigo' Díaz
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
using XNAnimation.Effects;

namespace XNAnimation
{
    public class point_light
    {
        public Vector3 Position;
        public Vector3 Color;

        public point_light()
        {

        }
    }

    public class SkinnedModel : IEffectMatrices
    {
        private SkinnedModelMeshCollection meshes;
        private SkinnedModelBoneCollection skeleton;
        private AnimationClipDictionary animationClips;

        #region Properties

        public SkinnedModelMeshCollection Meshes
        {
            get { return meshes; }
        }

        public SkinnedModelBoneCollection SkeletonBones
        {
            get { return skeleton; }
        }

        public AnimationClipDictionary AnimationClips
        {
            get { return animationClips; }
        }

        public Matrix World { get; set; }
        public Matrix View { get; set; }
        public Matrix Projection { get; set; }
        public Matrix[] BoneTransforms { get; set; }

        public Vector3 EmissiveColor { get; set; }
        public Vector3 DiffuseColor { get; set; }
        public Vector3 SpecularColor { get; set; }
        public float SpecularPower { get; set; }
        public Vector3 AmbientLightColor { get; set; }
        public bool LightEnabled { get; set; }
        public EnabledLights EnabledLights { get; set; }
        public List<point_light> PointLights;

        #endregion

        internal SkinnedModel()
        {
        }

        internal static SkinnedModel Read(ContentReader input)
        {
            SkinnedModel skinnedModel = new SkinnedModel();

            skinnedModel.ReadMeshes(input);
            skinnedModel.ReadBones(input);
            skinnedModel.ReadAnimations(input);

            skinnedModel.BoneTransforms = new Matrix[skinnedModel.SkeletonBones.Count];

            skinnedModel.EmissiveColor = Vector3.Zero;
            skinnedModel.DiffuseColor = Vector3.Zero;
            skinnedModel.SpecularColor = Vector3.Zero;
            skinnedModel.AmbientLightColor = Vector3.Zero;
            skinnedModel.SpecularPower = 1;

            skinnedModel.PointLights = new List<point_light>(SkinnedModelBasicEffect.MaxSupportedLights);
            for (int i = 0; i < SkinnedModelBasicEffect.MaxSupportedLights; i++)
                skinnedModel.PointLights.Add(new point_light());

            return skinnedModel;
        }

        private void ReadMeshes(ContentReader input)
        {
            int numMeshes = input.ReadInt32();
            List<SkinnedModelMesh> meshList = new List<SkinnedModelMesh>(numMeshes);
            for (int i = 0; i < numMeshes; i++)
            {
                meshList.Add(input.ReadObject<SkinnedModelMesh>());
            }

            meshes = new SkinnedModelMeshCollection(meshList);
        }

        private void ReadBones(ContentReader input)
        {
            int numSkeletonBones = input.ReadInt32();
            List<SkinnedModelBone> skinnedModelBoneList = new List<SkinnedModelBone>(numSkeletonBones);

            // Read all bones
            for (int i = 0; i < numSkeletonBones; i++)
            {
                input.ReadSharedResource<SkinnedModelBone>(
                    delegate(SkinnedModelBone skinnedBone) { skinnedModelBoneList.Add(skinnedBone); });
            }

            // Create the skeleton
            skeleton = new SkinnedModelBoneCollection(skinnedModelBoneList);
        }

        private void ReadAnimations(ContentReader input)
        {
            int numAnimationClips = input.ReadInt32();
            Dictionary<string, AnimationClip> animationClipDictionary =
                new Dictionary<string, AnimationClip>();

            // Read all animation clips
            for (int i = 0; i < numAnimationClips; i++)
            {
                input.ReadSharedResource<AnimationClip>(
                    delegate(AnimationClip animationClip) { animationClipDictionary.Add(animationClip.Name, animationClip); });
            }

            animationClips = new AnimationClipDictionary(animationClipDictionary);
        }

        public void Draw()
        {
            for (int i = 0; i < meshes.Count; i++)
            {
                meshes[i].effect.World = World;
                meshes[i].effect.View = View;
                meshes[i].effect.Projection = Projection;
                meshes[i].effect.Bones = BoneTransforms;

                meshes[i].effect.Material.DiffuseColor = DiffuseColor;
                meshes[i].effect.Material.EmissiveColor = EmissiveColor;
                meshes[i].effect.Material.SpecularColor = SpecularColor;
                meshes[i].effect.Material.SpecularPower = SpecularPower;

                meshes[i].effect.AmbientLightColor = AmbientLightColor;
                meshes[i].effect.LightEnabled = LightEnabled;
                meshes[i].effect.EnabledLights = EnabledLights;

                for (int j = 0; j < PointLights.Count; j++)
                {
                    meshes[i].effect.PointLights[j].Color = PointLights[j].Color;
                    meshes[i].effect.PointLights[j].Position = PointLights[j].Position;
                }

                meshes[i].Draw();
            }
        }
    }
}