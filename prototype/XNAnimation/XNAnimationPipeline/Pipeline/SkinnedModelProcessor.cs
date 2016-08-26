/*
 * SkinnedModelProcessor.cs
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
using System.Runtime.InteropServices;
using System.Text;
using System.Xml;
using Microsoft.Xna.Framework;
using Microsoft.Xna.Framework.Graphics;
using Microsoft.Xna.Framework.Content;
using Microsoft.Xna.Framework.Content.Pipeline;
using Microsoft.Xna.Framework.Content.Pipeline.Graphics;
using Microsoft.Xna.Framework.Content.Pipeline.Processors;
using Microsoft.Xna.Framework.Design;
using XNAnimation;
using XNAnimation.Effects;
using AnimationChannel = Microsoft.Xna.Framework.Content.Pipeline.Graphics.AnimationChannel;

namespace XNAnimationPipeline.Pipeline
{
    /// <summary>
    /// This class will be instantiated by the XNA Framework Content Pipeline
    /// to apply custom processing to content data, converting an object of
    /// type TInput to TOutput. The input and output types may be the same if
    /// the processor wishes to alter data without changing its type.
    ///
    /// This should be part of a Content Pipeline Extension Library project.
    ///
    /// </summary>
    [ContentProcessor(DisplayName = "Model - XNAnimation")]
    public class SkinnedModelProcessor : ContentProcessor<NodeContent, SkinnedModelContent>
    {
        private struct SplitTask
        {
            public string Name;
            public float StartTimeSeconds;
            public float EndTimeSeconts;

            public SplitTask(string name, float startTime, float endTime)
            {
                Name = name;
                StartTimeSeconds = startTime;
                EndTimeSeconts = endTime;
            }

            public override string ToString()
            {
                return string.Format("{0} [{1} : {2}]", Name, StartTimeSeconds, EndTimeSeconts);
            }
        }

        public static readonly int DefaultAnimationFramerate = 60;

        #region Content Properties

        private bool bakeMeshTransform = true;
        [DisplayName("Bake Mesh Transforms"),
        Description("If enabled, bake the mesh transform into its geometry."),
        DefaultValue(typeof(bool), "true")]
        public virtual bool BakeMeshTransforms
        {
            get { return bakeMeshTransform; }
            set { bakeMeshTransform = value; }
        }

        private bool generateTangentFrame = false;
        [DisplayName("Generate Tangent Frame"),
        Description("If enabled, generates the tangent and binormal for each mesh's vertex."),
        DefaultValue(typeof(bool), "false")]
        public virtual bool GenerateTangentFrame
        {
            get { return generateTangentFrame; }
            set { generateTangentFrame = value; }
        }

        private string texturePath = "";
        [DisplayName("Texture Path"),
        Description("If set, changes the path of all texture resources. " +
            "The new path can be an absolute or relative path."),
        DefaultValue(typeof(string), "")]
        public virtual string TexturePath
        {
            get { return texturePath; }
            set { texturePath = value; }
        }

        private PathType texturePathType = PathType.Relative;
        [DisplayName("Texture Path Type"),
        Description("Indicates whether the \"Texture Path\" is an absolute or relative path." +
            "If the path is relative, it is relative to the source asset path."),
        DefaultValue(typeof(PathType), "Relative")]
        public virtual PathType TexturePathType
        {
            get { return texturePathType; }
            set { texturePathType = value; }
        }

        private string splitAnimationFilename;
        [DisplayName("Split Animation Filename"),
        Description("If set, this file will be used to split the model's animations."),
        DefaultValue(typeof(string), "")]
        public virtual string SplitAnimationFilename
        {
            get { return splitAnimationFilename; }
            set { splitAnimationFilename = value; }
        }

        /*
        private bool changeTexturePathEnabled = false;
        [DisplayName("Change Texture Path Enabled"),
        Description("Indicates whether the \"Change Texture Path\" parameter is enabled."),
        DefaultValue(typeof(bool), "false")]
        public virtual bool ChangeTexturePathEnabled
        {
            get { return changeTexturePathEnabled; }
            set { changeTexturePathEnabled = value; }
        }
        */

        private Vector3 modelRotate = Vector3.Zero;
        [DisplayName("Model Rotation"),
        Description("Rotates the model a specified number of degrees over the Y, X and Z axis in this order."),
        DefaultValue(typeof(Vector3), "0, 0, 0")]
        public virtual Vector3 ModelRotate
        {
            get { return modelRotate; }
            set { modelRotate = value; }
        }

        private float modelScale = 1.0f;
        [DisplayName("Model Scaling"),
        Description("Uniformly scales the model."),
        DefaultValue(typeof(float), "1")]
        public virtual float ModelScale
        {
            get { return modelScale; }
            set { modelScale = value; }
        }

        private bool dumpHierarchyEnabled = false;
        [DisplayName(".Dump Hierarchy"),
        Description("Dump model hierarchy for debugging."),
        DefaultValue(typeof(bool), "false")]
        public virtual bool DumpHierarchyEnabled
        {
            get { return dumpHierarchyEnabled; }
            set { dumpHierarchyEnabled = value; }
        }

        private bool dumpAnimations = false;
        [DisplayName(".Dump Animations"),
        Description("Dump model animations for debugging."),
        DefaultValue(typeof(bool), "false")]
        public virtual bool DumpAnimationsEnabled
        {
            get { return dumpAnimations; }
            set { dumpAnimations = value; }
        }

        #endregion

        public override SkinnedModelContent Process(NodeContent input, ContentProcessorContext context)
        {
            if (input == null)
                throw new ArgumentNullException("input");

            if (dumpHierarchyEnabled)
                DumpHierarchy(input, context);
            if (dumpAnimations)
                DumpAnimations(input, context);

            BoneContent rootBone = MeshHelper.FindSkeleton(input);
            ValidateModel(input, rootBone, context);

            // Transform scene according to user defined rotation and scale
            TransformScene(input);

            // Processes model's meshes
            ProcessMeshes(input, context);

            // Process model's materials
            ProcessMaterials(input, context);

            // Processes model's skeleton
            SkinnedModelBoneContentCollection skinnedModelBoneCollection =
                ProcessBones(rootBone, context);

            // Processes model's animations
            AnimationClipContentDictionary animationClipDictionary =
                ProcessAnimations(input, rootBone.Animations, skinnedModelBoneCollection, context);

            // Uses the default model processor
            ModelContent modelContent =
                context.Convert<NodeContent, ModelContent>(input, "ModelProcessor");

            List<SkinnedModelMeshContent> parts = new List<SkinnedModelMeshContent>();

            foreach (ModelMeshContent meshContent in modelContent.Meshes)
            {
                foreach (ModelMeshPartContent partContent in meshContent.MeshParts)
                {
                    parts.Add(new SkinnedModelMeshContent(partContent.NumVertices, partContent.PrimitiveCount, partContent.VertexBuffer, partContent.IndexBuffer, partContent.Material));
                }
            }

            return new SkinnedModelContent(new SkinnedModelMeshContentCollection(parts), skinnedModelBoneCollection, animationClipDictionary);
        }

        private void TransformScene(NodeContent input)
        {
            Matrix transform = Matrix.Identity;

            // Rotate transfom
            if (modelRotate != Vector3.Zero)
            {
                Vector3 degreeRotation;
                degreeRotation.X = MathHelper.ToRadians(modelRotate.X);
                degreeRotation.Y = MathHelper.ToRadians(modelRotate.Y);
                degreeRotation.Z = MathHelper.ToRadians(modelRotate.Z);

                transform = Matrix.CreateRotationY(degreeRotation.Y) *
                    Matrix.CreateRotationX(degreeRotation.X) *
                        Matrix.CreateRotationZ(degreeRotation.Z);
            }

            // Scale transform
            if (modelScale != 1)
            {
                transform = Matrix.CreateScale(modelScale) * transform;
            }

            // Transform scene
            if (transform != Matrix.Identity)
            {
                MeshHelper.TransformScene(input, transform);
            }
        }

        private void ProcessMeshes(NodeContent node, ContentProcessorContext context)
        {
            MeshContent mesh = node as MeshContent;
            if (mesh != null)
            {
                // Validate mesh
                if (!ValidateMesh(mesh, context))
                    return;

                MeshHelper.OptimizeForCache(mesh);

                // Process the entire mesh
                ProcessMesh(mesh, context);

                // Now process each of its geometries
                foreach (GeometryContent geometry in mesh.Geometry)
                {
                    ProcessGeometry(geometry, context);
                }
            }

            foreach (NodeContent child in node.Children)
                ProcessMeshes(child, context);
        }

        protected virtual void ProcessMesh(MeshContent mesh, ContentProcessorContext context)
        {
            if (bakeMeshTransform)
            {
                Matrix vertexTransform = mesh.AbsoluteTransform;

                // Transform the position of all the vertices
                for (int i = 0; i < mesh.Positions.Count; i++)
                {
                    mesh.Positions[i] = Vector3.Transform(mesh.Positions[i], vertexTransform);
                }
            }

            if (generateTangentFrame)
            {
                MeshHelper.CalculateTangentFrames(mesh,
                    VertexChannelNames.TextureCoordinate(0),
                    VertexChannelNames.Tangent(0),
                    VertexChannelNames.Binormal(0));

            }
        }

        protected virtual void ProcessGeometry(GeometryContent geometry, ContentProcessorContext context)
        {
            if (bakeMeshTransform)
            {
                Matrix vertexTransform = geometry.Parent.AbsoluteTransform;
                Matrix vectorTransform = Matrix.Transpose(Matrix.Invert(vertexTransform));

                foreach (VertexChannel vertexChannel in geometry.Vertices.Channels)
                {
                    if (IsVectorChannel(vertexChannel) && vertexChannel.ElementType == typeof(Vector3))
                    {
                        // Cast for a channel of type Vector3
                        VertexChannel<Vector3> vectorChannel = (VertexChannel<Vector3>)vertexChannel;

                        // Transform all the vectors
                        for (int i = 0; i < vectorChannel.Count; i++)
                        {
                            vectorChannel[i] = Vector3.Transform(vectorChannel[i], vectorTransform);
                            vectorChannel[i].Normalize();
                        }
                    }
                }
            }
        }

        /// <summary>
        /// Traverses the model's mesh parts and processes each of its materials.
        /// Although the materials can be shared between multiple mesh parts, each material is 
        /// processed one unique time.
        /// </summary>
        /// <param name="node">The root node of the model.</param>
        /// <param name="context">The content processor context.</param>
        private void ProcessMaterials(NodeContent node, ContentProcessorContext context)
        {
            // Hash containing the processed materials
            Dictionary<MaterialContent, MaterialContent> processedMaterials =
                new Dictionary<MaterialContent, MaterialContent>();

            ProcessMaterials(node, processedMaterials, context);
        }

        /// <summary>
        /// Traverses the model's mesh parts and processes each of its materials.
        /// Although the materials can be shared between multiple mesh parts, each material is 
        /// processed one unique time.
        /// </summary>
        /// <param name="node">The root node of the model.</param>
        /// <param name="processedMaterials">Dictionary used to store the processed materials.</param>
        /// <param name="context">The content processor context.</param>
        private void ProcessMaterials(NodeContent node,
            Dictionary<MaterialContent, MaterialContent> processedMaterials, ContentProcessorContext context)
        {
            MeshContent mesh = node as MeshContent;
            if (mesh != null)
            {
                foreach (GeometryContent geometry in mesh.Geometry)
                {
                    // Validate the geometry
                    if (!ValidateGeometry(geometry, context))
                        continue;

                    // Check if the material of the mesh part has already been processed
                    MaterialContent processedMaterial;
                    if (!processedMaterials.TryGetValue(geometry.Material, out processedMaterial))
                    {
                        processedMaterial = ProcessMaterial(geometry.Material, context);

                        // Add the material to the processed material hash
                        processedMaterials.Add(geometry.Material, processedMaterial);
                    }

                    // Set the processed material
                    geometry.Material = processedMaterial;
                }
            }

            foreach (NodeContent child in node.Children)
                ProcessMaterials(child, processedMaterials, context);
        }

        /// <summary>
        /// Processes the material of each MeshPart in a model.
        /// </summary>
        /// <param name="materialContent">The material to be processed.</param>
        /// <param name="context">The content processor context.</param>
        /// <returns>The processed material.</returns>
        protected virtual MaterialContent ProcessMaterial(MaterialContent materialContent,
            ContentProcessorContext context)
        {

            OpaqueDataDictionary processorParameters = new OpaqueDataDictionary();

            foreach (string key in materialContent.Textures.Keys)
            {
                ExternalReference<TextureContent> texture = materialContent.Textures[key];

                if (!String.IsNullOrEmpty(texturePath))
                {
                    string fullFilePath;

                    if (texturePathType == PathType.Relative)
                    {
                        // If relative path
                        string sourceAssetPath = Path.GetDirectoryName(materialContent.Identity.SourceFilename);
                        fullFilePath = Path.GetFullPath(
                            Path.Combine(sourceAssetPath, texturePath));
                    }
                    else
                    {
                        fullFilePath = texturePath;
                    }

                    texture.Filename = Path.Combine(fullFilePath,
                        Path.GetFileName(texture.Filename));
                }
            }

            //            processorParameters.Add("DefaultEffect", MaterialProcessorDefaultEffect.SkinnedEffect);

            return
                context.Convert<MaterialContent, MaterialContent>(materialContent,
                    typeof(SkinnedModelMaterialProcessor).Name, processorParameters);
        }

        /// <summary>
        /// Extract and processes all the bones (BoneContent) of the model generating a 
        /// SkinnedModelBoneCollection.
        /// </summary>
        /// <param name="rootBone"></param>
        /// <returns></returns>
        private SkinnedModelBoneContentCollection ProcessBones(BoneContent rootBone,
            ContentProcessorContext context)
        {
            List<SkinnedModelBoneContent> skinnedBoneList =
                new List<SkinnedModelBoneContent>(SkinnedModelBasicEffect.MaxSupportedBones);

            ProcessBones(rootBone, null, skinnedBoneList, context);
            return new SkinnedModelBoneContentCollection(skinnedBoneList);
        }

        /// <summary>
        /// Recursively process each BoneContent of the model generating a new SkinnedModelBone
        /// </summary>
        /// <param name="boneContent"></param>
        /// <param name="skinnedBoneList"></param>
        /// <param name="skinnedModelParentBone"></param>
        /// <returns></returns>
        private SkinnedModelBoneContent ProcessBones(BoneContent boneContent,
            SkinnedModelBoneContent skinnedModelParentBone,
            List<SkinnedModelBoneContent> skinnedBoneList, ContentProcessorContext context)
        {
            // Add the current boneContent to the skinned boneContent list
            ushort boneIndex = (ushort)skinnedBoneList.Count;

            // Decompose boneContent bind pose from the transform matrix
            Pose bindPose;
            boneContent.Transform.Decompose(out bindPose.Scale, out bindPose.Orientation,
                out bindPose.Translation);

            // Calculates boneContent inverse bind pose
            Matrix inverseBindPose = Matrix.Invert(boneContent.AbsoluteTransform);

            // Create the skinned model's boneContent and add it to the skinned model's boneContent list
            SkinnedModelBoneContent skinnedModelBone =
                new SkinnedModelBoneContent(boneIndex, boneContent.Name, bindPose, inverseBindPose);
            skinnedBoneList.Add(skinnedModelBone);

            // Process all children
            List<SkinnedModelBoneContent> skinnedBoneChildrenList = new List<SkinnedModelBoneContent>();
            foreach (NodeContent nodeContent in boneContent.Children)
            {
                // Validate the bone
                if (!ValidateBone(nodeContent, context))
                    continue;

                BoneContent childBoneContent = nodeContent as BoneContent;
                SkinnedModelBoneContent skinnedBoneChild =
                    ProcessBones(childBoneContent, skinnedModelBone, skinnedBoneList, context);

                skinnedBoneChildrenList.Add(skinnedBoneChild);
            }

            // Sets boneContent parent and children
            skinnedModelBone.Parent = skinnedModelParentBone;
            skinnedModelBone.Children = new SkinnedModelBoneContentCollection(skinnedBoneChildrenList);

            return skinnedModelBone;
        }

        private AnimationClipContentDictionary ProcessAnimations(NodeContent input,
            AnimationContentDictionary animationDictionary,
            SkinnedModelBoneContentCollection boneCollection,
            ContentProcessorContext context)
        {
            // Create a collection here (Does not need a dictionary here)
            Dictionary<string, AnimationClipContent> animationClipDictionary =
                new Dictionary<string, AnimationClipContent>();

            foreach (AnimationContent animation in animationDictionary.Values)
            {
                // Validate animation
                if (!ValidateAnimation(animation, context))
                    continue;

                Dictionary<string, AnimationChannelContent> animationChannelDictionary =
                    new Dictionary<string, AnimationChannelContent>();

                // Process each animation channel (One channel per bone)
                foreach (KeyValuePair<string, AnimationChannel> animationChannelPair in
                    animation.Channels)
                {
                    // Validate animation channel
                    if (!ValidateAnimationChannel(animationChannelPair, animation, boneCollection,
                        context))
                        continue;

                    List<AnimationKeyframeContent> keyframeList =
                        new List<AnimationKeyframeContent>(animationChannelPair.Value.Count);

                    // Process all the keyframes of that channel
                    foreach (AnimationKeyframe channelKeyframe in animationChannelPair.Value)
                    {
                        // Extract the keyframe pose from its transform matrix
                        Pose keyframePose;
                        channelKeyframe.Transform.Decompose(out keyframePose.Scale,
                            out keyframePose.Orientation, out keyframePose.Translation);

                        keyframeList.Add(
                            new AnimationKeyframeContent(channelKeyframe.Time, keyframePose));
                    }

                    // Sort the keyframes by time
                    keyframeList.Sort();

                    animationChannelDictionary.Add(animationChannelPair.Key,
                        new AnimationChannelContent(keyframeList));
                }

                AnimationClipContent animationClip = new AnimationClipContent(
                    animation.Name, new AnimationChannelContentDictionary(animationChannelDictionary),
                    animation.Duration);

                animationClipDictionary.Add(animation.Name, animationClip);
            }

            // Split animations
            if (!string.IsNullOrEmpty(splitAnimationFilename))
                SplitAnimations(input, animationClipDictionary, context);

            return new AnimationClipContentDictionary(animationClipDictionary);
        }

        private string ParseChildElementInnerText(XmlElement element, string childName)
        {
            foreach (XmlNode childNode in element)
            {
                XmlElement childElement = childNode as XmlElement;
                if (childElement != null && childElement.Name.ToLower() == childName)
                {
                    return childElement.InnerText;
                }
            }

            return null;
        }

        private List<SplitTask> ParseSplitAnimationTaskList(XmlElement animationElement,
            int animationFramerate, TimeSpan animationDuration, ContentProcessorContext context)
        {
            List<SplitTask> splitAnimations = new List<SplitTask>();
            List<string> splitAnimationNames = new List<string>();

            string animationName = ParseChildElementInnerText(animationElement, "name");
            foreach (XmlNode childNode in animationElement.ChildNodes)
            {
                XmlElement childElement = childNode as XmlElement;
                if (childElement != null && childElement.Name.ToLower() == "splittask")
                {
                    // Try to get the inner text of all available tags
                    string splitName = ParseChildElementInnerText(childElement, "name");
                    string startTimeText = ParseChildElementInnerText(
                        childElement, "starttimeseconds");
                    string endTimeText = ParseChildElementInnerText(
                        childElement, "endtimeseconds");
                    string startFrameText = ParseChildElementInnerText(
                        childElement, "startframe");
                    string endFrameText = ParseChildElementInnerText(
                        childElement, "endframe");

                    // Check if the split animation has a name
                    if (splitName == null)
                    {
                        throw new InvalidContentException(string.Format(
                            "An split animation on animation {0} does not have tag.",
                                animationName));
                    }

                    // Check if the user have defined both start time and frame
                    if (startTimeText != null && startFrameText != null)
                    {
                        context.Logger.LogWarning(null, null, string.Format(
                            "Split animation {0} on animation {1} has both start time and " +
                            "start frame tags. Start frame will be discarded.",
                            splitName, animationName));
                    }

                    // Check if the user have defined both end time and frame
                    if (endTimeText != null && endFrameText != null)
                    {
                        context.Logger.LogWarning(null, null, string.Format(
                            "Split animation {0} on animation {1} has both end time and " +
                            "end frame tags. End frame will be discarded.",
                            splitName, animationName));
                    }

                    // Parse start time tag
                    float startTime;
                    if (startTimeText != null)
                    {
                        try
                        {
                            startTime = float.Parse(startTimeText);
                        }
                        catch (Exception)
                        {
                            throw new InvalidContentException(string.Format(
                                "Split animation {0} on animation {1} has an invalid start time.",
                                splitName, animationName));
                        }
                    }
                    else if (startFrameText != null)
                    {
                        try
                        {
                            int startFrame = int.Parse(startFrameText);
                            startTime = startFrame / (float)animationFramerate;
                        }
                        catch (Exception)
                        {
                            throw new InvalidContentException(string.Format(
                                "Split animation {0} on animation {1} has an invalid start time.",
                                splitName, animationName));
                        }
                    }
                    else
                    {
                        throw new InvalidContentException(string.Format(
                            "Split animation {0} on animation {1} does not have a start time tag.",
                            splitName, animationName));
                    }

                    // Parse end time tag
                    float endTime;
                    if (endTimeText != null)
                    {
                        try
                        {
                            endTime = float.Parse(endTimeText);
                        }
                        catch (Exception)
                        {
                            throw new InvalidContentException(string.Format(
                                "Split animation {0} on animation {1} has an invalid end time.",
                                splitName, animationName));
                        }
                    }
                    else if (endFrameText != null)
                    {
                        try
                        {
                            int endFrame = int.Parse(endFrameText);
                            endTime = endFrame / (float)animationFramerate;
                        }
                        catch (Exception)
                        {
                            throw new InvalidContentException(string.Format(
                                "Split animation {0} on animation {1} has an invalid end time.",
                                splitName, animationName));
                        }
                    }
                    else
                    {
                        throw new InvalidContentException(string.Format(
                            "Split animation {0} on animation {1} does not have an end time tag.",
                            splitName, animationName));
                    }

                    // Validate split animation name
                    if (splitAnimationNames.Contains(splitName))
                    {
                        throw new InvalidContentException(string.Format(
                            "Animation {0} contains two split animations named {1}.",
                            animationName, splitName));
                    }
                    splitAnimationNames.Add(splitName);

                    // Validate split animation duration
                    if (endTime <= startTime)
                    {
                        throw new InvalidContentException(string.Format(
                            "Split animation {0} on animation {1} has a duration less than or " +
                            "equals to zero.", splitName, animationName));
                    }

                    // Validate split animation start time
                    if (startTime < 0)
                    {
                        throw new InvalidContentException(string.Format(
                            "Split animation {0} on animation {1} does not have a positive " +
                            "start time.", splitName, animationName));
                    }

                    // Validate split animation end time
                    if (endTime > animationDuration.TotalSeconds)
                    {
                        throw new InvalidContentException(string.Format(
                            "Split animation {0} on animation {1} has an end time bigger than " +
                            "the original animation duration.", splitName, animationName));
                    }

                    splitAnimations.Add(new SplitTask(splitName, startTime, endTime));
                }
            }

            return splitAnimations;
        }

        private void SplitAnimations(NodeContent input,
            IDictionary<string, AnimationClipContent> animationDictionary,
            ContentProcessorContext context)
        {
            string sourceAssetPath = Path.GetDirectoryName(input.Identity.SourceFilename);
            string fullFilePath = Path.GetFullPath(
                Path.Combine(sourceAssetPath, splitAnimationFilename));

            // Read the XML document
            XmlDocument xmlDocument;
            try
            {
                if (!File.Exists(fullFilePath))
                    throw new FileNotFoundException();

                xmlDocument = new XmlDocument();
                xmlDocument.Load(fullFilePath);
            }
            catch (FileNotFoundException)
            {
                throw new InvalidContentException("Missing animation split file: " +
                    fullFilePath);
            }
            catch (XmlException e)
            {
                throw new InvalidContentException(
                    String.Format("Error parsing animation split file: {0}{1}{1}{2}",
                    fullFilePath, Environment.NewLine, e));
            }

            // Find the ANIMATIONS root tag
            XmlElement animationsElement = null;
            foreach (XmlNode xmlNode in xmlDocument.ChildNodes)
            {
                XmlElement xmlElement = xmlNode as XmlElement;
                if (xmlElement != null && xmlElement.Name.ToLower() == "animations")
                {
                    animationsElement = xmlElement;
                    break;
                }
            }

            if (animationsElement == null)
            {
                context.Logger.LogWarning(null, null, "Split animation document does not " +
                "contain an <ANIMATIONS> tag and will be skipped.");
                return;
            }

            // Parse each ANIMATION tag
            bool containsAnimationElement = false;
            foreach (XmlNode animationNode in animationsElement.ChildNodes)
            {
                XmlElement animationElement = animationNode as XmlElement;
                if (animationElement != null && animationElement.Name.ToLower() == "animation")
                {
                    containsAnimationElement = true;

                    string animationName = ParseChildElementInnerText(
                        animationElement, "name");
                    string animationFramerateText = ParseChildElementInnerText(
                        animationElement, "framerate");

                    if (animationName == null)
                    {
                        throw new InvalidContentException("Split animation document contains " +
                            "an animation that does not have a <NAME> tag.");
                    }

                    int animationFramerate = DefaultAnimationFramerate;
                    if (animationFramerateText == null)
                    {
                        context.Logger.LogWarning(null, null, "Using the default {0} " +
                            "frames per second framerate to split animations.");
                    }
                    else
                    {
                        try
                        {
                            animationFramerate = int.Parse(animationFramerateText);
                        }
                        catch (Exception)
                        {
                            throw new InvalidContentException(string.Format(
                                "Animation {0} has an invalid framerate.",
                                animationName));
                        }
                    }

                    // Try to get the animation in the animation dictionary
                    AnimationClipContent animationClip;
                    if (animationDictionary.TryGetValue(animationName, out animationClip))
                    {
                        // Get the list of split animations
                        List<SplitTask> splitAnimationTaskList =
                            ParseSplitAnimationTaskList(animationElement, animationFramerate,
                            animationClip.Duration, context);

                        // Get the new animation clips
                        List<KeyValuePair<string, AnimationClipContent>> keySplitAnimationClips =
                            SplitAnimation(animationClip, splitAnimationTaskList, context);

                        // Add each animation clip to the dictionary
                        foreach (KeyValuePair<string, AnimationClipContent> keySplitAnimationClip
                            in keySplitAnimationClips)
                        {
                            animationDictionary.Add(keySplitAnimationClip);
                        }

                    }
                    else
                    {
                        throw new InvalidContentException(string.Format(
                                "Input model does not have animation {0}.",
                                animationName));
                    }
                }
            }

            if (!containsAnimationElement)
            {
                context.Logger.LogWarning(null, null, "Split animation document does not contain " +
                    "any <ANIMATION> tag.");
            }
        }

        private List<KeyValuePair<string, AnimationClipContent>> SplitAnimation(
            AnimationClipContent animationClip, List<SplitTask> splitAnimationTaskList,
            ContentProcessorContext context)
        {
            List<KeyValuePair<string, AnimationClipContent>> keyAnimationClipList =
                new List<KeyValuePair<string, AnimationClipContent>>(splitAnimationTaskList.Count);

            foreach (SplitTask splitAnimationTask in splitAnimationTaskList)
            {
                //context.Logger.LogImportantMessage("Split animation " +
                //splitAnimationTask.Name + " [" + splitAnimationTask.StartTimeSeconds +
                //" : " + splitAnimationTask.EndTimeSeconts + "]");

                Dictionary<string, AnimationChannelContent> splitAnimationChannelsDictionary =
                    new Dictionary<string, AnimationChannelContent>();

                TimeSpan splitAnimationDuration = TimeSpan.Zero;

                foreach (KeyValuePair<string, AnimationChannelContent> animationChannelPair in
                    animationClip.Channels)
                {
                    List<AnimationKeyframeContent> splitKeyframes =
                        new List<AnimationKeyframeContent>();

                    TimeSpan? channelStartTime = null;
                    //string times = "";
                    foreach (AnimationKeyframeContent keyframe in animationChannelPair.Value)
                    {
                        if (keyframe.Time.TotalSeconds >= splitAnimationTask.StartTimeSeconds &&
                            keyframe.Time.TotalSeconds <= splitAnimationTask.EndTimeSeconts)
                        {
                            // Get the time of the first keyframe found
                            if (channelStartTime == null)
                                channelStartTime = keyframe.Time;

                            // Add keyframe
                            TimeSpan newKeyframeTime = keyframe.Time - channelStartTime.Value;
                            splitKeyframes.Add(new AnimationKeyframeContent(
                                newKeyframeTime, keyframe.Pose));

                            if (newKeyframeTime > splitAnimationDuration)
                                splitAnimationDuration = newKeyframeTime;

                            //times += string.Format("{0:F4}, ", newKeyframeTime.TotalSeconds);
                        }
                    }

                    //context.Logger.LogImportantMessage("-- {0}. Times: {1}", 
                    //animationChannelPair.Key, times);

                    if (splitKeyframes.Count > 0)
                        splitAnimationChannelsDictionary.Add(animationChannelPair.Key,
                            new AnimationChannelContent(splitKeyframes));
                }

                // Is it better to set the duration as the time of the last frame?
                //TimeSpan splitAnimationDuration = TimeSpan.FromSeconds(
                //splitAnimationTask.EndTimeSeconts - splitAnimationTask.StartTimeSeconds);

                //splitAnimationDuration += TimeSpan.FromSeconds(3.0f/60);

                // Create a new animation clip
                AnimationClipContent splitAnimationClip = new AnimationClipContent(
                    splitAnimationTask.Name,
                    new AnimationChannelContentDictionary(splitAnimationChannelsDictionary),
                    splitAnimationDuration);

                // Add the new animation clip to the animation clip list
                keyAnimationClipList.Add(new KeyValuePair<string, AnimationClipContent>(
                    splitAnimationTask.Name, splitAnimationClip));
            }

            return keyAnimationClipList;
        }

        private bool IsVectorChannel(VertexChannel vertexChannel)
        {
            return
                (vertexChannel.Name.StartsWith("Normal") || vertexChannel.Name.StartsWith("Tangent") ||
                    vertexChannel.Name.StartsWith("Binormal"));
        }

        private bool ValidateModel(NodeContent input, BoneContent rootBone,
            ContentProcessorContext context)
        {
            // Finds the root bone
            if (rootBone == null)
            {
                throw new InvalidContentException("Input model does not contain a skeleton.");
            }

            // Validate maximum supported bones
            IList<BoneContent> boneList = MeshHelper.FlattenSkeleton(rootBone);
            if (boneList.Count > SkinnedModelBasicEffect.MaxSupportedBones)
            {
                throw new InvalidContentException(string.Format(
                    "Model's skeleton has {0} bones, but the maximum supported is {1}.",
                    boneList.Count, SkinnedModelBasicEffect.MaxSupportedBones));
            }

            // Find animations
            AnimationContentDictionary animationDictionary = rootBone.Animations;
            if (animationDictionary.Count == 0)
            {
                context.Logger.LogWarning(null, rootBone.Identity,
                    "Input model does not contain any animation.");
            }

            return true;
        }

        private bool ValidateMesh(MeshContent mesh, ContentProcessorContext context)
        {
            if (mesh.Parent is BoneContent)
            {
                context.Logger.LogWarning(null, mesh.Identity,
                    "Mesh {0} is a child of bone {1}. Meshes that are children of bones might " +
                    "not be handled correct.", mesh.Name, mesh.Parent.Name);
            }

            return true;
        }

        private bool ValidateGeometry(GeometryContent geometry, ContentProcessorContext context)
        {
            // Check if the geometry has material
            if (geometry.Material == null)
            {
                throw new InvalidContentException(string.Format(
                    "Mesh {0} has a geometry that does not have a material.", geometry.Parent.Name));
            }

            // Check if the geometry has skinning information
            if (!geometry.Vertices.Channels.Contains(VertexChannelNames.Weights()))
            {
                throw new InvalidContentException(string.Format("Mesh {0} has a geometry " +
                    "that does not have a skinning blend weights channel.", geometry.Parent.Name));
            }

            return true;
        }

        private bool ValidateBone(NodeContent nodeContent, ContentProcessorContext context)
        {
            BoneContent boneContent = nodeContent as BoneContent;
            if (boneContent == null)
            {
                context.Logger.LogWarning(null, nodeContent.Identity, string.Format(
                    "Node {0} is invalid inside the model's skeleton and will be skipped.",
                    nodeContent.Name));

                return false;
            }

            return true;
        }

        private bool ValidateAnimation(AnimationContent animation, ContentProcessorContext context)
        {
            // Check if this animation has any channel
            if (animation.Channels.Count == 0)
            {
                context.Logger.LogWarning(null, animation.Identity, String.Format(
                    "Animation {0} does not contain any channel and will be skipped.",
                    animation.Name));

                return false;
            }

            // Check if this channel has any keyframe
            if (animation.Duration <= TimeSpan.Zero)
            {
                context.Logger.LogWarning(null, animation.Identity, String.Format(
                    "Animation {0} has a zero duration and will be skipped.", animation.Name));

                return false;
            }

            return true;
        }

        private bool ValidateAnimationChannel(
            KeyValuePair<string, AnimationChannel> animationChannelPair,
            AnimationContent parentAnimation, SkinnedModelBoneContentCollection boneCollection,
            ContentProcessorContext context)
        {
            // Check if this channel has any keyframe
            if (animationChannelPair.Value.Count == 0)
            {
                context.Logger.LogWarning(null, parentAnimation.Identity, String.Format(
                    "Channel {0} in animation {1} does not contain any keyframe and will be skipped.",
                    animationChannelPair.Key, parentAnimation.Name));

                return false;
            }

            // Check if the animation channel exists in the skeleton
            bool boneFound = false;
            foreach (SkinnedModelBoneContent boneContent in boneCollection)
            {
                if (boneContent.Name.Equals(animationChannelPair.Key))
                {
                    boneFound = true;
                    break;
                }
            }

            if (!boneFound)
            {
                context.Logger.LogWarning(null, parentAnimation.Identity, String.Format(
                    "Channel {0} in animation {1} affects a bone that does not exists in the " +
                    "model's skeleton and will be skipped.", animationChannelPair.Key,
                    parentAnimation.Name));

                return false;
            }

            return true;
        }

        private void DumpHierarchy(NodeContent input, ContentProcessorContext context)
        {
            context.Logger.LogImportantMessage("Dumping Hierarchy:");
            DumpHierarchy(input, " ", context);
            context.Logger.LogImportantMessage("");
        }

        private void DumpHierarchy(NodeContent input, string pad, ContentProcessorContext context)
        {
            string[] typeName = input.ToString().Split(new char[] { '.' });
            context.Logger.LogImportantMessage(pad + input.Name + " : " +
                typeName[typeName.Length - 1]);

            foreach (NodeContent child in input.Children)
                DumpHierarchy(child, "--" + pad, context);
        }

        private void DumpAnimations(NodeContent input, ContentProcessorContext context)
        {
            context.Logger.LogImportantMessage("Dumping Animations:");

            BoneContent rootBone = MeshHelper.FindSkeleton(input);
            AnimationContentDictionary animationDictionary = rootBone.Animations;

            if (animationDictionary.Count == 0)
                context.Logger.LogImportantMessage("None.");

            foreach (AnimationContent animation in animationDictionary.Values)
            {
                context.Logger.LogImportantMessage(string.Format("Animation {0}: " +
                    "{1} channels, {2:F4} duration", animation.Name, animation.Channels.Count,
                    animation.Duration.TotalSeconds));

                foreach (KeyValuePair<string, AnimationChannel> animationChannel in animation.Channels)
                {
                    AnimationKeyframe lastKeyframe =
                        animationChannel.Value[animationChannel.Value.Count - 1];

                    //context.Logger.LogImportantMessage(string.Format("-- Channel {0}: " +
                    //"{1:F4} duration.", animationChannel.Key,
                    //lastKeyframe.Time.TotalSeconds));

                    StringBuilder keyframeTimes = new StringBuilder();
                    if (animationChannel.Value.Count > 0)
                    {
                        keyframeTimes.Append(string.Format("{0:F4}",
                            animationChannel.Value[0].Time.TotalSeconds));

                        for (int i = 1; i < animationChannel.Value.Count; i++)
                        {
                            keyframeTimes.Append(", " + string.Format("{0:F4}",
                                animationChannel.Value[i].Time.TotalSeconds));
                        }

                        //context.Logger.LogImportantMessage(string.Format("---- {0} Keyframes: {1} ",
                        //animationChannel.Value.Count, keyframeTimes));

                        context.Logger.LogImportantMessage(string.Format("-- Channel {0}: " +
                            "{2:F4} duration, {1} keyframes: {3}", animationChannel.Key,
                            animationChannel.Value.Count, lastKeyframe.Time.TotalSeconds,
                            keyframeTimes));

                        //context.Logger.LogImportantMessage("---- Keyframes time: " + keyframeTimes);
                    }
                }

                context.Logger.LogImportantMessage("");
            }

            // Final breakline
            if (animationDictionary.Values.Count <= 0)
                context.Logger.LogImportantMessage("");
        }
    }
}