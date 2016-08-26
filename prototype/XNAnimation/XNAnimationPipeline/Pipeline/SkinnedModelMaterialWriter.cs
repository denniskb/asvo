/*
 * SkinnedModelMaterialWriter.cs
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
using Microsoft.Xna.Framework;
using Microsoft.Xna.Framework.Content.Pipeline;
using Microsoft.Xna.Framework.Content.Pipeline.Graphics;
using Microsoft.Xna.Framework.Content.Pipeline.Processors;
using Microsoft.Xna.Framework.Content.Pipeline.Serialization.Compiler;
using XNAnimation.Pipeline;
using XNAnimationPipeline.Effects;

namespace XNAnimationPipeline.Pipeline
{
    [ContentTypeWriter]
    public class SkinnedModelMaterialWriter : ContentTypeWriter<SkinnedModelMaterialContent>
    {
        protected override void Write(ContentWriter output, SkinnedModelMaterialContent value)
        {
            output.WriteObject<CompiledEffectContent>(value.SkinnedEffectContent);

            output.Write(value.EmissiveColor);
            output.Write(value.DiffuseColor);
            output.Write(value.SpecularColor);
            output.Write(value.SpecularPower);

            output.Write(value.DiffuseMapEnabled);
            output.Write(value.NormalMapEnabled);
            output.Write(value.SpecularMapEnabled);
            
            output.WriteExternalReference<TextureContent>(value.DiffuseMapContent);
            output.WriteExternalReference<TextureContent>(value.NormalMapContent);
            output.WriteExternalReference<TextureContent>(value.SpecularMapContent);
        }

        public override string GetRuntimeReader(TargetPlatform targetPlatform)
        {
            return AssemblyHelper.GetRuntimeReader(typeof (SkinnedModelBasicEffectReader), targetPlatform);
        }
    }
}