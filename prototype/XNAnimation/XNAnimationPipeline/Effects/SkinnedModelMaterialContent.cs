/*
 * SkinnedModelMaterialContent.cs
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

namespace XNAnimationPipeline.Effects
{
    public class SkinnedModelMaterialContent : MaterialContent
    {
        public Vector3 EmissiveColor { get; set; }
        public Vector3 DiffuseColor { get; set; }
        public Vector3 SpecularColor { get; set; }

        public float SpecularPower { get; set; }
        public bool DiffuseMapEnabled { get; set; }
        public bool NormalMapEnabled { get; set; }
        public bool SpecularMapEnabled { get; set; }

        public ExternalReference<TextureContent> DiffuseMapContent { get; set; }
        public ExternalReference<TextureContent> NormalMapContent { get; set; }
        public ExternalReference<TextureContent> SpecularMapContent { get; set; }

        public CompiledEffectContent SkinnedEffectContent { get; set; }

        internal SkinnedModelMaterialContent()
        {
        }
    }
}