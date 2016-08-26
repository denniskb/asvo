/*
 * AssemblyHelper.cs
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
using Microsoft.Xna.Framework;
using Microsoft.Xna.Framework.Content.Pipeline;

namespace XNAnimationPipeline.Pipeline
{
    internal static class AssemblyHelper
    {
        private static readonly string windowsPublicKeyTokens = "4b77d890149fbbaf";
        private static readonly string xboxPublicKeyTokens = "2ca0ea485e068871";
        private static readonly string phonePublicKeyTokens = "f7c734787c6af5aa";

        private static readonly string[] assemblySplitter = {", "};

        internal static string GetRuntimeReader(Type type, TargetPlatform targetPlatform)
        {
            // Type full name
            string typeFullName = type.FullName;

            // Assembly name tokenized
            string fullAssemblyName = type.Assembly.FullName;
            string[] assemblyTokens = fullAssemblyName.Split(assemblySplitter, StringSplitOptions.None);

            return
                typeFullName + ", " + assemblyTokens[0] + ", " + assemblyTokens[1] + ", " +
                    assemblyTokens[2] + ", " + GetAssemblyPublicKey(targetPlatform);
        }

        internal static string GetAssemblyPublicKey(TargetPlatform targetPlatform)
        {
            string publicKey = "PublicKeyToken=";

            switch (targetPlatform)
            {
                case TargetPlatform.Windows:
                    publicKey += windowsPublicKeyTokens;
                    break;

                case TargetPlatform.Xbox360:
                    publicKey += xboxPublicKeyTokens;
                    break;

                case TargetPlatform.WindowsPhone:
                    publicKey += phonePublicKeyTokens;
                    break;

                default:
                    throw new ArgumentException("targetPlatform");
            }

            return publicKey;
        }
    }
}