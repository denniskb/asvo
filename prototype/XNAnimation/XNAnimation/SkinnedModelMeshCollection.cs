/*
 * SkinnedModelMeshCollection.cs
 * Author: Rodrigo 'r2d2rigo' Díaz
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
using System.Collections.Generic;
using System.Collections.ObjectModel;

namespace XNAnimation
{
    public class SkinnedModelMeshCollection : ReadOnlyCollection<SkinnedModelMesh>
    {
        public SkinnedModelMeshCollection(IList<SkinnedModelMesh> list)
            : base(list)
        {
        }
    }
}