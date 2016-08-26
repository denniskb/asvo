/*
 * SkinnedModelBoneCollection.cs
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
using System.Collections.Generic;
using System.Collections.ObjectModel;

namespace XNAnimation
{
    public class SkinnedModelBoneCollection : ReadOnlyCollection<SkinnedModelBone>
    {
        public SkinnedModelBone this[string boneName]
        {
            get
            {
                for (int i = 0; i < Count; i++)
                {
                    if (this[i].Name == boneName)
                    {
                        return this[i];
                    }
                }
                return null;
            }
        }

        public SkinnedModelBoneCollection(IList<SkinnedModelBone> list)
            : base(list)
        {
        }

        public int GetBoneId(string boneName)
        {
            for (int i = 0; i < Count; i++)
            {
                if (this[i].Name == boneName)
                {
                    return i;
                }
            }
            return -1;
        }
    }
}