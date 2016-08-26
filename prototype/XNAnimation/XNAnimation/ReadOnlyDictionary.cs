/*
 * ReadOnlyDictionary.cs
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
using System;
using System.Collections;
using System.Collections.Generic;
using System.Collections.ObjectModel;

namespace XNAnimation
{
    public class ReadOnlyDictionary<T, V> : IDictionary<T, V>
    {
        private static readonly string ReadOnlyException = "Collection is read-only.";
        private readonly IDictionary<T, V> items;

        #region Properties

        V IDictionary<T, V>.this[T key]
        {
            get { return items[key]; }
            set { throw new NotSupportedException(ReadOnlyException); }
        }

        ICollection<T> IDictionary<T, V>.Keys
        {
            get { throw new NotSupportedException(ReadOnlyException); }
        }

        ICollection<V> IDictionary<T, V>.Values
        {
            get { throw new NotSupportedException(ReadOnlyException); }
        }

        public V this[T key]
        {
            get { return items[key]; }
        }

        public ReadOnlyCollection<T> Keys
        {
            get { return new ReadOnlyCollection<T>(new List<T>(items.Keys)); }
        }

        public ReadOnlyCollection<V> Values
        {
            get { return new ReadOnlyCollection<V>(new List<V>(items.Values)); }
        }

        public int Count
        {
            get { return items.Count; }
        }

        public bool IsReadOnly
        {
            get { return true; }
        }

        #endregion

        public ReadOnlyDictionary(IDictionary<T, V> dictionary)
        {
            if (dictionary == null)
                throw new ArgumentNullException("dictionary");

            items = dictionary;
        }

        #region Not supported methods

        void ICollection<KeyValuePair<T, V>>.CopyTo(KeyValuePair<T, V>[] array, int arrayIndex)
        {
        }

        void ICollection<KeyValuePair<T, V>>.Clear()
        {
            throw new NotSupportedException();
        }

        public bool Contains(KeyValuePair<T, V> item)
        {
            throw new NotSupportedException();
        }

        void ICollection<KeyValuePair<T, V>>.Add(KeyValuePair<T, V> item)
        {
            throw new NotSupportedException(ReadOnlyException);
        }

        void IDictionary<T, V>.Add(T key, V value)
        {
            throw new NotSupportedException(ReadOnlyException);
        }

        bool ICollection<KeyValuePair<T, V>>.Remove(KeyValuePair<T, V> item)
        {
            throw new NotSupportedException(ReadOnlyException);
        }

        bool IDictionary<T, V>.Remove(T key)
        {
            throw new NotSupportedException(ReadOnlyException);
        }

        #endregion

        public bool ContainsKey(T key)
        {
            return items.ContainsKey(key);
        }

        /*
        public bool ContainsValue(V value)
        {
            return items.ContainsValue(value);
        }
        */

        public bool TryGetValue(T key, out V value)
        {
            return items.TryGetValue(key, out value);
        }

        public IEnumerator GetEnumerator()
        {
            return items.GetEnumerator();
        }

        IEnumerator<KeyValuePair<T, V>> IEnumerable<KeyValuePair<T, V>>.GetEnumerator()
        {
            return (IEnumerator<KeyValuePair<T, V>>) items.GetEnumerator();
        }
    }
}