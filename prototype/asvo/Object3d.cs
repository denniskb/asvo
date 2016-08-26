using asvo.datastructures;
using Microsoft.Xna.Framework;

namespace asvo
{
    namespace world3D
    {
        /// <summary>
        /// Represents a visible three dimensional object, such as
        /// a box, sphere, vehicle, NPC, ...
        /// Abstracts from the representation (like voxels) and encapsulates
        /// common operations on 3D objects like transformations.
        /// </summary>
        internal class Object3D
        {
            protected BFSOctree _representation;
            private Matrix _rotation, _translation, _transformation;
            public int frame;

            /// <summary>
            /// Creates a new 3D object consiting of the provided octree.
            /// After creation, its transformation matrices reflect its position and
            /// orientation in the world space coordinate system.
            /// </summary>
            /// <param name="representation">The raw 3d data which represents the object in form of a BFSOctree.</param>
            /// <param name="rightHandedCoordinateSystem">Indicates whether the calling
            /// framework uses a right-handed coordinate system and if therefore the
            /// object needs to be translated into this system, if false nothing happens as
            /// octree data is always provided in a left-handed manner.</param>
            public Object3D(BFSOctree representation, bool rightHandedCoordinateSystem)
            {
                this._representation = representation;
                
                _rotation = _translation = Matrix.Identity;

                if (rightHandedCoordinateSystem)
                    _rotation.M33 = -1.0f;

                updateTransformation();
            }

            /// <summary>
            /// Returns the transformations of this object. Rotation is always applied before
            /// translation.
            /// </summary>
            /// <returns>Transformation (rotation * translation) of this object.</returns>
            public Matrix getTransformation()
            {
                return _transformation;
            }

            /// <summary>
            /// Returns the rotation matrix of this model only.
            /// Suitable for transforming normals.
            /// </summary>
            /// <returns>The rotation matrix of this object.</returns>
            public Matrix getRotation()
            {
                return _rotation;
            }

            /// <summary>
            /// Returns the raw 3d data that represents this object.
            /// </summary>
            /// <returns>3D data representing this object in form of a BFSOctree.</returns>
            public BFSOctree getData()
            {
                return _representation;
            }

            /// <summary>
            /// Rotates the object around <paramref name="axis"/> by <paramref name="angle"/>.
            /// </summary>
            /// <param name="axis">The axis to rotate this object around.</param>
            /// <param name="angle">The angle to rotate this object by.</param>
            public void rotate(Vector3 axis, float angle)
            {
                Matrix rotationMatrix = Matrix.CreateFromAxisAngle(axis, angle);
                Matrix.Multiply(ref _rotation, ref rotationMatrix, out _rotation);

                updateTransformation();
            }

            /// <summary>
            /// Translates this object by <paramref name="offset"/>.
            /// </summary>
            /// <param name="offset">The offset to translate this object by.</param>
            public void translate(Vector3 offset)
            {
                Matrix translationMatrix = Matrix.CreateTranslation(offset);
                Matrix.Multiply(ref _translation, ref translationMatrix, out _translation);

                updateTransformation();
            }

            /// <summary>
            /// Caches the transformations of this model after each modification of
            /// its position / orientation.
            /// </summary>
            private void updateTransformation()
            {
                _transformation = _rotation * _translation;
            }
        }
    }
}
