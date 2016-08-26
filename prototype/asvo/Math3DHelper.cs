using Microsoft.Xna.Framework;
using System;
using System.Collections.Generic;
using System.IO;

namespace asvo
{
    namespace tools
    {
        /// <summary>
        /// A helper class that encapsulates common operations NOT provided by
        /// the XNA framework.
        /// </summary>
        internal class Math3DHelper
        {
            private static DoublePoint[] boxpoints = new DoublePoint[8];
            private static DoublePoint[] tripoints = new DoublePoint[3];

            /// <summary>
            /// Computes the product of a 3 dimensional vector with a
            /// 4 dimensional matrix. The Vector is expanded to (x, y, z, 1) and the
            /// result is divided by w.
            /// </summary>
            /// <param name="vector">3 dimensional vector, will be changed to
            /// vector * matrix, normalized (last column of the vector equals 1)</param>
            /// <param name="matrix">4 dimensional matrix</param>     
            public static void mul(ref Vector3 vector, ref Matrix matrix)
            {
                Vector4 result = new Vector4(vector, 1.0f);
                Vector4.Transform(ref result, ref matrix, out result);
                float resMul = 1.0f / result.W;
                result = result * resMul;

                vector.X = result.X;
                vector.Y = result.Y;
                vector.Z = result.Z;
            }

            /// <summary>
            /// Writes the contents of the Vector2 to a binary stream using
            /// the provided BinaryWriter.
            /// Output-format:
            /// <code>
            /// bytes:  type/format:        interpretation:
            /// ===========================================
            /// 4       IEEE 32 bit float   first coordiante
            /// 4       IEEE 32 bit float   second coordinate
            /// </code>
            /// </summary>
            /// <param name="vector">The Vector2 to be written to the stream.</param>
            /// <param name="writer">The BinaryWriter used to write to the stream.</param>            
            public static void export(Vector2 vector, BinaryWriter writer)
            {
                writer.Write(vector.X);
                writer.Write(vector.Y);              
            }

            /// <summary>
            /// Writes the contents of the Vector3 to a binary stream using
            /// the provided BinaryWriter.
            /// Output-format:
            /// <code>
            /// bytes:  type/format:        interpretation:
            /// ===========================================
            /// 4       IEEE 32 bit float   first coordinate
            /// 4       IEEE 32 bit float   second coordinate
            /// 4       IEEE 32 bit float   third coordinate
            /// </code>
            /// </summary>
            /// <param name="vector">The Vector3 to be written to the stream.</param>
            /// <param name="writer">The BinaryWriter used to write to the stream.</param>            
            public static void export(Vector3 vector, BinaryWriter writer)
            {
                writer.Write(vector.X);
                writer.Write(vector.Y);
                writer.Write(vector.Z);
            }

            /// <summary>
            /// Writes the contents of the Vector4 to a binary stream using
            /// the provided BinaryWriter.
            /// Output-format:
            /// <code>
            /// bytes:  type/format:        interpretation:
            /// ===========================================
            /// 4       IEEE 32 bit float   first coordinate
            /// 4       IEEE 32 bit float   second coordinate
            /// 4       IEEE 32 bit float   third coordinate
            /// 4       IEEE 32 bit float   fourth coordinate
            /// </code>
            /// </summary>
            /// <param name="vector">The Vector4 to be written to the stream.</param>
            /// <param name="writer">The BinaryWriter used to write to the stream.</param>            
            public static void export(Vector4 vector, BinaryWriter writer)
            {
                writer.Write(vector.X);
                writer.Write(vector.Y);
                writer.Write(vector.Z);
                writer.Write(vector.W);
            }

            /// <summary>
            /// Writes the contents of the Matrix to a binary stream using
            /// the provided BinaryWriter.
            /// Output-format:
            /// <code>
            /// bytes:  type/format:        interpretation:
            /// ===========================================
            /// 4       IEEE 32 bit float   first row, first column
            /// 4       IEEE 32 bit float   first row, second column
            /// 4       IEEE 32 bit float   first row, third column
            /// 4       IEEE 32 bit float   first row, fourth column
            /// (repeat for second, third and fourth row)
            /// </code>
            /// </summary>
            /// <param name="matrix">The matrix to be written to the stream.</param>
            /// <param name="writer">The BinaryWriter used to write to the stream.</param>            
            public static void export(Matrix matrix, BinaryWriter writer)
            {
                writer.Write(matrix.M11);
                writer.Write(matrix.M12);
                writer.Write(matrix.M13);
                writer.Write(matrix.M14);

                writer.Write(matrix.M21);
                writer.Write(matrix.M22);
                writer.Write(matrix.M23);
                writer.Write(matrix.M24);

                writer.Write(matrix.M31);
                writer.Write(matrix.M32);
                writer.Write(matrix.M33);
                writer.Write(matrix.M34);

                writer.Write(matrix.M41);
                writer.Write(matrix.M42);
                writer.Write(matrix.M43);
                writer.Write(matrix.M44);
            }

            /// <summary>
            /// Rounds a floating point number to the nearest integer.
            /// </summary>
            /// <param name="d">Floating point number to be rounded.</param>
            /// <returns>d rounded to the nearest integer.</returns>
            public static int round(double d)
            {
                double floor = Math.Floor(d);
                if (d - floor <= 0.5)
                    return (int)floor;
                else
                    return (int)Math.Ceiling(d);
            }            

            /// <summary>
            /// Calling wrapper around isectboxtri.
            /// For more info see <see cref="Math3DHelper.isectboxtri"/> documentation.
            /// </summary>
            /// <param name="x">The x-index of the voxel within the virtual uniform grid
            /// defined by the octree.</param>
            /// <param name="y">The y-index of the voxel within the virtual uniform grid
            /// defined by the octree.</param>
            /// <param name="z">The z-index of the voxel within the virtual uniform grid
            /// defined by the octree.</param>
            /// <param name="level">The level of the voxel inside the octree.</param>
            /// <param name="gridDim">The dimenstion of the virtual uniform grid
            /// defined by the octree.</param>
            /// <param name="octreeMin">The minimum vector of the virtual
            /// uniform grid defined by the octree.</param>
            /// <param name="v0">First vertex of the triangle.</param>
            /// <param name="v1">Second vertex of the triangle.</param>
            /// <param name="v2">Third vertex of the triangle.</param>
            /// <param name="triBBmin">Minimum vector of the triangle's bounding box.</param>
            /// <param name="triBBmax">Maximum vector of the triangle's bounding box.</param>
            /// <returns>true, iff the triangle intersects the voxel.</returns>
            public static bool intersects(ushort x, ushort y, ushort z,
                                          byte level, double gridDim,
                                          Vector3 octreeMin,
                                          Vector3 v0, Vector3 v1, Vector3 v2,
                                          Vector3 triBBmin, Vector3 triBBmax)
            {
                uint gridRes = 1u << level;
                double gridResMul = 1.0 / gridRes;
                double voxelMinX = (x * gridDim) * gridResMul + octreeMin.X;
                double voxelMinY = (y * gridDim) * gridResMul + octreeMin.Y;
                double voxelMinZ = (z * gridDim) * gridResMul + octreeMin.Z;

                double voxelMaxX = ((x + 1) * gridDim) * gridResMul + octreeMin.X;
                                
                double voxelDim = voxelMaxX - voxelMinX;
                double voxelHalfDim = voxelDim * 0.5;

                double centerX = voxelMinX + voxelHalfDim;
                double centerY = voxelMinY + voxelHalfDim;
                double centerZ = voxelMinZ + voxelHalfDim;

                double triBBcenterX = triBBmin.X + (((double)triBBmax.X) - ((double)triBBmin.X)) * 0.5;
                double triBBcenterY = triBBmin.Y + (((double)triBBmax.Y) - ((double)triBBmin.Y)) * 0.5;
                double triBBcenterZ = triBBmin.Z + (((double)triBBmax.Z) - ((double)triBBmin.Z)) * 0.5;

                double distX = (centerX - triBBcenterX) * 2.0;
                double distY = (centerY - triBBcenterY) * 2.0;
                double distZ = (centerZ - triBBcenterZ) * 2.0;

                // do bounding boxes intersect?
                if (distX > voxelDim + (((double)triBBmax.X) - ((double)triBBmin.X)) ||
                    distY > voxelDim + (((double)triBBmax.Y) - ((double)triBBmin.Y)) ||
                    distZ > voxelDim + (((double)triBBmax.Z) - ((double)triBBmin.Z)))
                    return false;

                double voxelMaxY = ((y + 1) * gridDim) * gridResMul + octreeMin.Y;
                double voxelMaxZ = ((z + 1) * gridDim) * gridResMul + octreeMin.Z;

                // does voxel contain one of the triangles vertices?
                if ((voxelMinX <= v0.X && v0.X <= voxelMaxX &&
                     voxelMinY <= v0.Y && v0.Y <= voxelMaxY &&
                     voxelMinZ <= v0.Z && v0.Z <= voxelMaxZ) ||

                    (voxelMinX <= v1.X && v1.X <= voxelMaxX &&
                     voxelMinY <= v1.Y && v1.Y <= voxelMaxY &&
                     voxelMinZ <= v1.Z && v1.Z <= voxelMaxZ) ||

                    (voxelMinX <= v2.X && v2.X <= voxelMaxX &&
                     voxelMinY <= v2.Y && v2.Y <= voxelMaxY &&
                     voxelMinZ <= v2.Z && v2.Z <= voxelMaxZ))
                    return true;

                return isectboxtri(centerX, centerY, centerZ,
                                   voxelHalfDim,
                                   v0, v1, v2);
            }

            /// <summary>
            /// Tests whether a triangle intersects an axis-aligned bounding box.
            /// Original code by Mike Vandelay
            /// </summary>
            /// <param name="centerX">The x-component of the box's center</param>
            /// <param name="centerY">The y-component of the box's center</param>
            /// <param name="centerZ">The z-component of the box's center</param>
            /// <param name="boxHalfDim">Half of the box's dimension (our aabb is a cube)</param>
            /// <param name="v0">The first triangle vertex</param>           
            /// <param name="v1">The second triangle vertex</param>            
            /// <param name="v2">The third triangle vertex</param>            
            /// <returns>true, iff the triangle intersects the bounding box</returns>
            private static bool isectboxtri(double centerX, double centerY, double centerZ,
                                            double boxHalfDim,
                                            Vector3 v0, Vector3 v1, Vector3 v2)
            {
                int i = 0;
                boxpoints[i++] = 
                    new DoublePoint(
                        centerX + boxHalfDim, centerY + boxHalfDim, centerZ + boxHalfDim);
                boxpoints[i++] = 
                    new DoublePoint(
                        centerX + boxHalfDim, centerY + boxHalfDim, centerZ - boxHalfDim);
                boxpoints[i++] = 
                    new DoublePoint(
                        centerX + boxHalfDim, centerY - boxHalfDim, centerZ + boxHalfDim);
                boxpoints[i++] = 
                    new DoublePoint(
                        centerX + boxHalfDim, centerY - boxHalfDim, centerZ - boxHalfDim);

                boxpoints[i++] = 
                    new DoublePoint(
                        centerX - boxHalfDim, centerY + boxHalfDim, centerZ + boxHalfDim);
                boxpoints[i++] = 
                    new DoublePoint(
                        centerX - boxHalfDim, centerY + boxHalfDim, centerZ - boxHalfDim);
                boxpoints[i++] = 
                    new DoublePoint(
                        centerX - boxHalfDim, centerY - boxHalfDim, centerZ + boxHalfDim);
                boxpoints[i++] = 
                    new DoublePoint(
                        centerX - boxHalfDim, centerY - boxHalfDim, centerZ - boxHalfDim);

                i = 0;
                tripoints[i++] = new DoublePoint(v0.X, v0.Y, v0.Z);
                tripoints[i++] = new DoublePoint(v1.X, v1.Y, v1.Z);
                tripoints[i++] = new DoublePoint(v2.X, v2.Y, v2.Z);

                // test the x, y, and z axes
                if (!isect(boxpoints, tripoints, DoublePoint.UnitX)) return false;
                if (!isect(boxpoints, tripoints, DoublePoint.UnitY)) return false;
                if (!isect(boxpoints, tripoints, DoublePoint.UnitZ)) return false;

                // test the triangle normal
                DoublePoint triedge1 = tripoints[1].Sub(tripoints[0]);
                DoublePoint triedge2 = tripoints[2].Sub(tripoints[1]);
                DoublePoint trinormal = triedge1.Cross(triedge2);
                if (!isect(boxpoints, tripoints, trinormal)) return false;

                // test the 9 edge cross products
                DoublePoint triedge3 = tripoints[0].Sub(tripoints[2]);               

                if (!isect(boxpoints, tripoints, DoublePoint.UnitX.Cross(triedge1))) return false;
                if (!isect(boxpoints, tripoints, DoublePoint.UnitX.Cross(triedge2))) return false;
                if (!isect(boxpoints, tripoints, DoublePoint.UnitX.Cross(triedge3))) return false;

                if (!isect(boxpoints, tripoints, DoublePoint.UnitY.Cross(triedge1))) return false;
                if (!isect(boxpoints, tripoints, DoublePoint.UnitY.Cross(triedge2))) return false;
                if (!isect(boxpoints, tripoints, DoublePoint.UnitY.Cross(triedge3))) return false;

                if (!isect(boxpoints, tripoints, DoublePoint.UnitZ.Cross(triedge1))) return false;
                if (!isect(boxpoints, tripoints, DoublePoint.UnitZ.Cross(triedge2))) return false;
                if (!isect(boxpoints, tripoints, DoublePoint.UnitZ.Cross(triedge3))) return false;

                return true;
            }

            /// <summary>
            /// Helper function for isectboxtri.
            /// </summary>
            private static bool isect(DoublePoint[] boxpoints,
                                      DoublePoint[] tripoints,
                                      DoublePoint axis)
            {
                if (getmin(boxpoints, axis) > getmax(tripoints, axis)) return false;
                if (getmax(boxpoints, axis) < getmin(tripoints, axis)) return false;
                return true;               
            }

            /// <summary>
            /// Helper function for isectboxtri.
            /// </summary>
            private static double getmin(DoublePoint[] points,
                                         DoublePoint axis)
            {
                double min = double.MaxValue; 

                for (int ctr = 0; ctr < points.Length; ctr++)
                {
                    double dotprod = points[ctr].Dot(axis);
                    if (dotprod < min) min = dotprod;
                }
                return min;
            }

            /// <summary>
            /// Helper function for isectboxtri.
            /// </summary>
            private static double getmax(DoublePoint[] points,
                                         DoublePoint axis)
            {
                double max = double.MinValue; 

                for (int ctr = 0; ctr < points.Length; ctr++)
                {
                    double dotprod = points[ctr].Dot(axis);
                    if (dotprod > max) max = dotprod;
                }
                return max;
            }

            /// <summary>
            /// Helper struct for isectboxtri. Like Vector3, but stores doubles instead
            /// of floats.
            /// </summary>
            private struct DoublePoint
            {
                public double x;
                public double y;
                public double z;

                public static readonly DoublePoint UnitX = new DoublePoint(1.0, 0.0, 0.0);
                public static readonly DoublePoint UnitY = new DoublePoint(0.0, 1.0, 0.0);
                public static readonly DoublePoint UnitZ = new DoublePoint(0.0, 0.0, 1.0);

                /// <summary>
                /// Constructs a new DoublePoint.
                /// </summary>
                /// <param name="x">x-coordinate</param>
                /// <param name="y">y-coordinate</param>
                /// <param name="z">z-coordinate</param>
                public DoublePoint(double x, double y, double z)
                {
                    this.x = x;
                    this.y = y;
                    this.z = z;
                }

                /// <summary>
                /// Subtracts another DoublePoint from this point.
                /// </summary>
                /// <param name="p">The vector to subtract from this one.</param>
                /// <returns>this - p</returns>
                public DoublePoint Sub(DoublePoint p)
                {
                    return new DoublePoint(x - p.x, y - p.y, z - p.z);
                }

                /// <summary>
                /// Computes the dot product between this DoublePoint and
                /// p.
                /// </summary>
                /// <param name="p">The second point.</param>
                /// <returns>dot(this, p)</returns>
                public double Dot(DoublePoint p)
                {
                    return x * p.x + y * p.y + z * p.z;
                }

                /// <summary>
                /// Computes the cross product between this DoublePoint and
                /// p.
                /// </summary>
                /// <param name="p">The second point.</param>
                /// <returns>cross(this, p)</returns>
                public DoublePoint Cross(DoublePoint p)
                {
                    return new DoublePoint(y * p.z - p.y * z,
                                           z * p.x - x * p.z,
                                           x * p.y - y * p.x);                    
                }
            }            
        }
    }
}