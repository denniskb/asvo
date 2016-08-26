using Microsoft.Xna.Framework;
using Microsoft.Xna.Framework.Input;

using asvo.tools;

namespace asvo
{
    namespace world3D
    {
        /// <summary>
        /// Represents a camera - an imaginary observer of a scene in 3D space.
        /// The camera uses perspective projection.
        /// </summary>
        internal struct Camera
        {
            private float _nearPlane, _farPlane;
            private float _ratio;

            private float _fov; //in radians
            private Vector3 _position, _lookAt;

            private Matrix _view, _projection, _viewProjection;
            private Vector3 _eyeVector;

            private bool _buttonDown;
            private Vector2 _start, _end;
            private int _lastScrollWheelValue;

            private static Vector3 unitY = Vector3.UnitY;

            /// <summary>
            /// Constructs a new camera with common properties.
            /// </summary>
            /// <param name="nearPlane">Near clipping plane distance.</param>
            /// <param name="farPlane">Far clipping plane distance.</param>
            /// <param name="pos">Position of the camera.</param>
            /// <param name="lookAt">Target the camera is pointing at.</param>
            /// <param name="ratio">Aspect ratio of the projection (width/height)</param>
            public Camera(float nearPlane, float farPlane, 
                          Vector3 pos, Vector3 lookAt,
                          float ratio)
            {
                this._nearPlane = nearPlane;
                this._farPlane = farPlane;
                this._position = pos;
                this._lookAt = lookAt;
                this._ratio = ratio;

                _fov = MathHelper.PiOver4;

                _view = _projection = _viewProjection = Matrix.Identity;
                _eyeVector = Vector3.UnitY;

                _buttonDown = false;
                _start = _end = new Vector2();
                _lastScrollWheelValue = 0;

                updateMatrices();
            }

            /// <summary>
            /// Sets the field of vision of the camera in radians.
            /// </summary>
            /// <param name="newFov">Field of vision in radians.</param>
            public void setFov(float newFov)
            {
                _fov = newFov;
                updateMatrices();
            }

            /// <summary>
            /// Returns the field of vision of this camera in radians.
            /// </summary>
            /// <returns>Field of vision in radians.</returns>
            public float getFov()
            {
                return _fov;
            }

            /// <summary>
            /// Sets the camera's position.
            /// </summary>
            /// <param name="newPosition">The camera's new position.</param>
            public void setPosition(Vector3 newPosition)
            {
                _position = newPosition;
                updateMatrices();
            }

            /// <summary>
            /// Returns the camera's position.
            /// </summary>
            /// <returns>The camera's position.</returns>
            public Vector3 getPosition()
            {
                return _position;
            }

            /// <summary>
            /// Sets the point the camera looks-at.
            /// </summary>
            /// <param name="newLookAt">The new look-at point.</param>
            public void setLookAt(Vector3 newLookAt)
            {
                _lookAt = newLookAt;
                updateMatrices();
            }

            /// <summary>
            /// Returns the camera's look-at point.
            /// </summary>
            /// <returns>The camera's look-at point.</returns>
            public Vector3 getLookAt()
            {
                return _lookAt;
            }

            /// <summary>
            /// Returns the eye vector. That's a normalized vector pointing from
            /// the look-at point towards the camera's position.
            /// </summary>
            /// <returns>A normalized vector pointing from
            /// the look-at point towards the camera's position.</returns>
            public Vector3 getEyeVector()
            {
                return _eyeVector;
            }

            /// <summary>
            /// Returns the projection matrix of this camera.
            /// </summary>
            /// <returns>This camera's projection matrix.</returns>
            public Matrix getProjectionMatrix()
            {
                return _projection;
            }

            /// <summary>
            /// Returns the view matrix of this camera.
            /// </summary>
            /// <returns>This camera's view matrix.</returns>
            public Matrix getViewMatrix()
            {
                return _view;
            }

            /// <summary>
            /// Retuns the view matrix multiplied by the projection matrix.
            /// </summary>
            /// <returns><see cref="Camera.getViewMatrix"/> * <see cref="getProjectionMatrix"/></returns>
            public Matrix getViewProjectionMatrix()
            {
                return _viewProjection;
            }

            /// <summary>
            /// Caches the view, projection and viewProjection matrices after each
            /// change to data which they depend on (like position for example).
            /// </summary>
            private void updateMatrices()
            {
                Matrix.CreatePerspectiveFieldOfView(_fov, _ratio, _nearPlane, _farPlane,
                                                    out _projection);

                Matrix.CreateLookAt(ref _position, ref _lookAt, ref unitY, out _view);

                _viewProjection = _view * _projection;

                _eyeVector = _position - _lookAt;
                _eyeVector.Normalize();
            }

            /// <summary>
            /// Used to animate camera movement which is caused by user input.
            /// The user can steer the camera with the mouse:
            /// By clicking and holding the left mouse button and then
            /// dragging the mouse, the camera will rotate around its look-at point.
            /// 
            /// The scroll wheel can be used to zoom in and out of the scene. Note:
            /// Scrolling doesn't change the camera's fov, but its distance from its
            /// look-at position.
            /// </summary>
            /// <param name="time">Elapsed time since last frame.</param>
            /// <param name="horRes">Horizontal resolution of the screen.</param>
            /// <param name="vertRes">Vertical resolution of the screen.</param>
            public void update(GameTime time, int horRes, int vertRes)
            {
                if (Mouse.GetState().LeftButton.Equals(ButtonState.Pressed))
                {
                    if (_buttonDown)
                    {
                        _end.X = Mouse.GetState().X;
                        _end.Y = Mouse.GetState().Y;

                        Matrix horRotation = Matrix.CreateFromAxisAngle(Vector3.UnitY,
                                               (_end.X - _start.X) / horRes *
                                               time.ElapsedGameTime.Milliseconds * 0.01f);

                        Matrix vertRotation = Matrix.CreateFromAxisAngle(
                                               Vector3.Cross(((_position - _lookAt) / (_position - _lookAt).Length()), Vector3.UnitY),
                                               -(_end.Y - _start.Y) / vertRes *
                                               time.ElapsedGameTime.Milliseconds * 0.01f);

                        _position -= _lookAt;                       
                        Math3DHelper.mul(ref _position, ref horRotation);                  
                        Math3DHelper.mul(ref _position, ref vertRotation);
                        _position += _lookAt;

                        updateMatrices();
                    }
                    else
                    {
                        _buttonDown = true;
                        _start.X = Mouse.GetState().X;
                        _start.Y = Mouse.GetState().Y;
                    }
                }
                else
                    _buttonDown = false;

                if (_lastScrollWheelValue > Mouse.GetState().ScrollWheelValue)
                {
                    _position *= 1.1f;
                    updateMatrices();
                }
                else if (_lastScrollWheelValue < Mouse.GetState().ScrollWheelValue)
                {
                    _position *= 0.9f;
                    updateMatrices();
                }

                _lastScrollWheelValue = Mouse.GetState().ScrollWheelValue;
            }
        }
    }
}
