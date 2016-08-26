using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.Xna.Framework;
using Microsoft.Xna.Framework.Audio;
using Microsoft.Xna.Framework.Content;
using Microsoft.Xna.Framework.GamerServices;
using Microsoft.Xna.Framework.Graphics;
using Microsoft.Xna.Framework.Input;
using Microsoft.Xna.Framework.Media;
using Microsoft.Xna.Framework.Net;
using Microsoft.Xna.Framework.Storage;

using asvo.datastructures;
using asvo.renderers;
using asvo.world3D;
using asvo.multithreading;

namespace asvo
{
    /// <summary>
    /// This is the main type for your game
    /// </summary>
    public class Game1 : Microsoft.Xna.Framework.Game
    {
        GraphicsDeviceManager graphics;       

        Rasterizer testRasterizer;
        Object3D testObj;
        TriangleMesh testMesh;
        
        Camera cam;

        private KeyboardState ks;

        /// <summary>
        /// Application for testing purposes.
        /// </summary>
        public Game1()
        {
            // Set necessary values.
            graphics = new GraphicsDeviceManager(this);
            Content.RootDirectory = "Content";

            graphics.PreferredBackBufferWidth = 512;
            graphics.PreferredBackBufferHeight = 512;           

            this.IsFixedTimeStep = false;
            graphics.SynchronizeWithVerticalRetrace = false;
    
            // Create as many threads as you have cores.
            // Used for rendering - greatly enhances performance.
            JobCenter.initializeWorkers(6);            

            // Initialize the software-rasterizer.
            testRasterizer = new Rasterizer(
                graphics.PreferredBackBufferWidth,
                graphics.PreferredBackBufferHeight
            );            
    
            // Initialize the virtual camera.
            cam = new Camera(10, 200, new Vector3(0, 25, 80), new Vector3(0, 0, 0),
                             ((float)graphics.PreferredBackBufferWidth) /
                             graphics.PreferredBackBufferHeight);
        }

        /// <summary>
        /// Allows the game to perform any initialization it needs to before starting to run.
        /// This is where it can query for any required services and load any non-graphic
        /// related content.  Calling base.Initialize will enumerate through any components
        /// and initialize them as well.
        /// </summary>
        protected override void Initialize()
        {
            base.Initialize();           
        }

        /// <summary>
        /// LoadContent will be called once per game and is the place to load
        /// all of your content.
        /// </summary>
        protected override void LoadContent()
        {
            // call LoadContent on all used components that implement it.
            testRasterizer.loadContent(GraphicsDevice, Content);

            // Load a triangle mesh.
            testMesh = new TriangleMesh("imrod_walk_high", Content);

            // Create a BFSOctree out of that mesh.
            testObj = new Object3D(testMesh.toBFSOctree(7), false);
            // Store the octree into a binary file.
            //testObj.getData().export("imrod.asvo");
            testObj.rotate(Vector3.UnitX, MathHelper.PiOver2);
            
            System.GC.Collect();
            ks = Keyboard.GetState();
        }

        /// <summary>
        /// UnloadContent will be called once per game and is the place to unload
        /// all content.
        /// </summary>
        protected override void UnloadContent()
        {
            // TODO: Unload any non ContentManager content here
        }

        /// <summary>
        /// Allows the game to run logic such as updating the world,
        /// checking for collisions, gathering input, and playing audio.
        /// </summary>
        /// <param name="gameTime">Provides a snapshot of timing values.</param>
        protected override void Update(GameTime gameTime)
        {
            // Allows the game to exit
            if (GamePad.GetState(PlayerIndex.One).Buttons.Back == ButtonState.Pressed)
                this.Exit();

            // Update camera movement.
            cam.update(gameTime,
                       graphics.PreferredBackBufferWidth,
                       graphics.PreferredBackBufferHeight);          

            base.Update(gameTime);
        }

        /// <summary>
        /// This is called when the game should draw itself.
        /// </summary>
        /// <param name="gameTime">Provides a snapshot of timing values.</param>
        protected override void Draw(GameTime gameTime)
        {
            GraphicsDevice.Clear(Color.Black);

            // Control skinning animations with the keyboard: Pressing "down" advances the
            // animation by 1 frame.
            //if (ks.IsKeyUp(Keys.Down) && (ks = Keyboard.GetState()).IsKeyDown(Keys.Down))
                testObj.frame = (int)((testObj.frame + 1) % testObj.getData().frameCount);
            //else
                //ks = Keyboard.GetState();

            // Render the octree using the JobCenter
            JobCenter.assignJob(new RenderObjectJob(testObj, cam, testRasterizer));
            JobCenter.wait();
            JobCenter.assignJob(new MergeDepthBufferJob(testRasterizer));
            JobCenter.wait();
            
            // Draw the final image on the screen.
            testRasterizer.draw(GraphicsDevice);            
        }
    }
}
