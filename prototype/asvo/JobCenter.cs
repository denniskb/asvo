using System.Threading;
using asvo.world3D;
using asvo.renderers;
using asvo.datastructures;

using System.Collections.Generic;

namespace asvo
{
    namespace multithreading
    {
        /// <summary>
        /// Represents a job center that creates and manages workers and
        /// assignes jobs to them. Not thread-safe!, should only be used from a
        /// single thread environment.
        /// </summary>
        internal class JobCenter
        {
            private static Worker[] workers = null;

            /// <summary>
            /// Creates <paramref name="workerCount"/> many workers.
            /// Can only be called once.
            /// </summary>
            /// <param name="workerCount">The number of workers to create.</param>
            public static void initializeWorkers(int workerCount)
            {
                if (workers == null)
                {
                    workers = new Worker[workerCount];
                    for (int i = 0; i < workerCount; ++i)
                    {
                        workers[i] = new Worker();
                        workers[i].start();
                    }
                }
            }

            /// <summary>
            /// Assigns <paramref name="job"/> to all existing workers.
            /// </summary>
            /// <param name="job"></param>
            public static void assignJob(IJob job)
            {
                if (workers != null)
                    for (int i = 0; i < workers.Length; ++i)
                        workers[i].assignJob(job);
            }

            /// <summary>
            /// Waits, until all workers are finished with their current jobs.
            /// </summary>
            public static void wait()
            {
                if (workers != null)
                    for (int i = 0; i < workers.Length; ++i)
                        workers[i].join();
            }

            /// <summary>
            /// Returns the number of existing workers.
            /// </summary>
            /// <returns>The number of existing workers.</returns>
            public static int getWorkerCount()
            {
                return workers.Length;
            }

            /// <summary>
            /// Represents a worker. A worker does work. One can assign as many jobs to him,
            /// as one likes. They all get stored in a queue, which the worker works on.
            /// 
            /// You can rest while he's doing your work and he will wake you up, if he's finished.
            /// If there are no jobs to do for the moment, the worker rests, releasing system
            /// resources.
            /// 
            /// A worker is implemented as a thread, so creating multiple workers and letting them
            /// work on independent jobs grants a performance benefit on multi processor systems.
            /// Or you just can realize asnychronous operations (on a single processer system).
            /// </summary>
            private class Worker
            {
                public readonly int id;
                private static int _currentId = 0;
                private static readonly object idLock = new object();
                private readonly object queueLock = new object();

                private readonly Thread _thread;
                private readonly ManualResetEvent _newJob;
                private readonly ManualResetEvent _done;
                private Queue<IJob> _jobs;

                /// <summary>
                /// Creates a new worker and assigns it a unique id.
                /// Workers should be reused throuout the execution time rather than
                /// being destroyed and recreated.
                /// </summary>
                public Worker()
                {
                    lock (idLock)
                    {
                        id = _currentId++;
                    }
                    _jobs = new Queue<IJob>();
                    _thread = new Thread(doWork);
                    _thread.IsBackground = true;
                    _newJob = new ManualResetEvent(false);
                    _done = new ManualResetEvent(false);
                }

                /// <summary>
                /// Returns the count of all created workers since application start.
                /// </summary>
                /// <returns>The count of all workers ever created.</returns>
                public static int getWorkerCount()
                {
                    return _currentId;
                }

                /// <summary>
                /// Starts this worker's every day work described in the class
                /// summary.
                /// </summary>
                public void start()
                {
                    _thread.Start();
                }

                /// <summary>
                /// Puts the calling thread to sleep, until the worker has finished all
                /// his work.
                /// </summary>
                public void join()
                {
                    _done.WaitOne();
                }

                /// <summary>
                /// Assigns this worker a job which gets inserted into the worker's job
                /// queue.
                /// </summary>
                /// <param name="job">The job to be exectued by the worker.</param>
                public void assignJob(IJob job)
                {
                    lock (queueLock)
                    {
                        _jobs.Enqueue(job);
                    }
                    _done.Reset();
                    _newJob.Set();
                }               

                /// <summary>
                /// (Fetch job -> execute -> sleep)-loop described in the class summary.
                /// </summary>
                private void doWork()
                {
                    while (true)
                    {
                        IJob job = null;
                        lock (queueLock)
                        {
                            if (_jobs.Count > 0)
                                job = _jobs.Dequeue();
                        }
                        if (job != null)
                        {
                            job.execute(id);
                        }
                        else
                        {
                            _newJob.Reset();
                            _done.Set();
                            _newJob.WaitOne();
                        }
                    }
                }
            }
        }

        /// <summary>
        /// Represents a Job that can be executed by a worker.
        /// </summary>
        internal interface IJob
        {
            /// <summary>
            /// Executes the job.
            /// </summary>
            /// <param name="threadId">The id of the executing thread.</param>
            void execute(int threadId);
        }

        /// <summary>
        /// A job that renders an object.
        /// </summary>
        internal struct RenderObjectJob : IJob
        {
            private readonly Object3D _object3D;           
            private readonly Camera _camera;
            private readonly Rasterizer _rasterizer;

            /// <summary>
            /// Creates a render object job. When executed it will render <paramref name="object3D"/> from
            /// <paramref name="cam"/>'s point of view with <paramref name="rasterizer"/>.
            /// </summary>
            /// <param name="object3D">The 3d object to be rendered.</param>
            /// <param name="camera">The camera from which object3d shall be rendered.</param>
            /// <param name="rasterizer">The rasterizer to use for the
            /// rendering process.</param>
            public RenderObjectJob(Object3D object3D, Camera camera, Rasterizer rasterizer)
            {
                _object3D = object3D;               
                _camera = camera;
                _rasterizer = rasterizer;
            }

            /// <summary>
            /// <seealso cref="IJob.execute"/>
            /// </summary> 
            public void execute(int threadId)
            {
                _rasterizer.render(_object3D, _camera, threadId);
            }
        }

        /// <summary>
        /// A job that merges a shared depth buffer.
        /// </summary>
        internal struct MergeDepthBufferJob : IJob
        {
            private readonly Rasterizer _rasterizer;

            /// <summary>
            /// Creates a new merge depth buffer job. When executed,
            /// it will merge the depth buffer of the given rasterizer.
            /// </summary>
            /// <param name="rasterizer">The rasterizer whoose depth buffer shall
            /// be merged.</param>
            public MergeDepthBufferJob(Rasterizer rasterizer)
            {
                _rasterizer = rasterizer;
            }

            /// <summary>
            /// <seealso cref="IJob.execute"/>
            /// </summary>
            public void execute(int threadId)
            {
                _rasterizer._depthBuffer.merge(threadId, _rasterizer._colorBuffer);
            }
        }
    }
}
