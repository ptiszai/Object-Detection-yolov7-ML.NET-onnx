using System.Drawing;
using Emgu.CV;

namespace Object_Detection_yolov7_ML.NET
{
    public class QueueFrame : IDisposable
    {
        #region public eventHandlers
        public Action<bool> Ended;       
        #endregion

        #region private variables 
        private Queue<Mat> queryFrame;
        private Thread? thread;
        private VideoCapture? videoCapture;        
        private long sleep_ms = 200;
        private bool run = false;
        private int frameWidth;
        private int frameHeight;
        private int totFrames;
        private int fcount = 1;
        #endregion

        #region public functions
        //-----------------------
        public QueueFrame(ref Queue<Mat> queryFrame_a)
        {
            queryFrame = queryFrame_a;
            thread = new Thread(executor);
        }

        //-----------------------
        public bool StartReadFrames(string filename_a)
        {
            videoCapture = new VideoCapture(filename_a);
            if ((videoCapture == null) || (!videoCapture.Grab()))
            {
                Console.WriteLine("ERROR: start frame grab!");
                return false;
            }
            queryFrame.Clear();
            run = true;
            thread.Start();
            return true;
        }

        //-----------------------
        public bool IfStartedReadFrames()
        {
            if (thread.IsAlive && run && fcount > 1)
            {
                return true;
            }
            return false;
        }

        //-----------------------
        public void StopReadFrames()
        {
            if (thread.IsAlive)
            {
                run = false;
            //    thread.Abort();
                queryFrame.Clear();
            }
        }

        public void SetSleepMs(long sleep_ms_a)
        {
            sleep_ms = sleep_ms_a;
        }

        //-----------------------
        public void Dispose()
        {
            if (videoCapture != null)
            {
                videoCapture.Dispose();
            }
            //videowriter.Dispose();
            if (thread.IsAlive)
            {
                run = false;
                thread.Abort();
            }
        }

        //-----------------------
        public Size GetSize()
        {
            Size result = new Size(frameWidth, frameHeight);
            return result;
        }
        #endregion
        #region private functions       
        //-----------------------
        private void executor()
        {
            totFrames = (int)videoCapture.Get(Emgu.CV.CvEnum.CapProp.FrameCount);
            frameWidth = (int)videoCapture.Get(Emgu.CV.CvEnum.CapProp.FrameWidth);
            frameHeight = (int)videoCapture.Get(Emgu.CV.CvEnum.CapProp.FrameHeight);           

            try
            {
                while (run)
                {
                    if (videoCapture.IsOpened)
                    {                    
                        Mat frame = videoCapture.QueryFrame();
                        //videoCapture.Read(frame);
                        if (frame == null)
                        {
                            run = false;
                            if (totFrames >= fcount)
                            {
                                Console.WriteLine($"Video read frames END, queryFrame.Count:{queryFrame.Count}");                             
                                Ended?.Invoke(true);
                            }
                            else
                            {
                                Console.WriteLine("ERROR:frame grab false!");                               
                                queryFrame.Clear();
                            }
                            break;
                        }
                        queryFrame.Enqueue(frame);
                        fcount++;
                        Console.WriteLine($"frame number {fcount}");                   
                    }
                    Thread.Sleep((int)sleep_ms);                  
                }
            }
            catch (ThreadAbortException ex)
            {
                //Program.mutex.ReleaseMutex();
                Console.WriteLine($"ERROR:QueueFrame:thread.executor {ex.Message}");
                run = false;
            }
        }
        #endregion
    }
}
