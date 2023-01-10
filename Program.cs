using System.Drawing;
using System.Diagnostics;

using Emgu.CV;
using Emgu.CV.Structure;

using Yolov5Net.Scorer.Models;
using Yolov5Net.Scorer;

//https://emgu.com/wiki/index.php/Camera_Capture
namespace Object_Detection_yolov7_ML.NET
{
    internal class Program
    {
        private string assetsPath = "";
        //private bool queueFrameEnd = false;
        /*assetsPath = GetAbsolutePath(@"../../../");
        private string modelFilePath = Path.Combine(assetsPath, "models", "yolov7-tiny-norm.onnx");
        private string imagesFolder = Path.Combine(assetsPath, "images");
        private string outputFolder = Path.Combine(assetsPath,  "output");*/
        //private string windowName = "Your Captcha";
        //private Stopwatch stopwatch = new Stopwatch();
        //private VideoCapture videoCapture;

        //-------------------------------------------------------
        private static string GetAbsolutePath(string relativePath)
        {
            FileInfo _dataRoot = new FileInfo(typeof(Program).Assembly.Location);
            string assemblyFolderPath = _dataRoot.Directory!.FullName;

            string fullPath = Path.Combine(assemblyFolderPath, relativePath);

            return fullPath;
        }

        //-------------------------------------------------------
        static private Mat GetMatFromSDImage(System.Drawing.Image image)
        {
            int stride = 0;
            Bitmap bmp = new Bitmap(image);

            System.Drawing.Rectangle rect = new System.Drawing.Rectangle(0, 0, bmp.Width, bmp.Height);
            System.Drawing.Imaging.BitmapData bmpData = bmp.LockBits(rect, System.Drawing.Imaging.ImageLockMode.ReadWrite, bmp.PixelFormat);

            System.Drawing.Imaging.PixelFormat pf = bmp.PixelFormat;
            if (pf == System.Drawing.Imaging.PixelFormat.Format32bppArgb)
            {
                stride = bmp.Width * 4;
            }
            else
            {
                stride = bmp.Width * 3;
            }

            Image<Bgra, byte> cvImage = new Image<Bgra, byte>(bmp.Width, bmp.Height, stride, (IntPtr)bmpData.Scan0);

            bmp.UnlockBits(bmpData);

            return cvImage.Mat;
        }

        //******************************************************************************
        //* MAIN
        //******************************************************************************
        static void Main(string[] args)
        {
            bool wr = true;
            bool mp4 = true;
            bool img = false;
            string assetsPath = GetAbsolutePath(@"../../../");
            string modelFilePath = Path.Combine(assetsPath, "models", "yolov7-tiny-norm.onnx");
            string imagesFolder = Path.Combine(assetsPath, "images");
            string outputFolder = Path.Combine(assetsPath, "output");           
            VideoWriter? videowriter = null ;
            string windowName = "Your Captcha";
            Stopwatch stopwatch = new Stopwatch();
            Queue<Mat>? query = null;
            QueueFrame? queryFrame = null;
            bool queueFrameEnd = false;

            Console.WriteLine("Hello, Yolov7 + ML.NET!");                               

            if (!Directory.Exists(outputFolder))
            {
                Directory.CreateDirectory(outputFolder);
            }


            var imgs = Directory.GetFiles(imagesFolder).Where(filePath => Path.GetExtension(filePath) == ".jpg");
            //stopwatch.Start();
            YoloScorer<YoloCocoP5Model> scorer = new YoloScorer<YoloCocoP5Model>(modelFilePath); //460 ms
            //stopwatch.Stop();
            //long elapsed_time = stopwatch.ElapsedMilliseconds;
            //Console.WriteLine($"Inference time {elapsed_time}!");

            if (mp4)
            {
                try
                {
                    query = new Queue<Mat>();
                    queryFrame = new QueueFrame(ref query);
                    queryFrame.Ended += (mode) =>
                    {
                        queueFrameEnd = true;
                    };
               
                    string video_fileName = imagesFolder + "/cars.mp4";
                    if (!File.Exists(video_fileName))
                    {
                        Console.WriteLine($"ERROR:video file not founded:{video_fileName}!");
                        return;
                    }
                    if (!queryFrame.StartReadFrames(video_fileName))
                    {
                        Console.WriteLine($"ERROR:not started:{video_fileName}!");
                        return;
                    }
                    Thread.Sleep(1000);
                    if (queryFrame.IfStartedReadFrames())
                    {
                        if (wr)
                        {
                            FileInfo fi = new FileInfo("cars.mp4");
                            string name = fi.Name;
                            string videopath_out = Path.Combine(outputFolder, $"result_{name}");
                            int codec = VideoWriter.Fourcc('m', 'p', '4', 'v');
                            int fps = 10;
                            Size size = queryFrame.GetSize();
                            videowriter = new VideoWriter(videopath_out, codec, fps, size, true);
                        }
                    }
                    else
                    {
                        Console.WriteLine($"ERROR:not started thread for video writing!");
                        return;
                    }
                    while (true)
                    {
                        if (query.Count>0)
                        {
                            Console.WriteLine($"query.Count:{query.Count}");
                            Mat frame = query.Dequeue();
                            stopwatch.Restart();
                            Bitmap image = frame.ToImage<Bgr, Byte>().ToBitmap();
                            List<YoloPrediction> predictions = scorer.Predict(image);                            
                            using var graphics = Graphics.FromImage(image);
                            stopwatch.Stop();
                            long elapsed_time = stopwatch.ElapsedMilliseconds;                            
                            string label = $"Inference time {elapsed_time} ms!"; // 450-500 ms
                            Console.WriteLine(label);
                            queryFrame.SetSleepMs(elapsed_time);
                            //stopwatch.Restart();
                            foreach (var prediction in predictions)
                            {
                                double score = Math.Round(prediction.Score, 2);
                                graphics.DrawRectangles(new Pen(prediction.Label.Color, 3), new[] { prediction.Rectangle });
                                var (x, y) = (prediction.Rectangle.X - 3, prediction.Rectangle.Y - 23);
                                graphics.DrawString($"{prediction.Label.Name} ({score})", new Font("Consolas", 16, GraphicsUnit.Pixel), new SolidBrush(prediction.Label.Color), new PointF(x, y));
                            }
                            graphics.DrawString(label, new Font("Consolas", 16, GraphicsUnit.Pixel), new SolidBrush(Color.White), new PointF(0, 15));
                            Mat cvImage = GetMatFromSDImage(image);
                            //stopwatch.Stop();
                           // Console.WriteLine($"predictiont ms:{stopwatch.ElapsedMilliseconds}"); // 10-15 ms
                            if (wr)
                            {
                                videowriter.Write(frame); !!!!!!!!!!!!!!
                            }
                            //stopwatch.Stop();
                           // Console.WriteLine($"predictiont+wr ms:{stopwatch.ElapsedMilliseconds}"); // 19-25 ms
                            CvInvoke.Imshow(windowName, cvImage);
                        }
                        if (queueFrameEnd && query.Count < 1)
                        {
                            Console.WriteLine("frames process END!");
                            break;
                        }
                        if (CvInvoke.PollKey()>0)
                        {
                            queryFrame.StopReadFrames();
                            break;
                        }
                        Thread.Sleep(10);                       
                    }
                    if (videowriter != null)
                    {
                        videowriter.Dispose();
                    }
                }
                catch (Exception ex)
                {
                    //Program.mutex.ReleaseMutex();
                    Console.WriteLine($"ERROR:video process {ex.Message}");
                    if (queryFrame != null)
                    {
                        queryFrame.StopReadFrames();
                    }
                }
            }
            else if (img)
            {          
                foreach (var imgsFile in imgs)
                {
                    stopwatch.Start();
                    using var image = System.Drawing.Image.FromFile(imgsFile);
                    //stopwatch.Start();
                    List<YoloPrediction> predictions = scorer.Predict(image);
                    using var graphics = Graphics.FromImage(image);
                    stopwatch.Stop();
                    long elapsed_time = stopwatch.ElapsedMilliseconds;
                    string label = $"Inference time {elapsed_time} ms!";
                    //Console.WriteLine(label);                
                    foreach (var prediction in predictions)
                    {
                        double score = Math.Round(prediction.Score, 2);
                        graphics.DrawRectangles(new Pen(prediction.Label.Color, 3), new[] { prediction.Rectangle });
                        var (x, y) = (prediction.Rectangle.X - 3, prediction.Rectangle.Y - 23);
                        graphics.DrawString($"{prediction.Label.Name} ({score})", new Font("Consolas", 16, GraphicsUnit.Pixel), new SolidBrush(prediction.Label.Color), new PointF(x, y));
                    }
                    graphics.DrawString(label, new Font("Consolas", 16, GraphicsUnit.Pixel), new SolidBrush(Color.White), new PointF(0, 15));
                    if (wr)
                    {
                        FileInfo fi = new FileInfo(imgsFile);
                        string name = fi.Name;
                        image.Save(Path.Combine(outputFolder, $"redsult_{name}"));
                    }

                    Mat cvImage = GetMatFromSDImage(image);

                    //stopwatch.Stop();
                    //long elapsed_time = stopwatch.ElapsedMilliseconds;
                    // Console.WriteLine($"Inference time {elapsed_time}!");
                    CvInvoke.Imshow(windowName, cvImage);
                    //CvInvoke.WaitKey(0);
                }
            }
            CvInvoke.WaitKey(0);
        }
    }
}