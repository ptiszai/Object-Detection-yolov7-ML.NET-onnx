using System.Drawing;
using System.Diagnostics;
using System.CommandLine;
using Emgu.CV;
using Emgu.CV.Structure;

using Yolov5Net.Scorer.Models;
using Yolov5Net.Scorer;
using static System.Formats.Asn1.AsnWriter;
using System.IO;
using System.Xml.Linq;

namespace Object_Detection_yolov7_ML.NET
{
    internal class Program
    {
        private static string? modelFile = null;
        private static string? imageFile = null;
        private static bool bwr = false;
        private static bool bmp4 = false;
        private static bool bgpu = false;
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

        //-------------------------------------------------------
        static bool SetGpu()
        {
            //scorer you need an option set for GPU
            //https://stackoverflow.com/questions/70369664/guide-to-use-yolo-with-gpu-c-sharp
            return true;
        }

        //-------------------------------------------------------
        internal static async Task ReadFile(string model, string image, int wr, int gpu)
        {
            modelFile = model;
            imageFile = image;
            bwr = (wr>0)?true:false;
            bgpu = (gpu > 0) ? true : false;
        }

        //******************************************************************************
        //* MAIN
        //******************************************************************************
        static async Task<int> Main(string[] args)
        {
            // https://learn.microsoft.com/en-us/dotnet/standard/commandline/get-started-tutorial        
            bool bmp4 = false;
            bool bimg = false;
            string assetsPath = GetAbsolutePath(@"../../../");            
            string? modelFilePath = null;
            string? imageFilePath = null;
            string outputFolder = Path.Combine(assetsPath, "output");           
            VideoWriter? videowriter = null ;
            string windowName = "ML.NET";
            Stopwatch stopwatch = new Stopwatch();
            Queue<Mat>? query = null;
            QueueFrame? queryFrame = null;
            bool queueFrameEnd = false;            

            Console.WriteLine("Hello, Yolov7 + ML.NET!");
            var modeloption = new Option<string?>(
                name: "--model",
                description: "The *.onnx binary model file to read.");

            var imageoption = new Option<string?>(
            name: "--image",
            description: "The *.png,jpg,bmp or *.mp4 video binary file to read.");

            var videoWriteoption = new Option<int>(
            name: "--wr",
            description: "The result_*.png,result_jpg,result_bmp or result_*.mp4 video binary file file to write.",
            getDefaultValue: () => 0);

            var gpuoption = new Option<int>(
            name: "--gpu",
            description: "gpu or only cpu.",
            getDefaultValue: () => 1);

            var rootCommand = new RootCommand("Object-Detection-yolov7-ML.NET app for Windows ML.NET");
            var readCommand = new Command("read", "Read and display the file.")
            {
                modeloption,
                imageoption,               
                videoWriteoption,
                gpuoption
            };

            rootCommand.AddCommand(readCommand);

            readCommand.SetHandler(async (model, image, wr, gpu) =>
            {
                await ReadFile(model!, image, wr, gpu);
            },
            modeloption, imageoption, videoWriteoption, gpuoption);

            int result = rootCommand.InvokeAsync(args).Result;

            if (result > 0)
            {
                Console.WriteLine("ERROR:command parser");
                return 1;
            }
            if (modelFile == null)
            {
                Console.WriteLine("ERROR:modelFile == null");
                return 1;
            }
            modelFilePath = Path.Combine(assetsPath, "models", modelFile);
            if (!File.Exists(modelFilePath))
            {
                Console.WriteLine($"ERROR:modelFilePath not exist:{modelFilePath}");
                return 1;
            }
            if (imageFile == null)
            {
                Console.WriteLine("ERROR:imageFile == null");
                return 1;
            }
            imageFilePath = Path.Combine(assetsPath, "images", imageFile) ;
            if (!File.Exists(imageFilePath))
            {
                Console.WriteLine($"ERROR:imageFilePath not exist:{imageFilePath}");
                return 1;
            }
            FileInfo fi  = new FileInfo(imageFilePath);
            if (fi.Extension == ".mp4")
            {
                bmp4 = true;
            }
            else
            if ((fi.Extension == ".jpg") || (fi.Extension == ".png") || (fi.Extension == ".bmp"))
            {
                bimg = true;
            }
            else 
            {
                Console.WriteLine($"ERROR:image extension s failer:{fi.Extension}");
                return 1;
            }

            if (bgpu)
            {
                if (!SetGpu())
                {
                    Console.WriteLine("ERROR:not foundid GPU, it is failer");
                    return 1;
                }
            }
            YoloScorer<YoloCocoP5Model> scorer = new YoloScorer<YoloCocoP5Model>(modelFilePath); //460 ms
            if (scorer == null)
            {
                Console.WriteLine($"ERROR:model not DOWNLOADED:{modelFilePath}");
                return 1;
            }
            Console.WriteLine($"Model success DOWNLOADED:{modelFilePath}");

            if (!Directory.Exists(outputFolder))
            {
                Directory.CreateDirectory(outputFolder);
            }

            if (bmp4)
            {
                try
                {
                    query = new Queue<Mat>();
                    queryFrame = new QueueFrame(ref query);
                    queryFrame.Ended += (mode) =>
                    {
                        queueFrameEnd = true;
                    };
               
                    string video_fileName = imageFilePath;
                    if (!File.Exists(video_fileName))
                    {
                        Console.WriteLine($"ERROR:video file not founded:{video_fileName}!");
                        return 1;
                    }
                    if (!queryFrame.StartReadFrames(video_fileName))
                    {
                        Console.WriteLine($"ERROR:not started:{video_fileName}!");
                        return 1;
                    }
                    Thread.Sleep(1000);
                    if (queryFrame.IfStartedReadFrames())
                    {
                        if (bwr)
                        {
                            string name = Path.GetFileNameWithoutExtension(video_fileName); 
                            string videopath_out = Path.Combine(outputFolder, $"result_{name}.mp4");
                            //int codec = VideoWriter.Fourcc('m', 'p', '4', 'v');
                            int codec = VideoWriter.Fourcc('R', 'G', 'B', 'A'); // mp3
                            int fps = 10;
                            Size size = queryFrame.GetSize();
                            videowriter = new VideoWriter(videopath_out, codec, fps, size, true);
                        }
                    }
                    else
                    {
                        Console.WriteLine($"ERROR:not started thread for video writing!");
                        return 1;
                    }
                    while (true)
                    {
                        if (query.Count>0)
                        {
                           // Console.WriteLine($"query.Count:{query.Count}");
                            Mat frame = query.Dequeue();
                            stopwatch.Restart();
                            Bitmap image = frame.ToImage<Bgra, Byte>().ToBitmap();
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
                            if (bwr)
                            {
                                videowriter.Write(cvImage);
                                Console.WriteLine("Video success SAVED.");
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
                    Console.WriteLine($"ERROR:video process {ex.Message}");
                    if (queryFrame != null)
                    {
                        queryFrame.StopReadFrames();
                    }
                }
            }
            else if (bimg)
            {
                stopwatch.Start();
                using var image = System.Drawing.Image.FromFile(imageFilePath);
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
                if (bwr)
                {
                    FileInfo fii = new FileInfo(imageFile);
                    string name = fii.Name;
                    image.Save(Path.Combine(outputFolder, $"result_{name}"));
                    Console.WriteLine($"Image success SAVED:result_{name}");
                }

                Mat cvImage = GetMatFromSDImage(image);                    
                CvInvoke.Imshow(windowName, cvImage);                                   
            }
            CvInvoke.WaitKey(0);
            return 0;
        }
    }
}
