using System.Drawing;
using System.IO;
using System.Diagnostics;

using Emgu.CV;
using Emgu.CV.Structure;
using Emgu.CV.CvEnum;
using Yolov5Net.Scorer.Models;
using Yolov5Net.Scorer;
using System;
using System.Reflection.Emit;
using static System.Net.Mime.MediaTypeNames;
using static System.Formats.Asn1.AsnWriter;


namespace Object_Detection_yolov7_ML.NET
{    
    internal class Program
    {       
        static readonly string assetsPath = GetAbsolutePath(@"../../../");
        static readonly string modelFilePath = Path.Combine(assetsPath, "models", "yolov7-tiny-norm.onnx");
        static readonly string imagesFolder = Path.Combine(assetsPath, "images");
        static readonly string outputFolder = Path.Combine(assetsPath,  "output");
        static String windowName = "Your Captcha";
        static Stopwatch stopwatch = new Stopwatch();

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

            //Console.WriteLine("Hello, World!");

            if (!Directory.Exists(outputFolder))
            {
                Directory.CreateDirectory(outputFolder);
            }
            var imgs = Directory.GetFiles(imagesFolder).Where(filePath => Path.GetExtension(filePath) == ".jpg");
            //stopwatch.Start();
            using var scorer = new YoloScorer<YoloCocoP5Model>(modelFilePath); //460 ms
            //stopwatch.Stop();
            //long elapsed_time = stopwatch.ElapsedMilliseconds;
            //Console.WriteLine($"Inference time {elapsed_time}!");


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
                    graphics.DrawString($"{prediction.Label.Name} ({score})",new Font("Consolas", 16, GraphicsUnit.Pixel), new SolidBrush(prediction.Label.Color),new PointF(x, y));
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
                CvInvoke.WaitKey(0);
            }

        }
    }
}