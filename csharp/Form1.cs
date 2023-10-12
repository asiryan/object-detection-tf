using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Windows.Forms;

namespace object_detection
{
    public partial class Form1 : Form
    {
        #region Private data

        private readonly string model = @"..\..\..\ssd_mobilenet_v1_coco_2018_01_28\model.onnx";
        private readonly string prototxt = @"..\..\..\coco.prototxt";
        private readonly InferenceSession session;

        #endregion

        #region Form voids

        public Form1()
        {
            InitializeComponent();
            DragDrop += Form1_DragDrop;
            DragEnter += Form1_DragEnter;
            AllowDrop = true;
            session = new InferenceSession(model);
            string file = @"..\..\..\\images\airport.jpg";
            BackgroundImage = new Bitmap(file);
        }

        private void Form1_DragEnter(object sender, DragEventArgs e)
        {
            e.Effect = e.Data.GetDataPresent(DataFormats.FileDrop) ? DragDropEffects.All : DragDropEffects.None;
        }

        private void Form1_DragDrop(object sender, DragEventArgs e)
        {
            Cursor = Cursors.WaitCursor;
            var file = ((string[])e.Data.GetData(DataFormats.FileDrop, true))[0];
            BackgroundImage?.Dispose();
            BackgroundImage = new Bitmap(file, false);
            Process();
            Cursor = Cursors.Default;
        }

        private void Form1_Load(object sender, EventArgs e)
        {
            Process();
        }

        private void Process()
        {
            // params
            var threshold = 0.0f;
            var c = Color.Yellow;
            using var font = new Font("Arial", 22);

            // inference session
            Console.WriteLine("Starting inference session...");
            var tic = Environment.TickCount;
            var inputMeta = session.InputMetadata;
            var name = inputMeta.Keys.ToArray()[0];
            var labels = File.ReadAllLines(prototxt);
            Console.WriteLine("Session started in " + (Environment.TickCount - tic) + " mls.");

            // image
            Console.WriteLine("Creating image tensor...");
            tic = Environment.TickCount;
            var image = (Bitmap)BackgroundImage;
            var width = image.Width;
            var height = image.Height;
            var dimentions = new int[] { 1, height, width, 3 };
            var inputData = Onnx.ToTensor(image);
            Console.WriteLine("Tensor was created in " + (Environment.TickCount - tic) + " mls.");

            // prediction
            Console.WriteLine("Detecting objects...");
            tic = Environment.TickCount;
            var t1 = new DenseTensor<byte>(inputData, dimentions);
            var inputs = new List<NamedOnnxValue>() { NamedOnnxValue.CreateFromTensor(name, t1) };
            var results = session.Run(inputs).ToArray();

            // dump the results
            foreach (var r in results)
            {
                Console.WriteLine(r.Name + "\n");
                Console.WriteLine(r.AsTensor<float>().GetArrayString());
            }
            Console.WriteLine("Detecting was finished in " + (Environment.TickCount - tic) + " mls.");

            // drawing results
            Console.WriteLine("Drawing inference results...");
            tic = Environment.TickCount;
            var detection_boxes = results[0].AsTensor<float>();
            var detection_classes = results[1].AsTensor<float>();
            var detection_scores = results[2].AsTensor<float>();
            var num_detections = results[3].AsTensor<float>()[0];

            using (var g = Graphics.FromImage(image))
            {
                for (int i = 0; i < num_detections; i++)
                {
                    var score = detection_scores[0, i];

                    if (score > threshold)
                    {
                        var label = labels[(int)detection_classes[0, i] - 1];

                        var x = (int)(detection_boxes[0, i, 0] * height);
                        var y = (int)(detection_boxes[0, i, 1] * width);
                        var w = (int)(detection_boxes[0, i, 2] * height);
                        var h = (int)(detection_boxes[0, i, 3] * width);

                        // python rectangle
                        var rectangle = Rectangle.FromLTRB(y, x, h, w);
                        using var brush = new SolidBrush(c);
                        using var pen = new Pen(c) { Width = 3 };
                        g.DrawString(label, font, brush, y, x);
                        g.DrawRectangle(pen, rectangle);
                    }
                }
            }

            BackgroundImage = image;
            Console.WriteLine("Drawing was finished in " + (Environment.TickCount - tic) + " mls.");
        }

        #endregion
    }
}
