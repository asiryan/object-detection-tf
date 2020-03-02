using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Windows.Forms;

namespace ssd_mobilenet_v1_coco_2018_01_28_test
{
    public partial class Form1 : Form
    {
        #region Private data
        string model = @"C:\ssd2onnx\ssd_mobilenet_v1_coco_2018_01_28\model.onnx";
        string file = @"C:\ssd2onnx\images\airport.jpg";
        string[] labels = new string[] { "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light" };
        Color c = Color.Yellow;
        int tic;
        #endregion

        #region Form voids
        public Form1()
        {
            InitializeComponent();
        }
        private void Form1_Load(object sender, EventArgs e)
        {
            // inference session
            Console.WriteLine("Starting inference session");
            tic = Environment.TickCount;
            var session = new InferenceSession(model);
            var inputMeta = session.InputMetadata;
            string name = inputMeta.Keys.ToArray()[0]; // "image_tensor:0"
            Console.WriteLine("Session started in " + (Environment.TickCount - tic) + " mls.");

            // image
            Console.WriteLine("Creating image tensor");
            tic = Environment.TickCount;
            Bitmap image = new Bitmap(file, false);
            int width = image.Width;
            int height = image.Height;
            int[] dimentions = new int[] { 1, height, width, 3 };
            byte[] inputData = Onnx.ToTensor(image);
            Console.WriteLine("Tensor was created in " + (Environment.TickCount - tic) + " mls.");

            // prediction
            Console.WriteLine("Detecting objects");
            tic = Environment.TickCount;
            Tensor<byte> t1 = new DenseTensor<byte>(inputData, dimentions);
            var inputs = new List<NamedOnnxValue>() { NamedOnnxValue.CreateFromTensor<byte>(name, t1) };
            var results = session.Run(inputs).ToArray();

            // dump the results
            foreach (var r in results)
            {
                Console.WriteLine(r.Name + "\n");
                Console.WriteLine(r.AsTensor<float>().GetArrayString());
            }
            Console.WriteLine("Detecting was finished in " + (Environment.TickCount - tic) + " mls.");

            // drawing results
            Console.WriteLine("Drawing");
            tic = Environment.TickCount;
            var detection_boxes = results[0].AsTensor<float>();
            var detection_classes = results[1].AsTensor<float>();
            var detection_scores = results[2].AsTensor<float>();
            var num_detections = results[3].AsTensor<float>()[0];

            using (Graphics g = Graphics.FromImage(image))
            {
                for (int i = 0; i < num_detections; i++)
                {
                    float score = detection_scores[0, i];
                    string label = labels[(int)detection_classes[0, i] - 1];

                    int x = (int)(detection_boxes[0, i, 0] * height);
                    int y = (int)(detection_boxes[0, i, 1] * width);
                    int w = (int)(detection_boxes[0, i, 2] * height);
                    int h = (int)(detection_boxes[0, i, 3] * width);

                    // python rectangle
                    Rectangle rect = new Rectangle(y, x, h - y, w - x);
                    g.DrawString(label, new Font("Arial", 22), new SolidBrush(c), y, x);
                    g.DrawRectangle(new Pen(c) { Width = 3 }, rect);
                }
            }

            pictureBox1.Image = image;
            Console.WriteLine("Drawing was finished in " + (Environment.TickCount - tic) + " mls.");
        }
        #endregion
    }
}
