using NUnit.Framework;
using SpiffyLibrary;
using SpiffyLibrary.MachineLearning;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Linq;
using Unity.Barracuda;
using Unity.Mathematics;
using UnityEngine;
using Random = Unity.Mathematics.Random;
public class TensorTest
{

  [Test]
  public void ReferenceComputeOps_BasicTest()
  {
    ReferenceComputeOps gpuOps;
    Debug.Log(ComputeShaderSingleton.Instance);
    gpuOps = new ReferenceComputeOps(ComputeShaderSingleton.Instance.referenceKernels);
    int[] shape = new[] { 2, 3, 5, 1 };
    Tensor X = new Tensor(shape, "TestX");
    Tensor W = new Tensor(new TensorShape(15, 7), "TestW");
    X[0] = 3;
    W[0] = 5;
    Debug.Log($"X WxH:{X.flatHeight} {X.flatWidth}");
    Debug.Log($"W WxH:{W.flatHeight} {W.flatWidth}");
    Tensor Y = gpuOps.MatMul(X, false, W, false);
    Debug.Log($"Y WxH:{Y.flatHeight} {Y.flatWidth}");
    X.Dispose();
    W.Dispose();
    Y.Dispose();
    gpuOps.ResetAllocator(false);
    Debug.Assert(true); // Just getting here is good enough
  }
  [Test]
  public void TensorCachingAllocatorTest()
  {
    ReferenceComputeOps gpuOps;
    Debug.Log(ComputeShaderSingleton.Instance);
    gpuOps = new ReferenceComputeOps(ComputeShaderSingleton.Instance.referenceKernels);

    TensorCachingAllocator tca = new TensorCachingAllocator();
    int[] shape = new[] { 2, 3, 5, 1 };
    Tensor X = tca.Alloc(new TensorShape(shape));
    Tensor W = tca.Alloc(new TensorShape(15, 7));
    X[0] = 3;
    W[0] = 5;
    Debug.Log($"X WxH:{X.flatHeight} {X.flatWidth}");
    Debug.Log($"W WxH:{W.flatHeight} {W.flatWidth}");
    Tensor Y = gpuOps.MatMul(X, false, W, false);
    Debug.Log($"Y WxH:{Y.flatHeight} {Y.flatWidth}");
    Debug.Log(X.data.GetType());
    tca.Dispose();
    gpuOps.ResetAllocator(false);
    Debug.Assert(true); // Just getting here is good enough

  }
  [Test]
  public void TensorFlattenTest()
  {
    ReferenceComputeOps gpuOps;
    Debug.Log(ComputeShaderSingleton.Instance);
    gpuOps = new ReferenceComputeOps(ComputeShaderSingleton.Instance.referenceKernels);

    TensorCachingAllocator tca = new TensorCachingAllocator();
    int[] shape = new[] { 2, 2, 3, 4 };
    Tensor X = tca.Alloc(new TensorShape(shape));
    for (int idx = 0; idx < new TensorShape(shape).length; idx++)
    {
      X[idx] = idx;
    }
    Debug.Log($"X WxH:{X.flatHeight} {X.flatWidth}");
    Debug.Log($"{X[0, 0]} {X[1, 0]}");
    Debug.Log($"{X[0, 0, 0, 0]} {X[0, 1, 0, 0]}");
    Debug.Log($"{X[0, 0, 0, 0]} {X[0, 0, 1, 0]}");
    Debug.Log($"{X[0, 0, 0, 0]} {X[0, 0, 0, 1]}");
    tca.Dispose();
    Debug.Assert(true); // Just getting here is good enough

  }

  [Test]
  public void ModelBuilderTest()
  {
    TensorCachingAllocator tca = new TensorCachingAllocator();
    ModelBuilder mb = new ModelBuilder();
    Model.Input inputLayer = mb.Input("Input", new int[] { -1, 1, 1, 1 });
    Layer prevLayer = null;
    prevLayer = mb.Dense(MultiLayerPerception.LayerNames.Hidden, inputLayer, tca.Alloc(new TensorShape(1, 1)), tca.Alloc(new TensorShape(1, 1)));
    prevLayer.weights[0] = 1;
    prevLayer.weights[1] = 1;
    Debug.Log(prevLayer.weights.Length + ": " + string.Join(",", prevLayer.weights));
    for (int i = 0; i < prevLayer.datasets.Length; i++)
    {
      Debug.Log(prevLayer.datasets[i].name + ":" + prevLayer.datasets[i].offset);

    }
    prevLayer = mb.Identity("hiddenAct", prevLayer);
    Debug.Log(prevLayer.weights.Length + ": " + string.Join(",", prevLayer.weights));
    prevLayer = mb.Dense("output", prevLayer, tca.Alloc(new TensorShape(1, 1)), tca.Alloc(new TensorShape(1, 1)));
    prevLayer.weights[0] = 3;
    prevLayer.weights[1] = 5;
    Debug.Log(prevLayer.weights.Length + ": " + string.Join(",", prevLayer.weights));
    prevLayer = mb.Identity("outputActive", prevLayer);
    Debug.Log(prevLayer.weights.Length + ": " + string.Join(",", prevLayer.weights));
    mb.Output(prevLayer);
    IWorker worker = WorkerFactory.CreateWorker(mb.model, WorkerFactory.Device.GPU);
    Tensor input = tca.Alloc(new TensorShape(4, 1, 1, 1));
    for (int i = 0; i < 4; i++)
    {
      input[i] = i;
    }
    IWorker ex = worker.Execute(input);
    ex.FlushSchedule(true);
    Tensor output = ex.PeekOutput();
    for (int i = 0; i < 4; i++)
    {
      Debug.Log($"output[{i}] = {output[i]}");
    }
    tca.Dispose();
    ex.Dispose();
    worker.Dispose();
    Debug.Assert(true); // Just getting here is good enough
  }

  [Test]
  public void MLP_Shape()
  {
    TensorCachingAllocator tca = new TensorCachingAllocator();
    var shape = new MultiLayerPerception.Shape {
      inputSize = 2,
      outputSize = 3,
      hiddenSize = 5
    };
    MultiLayerPerception mlp = new MultiLayerPerception(shape);
    IWorker worker = WorkerFactory.CreateWorker(mlp.model, WorkerFactory.Device.GPU);
    Tensor input = tca.Alloc(new TensorShape(1, 1, 1, shape.inputSize));
    for (int i = 0; i < shape.inputSize; i++)
    {
      input[i] = i;
    }
    IWorker ex = worker.Execute(input);
    ex.FlushSchedule(true);
    Tensor output = ex.PeekOutput();
    for (int i = 0; i < shape.outputSize; i++)
    {
      Debug.Log($"output[{i}] = {output[i]}");
    }
    tca.Dispose();
    ex.Dispose();
    worker.Dispose();
    Debug.Assert(true);
  }
  [Test]
  public void MLP_Calc()
  {
    TensorCachingAllocator tca = new TensorCachingAllocator();
    var shape = new MultiLayerPerception.Shape {
      inputSize = 2,
      outputSize = 3,
      hiddenSize = 2
    };
    MultiLayerPerception mlp = new MultiLayerPerception(shape);
    int layerCnt = 0;
    foreach (Layer layer in mlp.model.layers)
    {
      layerCnt++;
      for (int iWB = 0; iWB < layer.weights.Length; iWB++)
      {
        layer.weights[iWB] = iWB * layerCnt;
      }

      if (layer.datasets.Length == 2)
      {
        Debug.Log($"" +
                  $"{layer.name} " +
                  $"({layer.weights.Length}: W{layer.datasets[0].length} + B{layer.datasets[1].length}): " +
                  $"<{string.Join(", ", layer.weights)}>");
      }
    }

    string HiddenLayer = MultiLayerPerception.LayerNames.Hidden;
    IWorker worker = WorkerFactory.CreateWorker(mlp.model, new string[] { HiddenLayer }, WorkerFactory.Device.GPU);
    Tensor inTensor = tca.Alloc(new TensorShape(1, 1, 1, shape.inputSize));
    for (int i = 0; i < shape.inputSize; i++)
    {
      inTensor[i] = i;
      Debug.Log($"input[{i}] = {inTensor[i]}");
    }
    IWorker ex = worker.Execute(inTensor);
    ex.FlushSchedule(true);


    Tensor hTensor = ex.PeekOutput(HiddenLayer);
    Debug.Assert(hTensor.length == shape.hiddenSize);
    for (int i = 0; i < hTensor.length; i++)
    {
      Debug.Log($"hidden1[{i}] = {hTensor[i]}");
    }
    Tensor output = ex.PeekOutput();
    Debug.Assert(output.length == shape.outputSize);
    for (int i = 0; i < output.length; i++)
    {
      Debug.Log($"output[{i}] = {output[i]}");
    }
    
    for (int iHNode = 0; iHNode < shape.hiddenSize; iHNode++)
    {
      string str = "";
      float sum = 0;
      for (int iINode = 0; iINode < shape.inputSize; iINode++)
      {
        float w = mlp.GetWeight(HiddenLayer, iINode, iHNode);
        str += $"{w} * {inTensor[iINode]} + ";
        sum += w * inTensor[iINode];
      }

      float b = mlp.GetBias(HiddenLayer, iHNode);
      str += $"{b}";
      sum += b;
      str += $"= {hTensor[iHNode]} ({sum})";
      Debug.Assert(Mathf.Approximately(sum,hTensor[iHNode]));
      Debug.Log(str);
    }
    tca.Dispose();
    ex.Dispose();
    worker.Dispose();
    Debug.Assert(true);
  }
  [Test]
  public void GaussianTest() {

    Random _rndu = new Random((uint)UnityEngine.Random.Range(0, int.MaxValue));
    GaussianGenerator _rndn = new GaussianGenerator(_rndu);
    for (int iCnt = 0; iCnt < 100; iCnt++) {
      float val = _rndn.NextFloat1();
      Debug.Log(val);
      Debug.Assert(!float.IsNaN(val));
    }
    
  }
  [Test]
  public static void TestGaussian()
  {

    float t = math.sqrt(-2.0f * math.log(UnityEngine.Random.value)) * math.cos(UnityEngine.Random.value);
    Debug.Log(t);


    Dictionary<int, int> _cnt1 = new Dictionary<int, int>();
    Dictionary<int, int> _cnt2 = new Dictionary<int, int>();
    GaussianGenerator rndg = new GaussianGenerator(new Random((uint)UnityEngine.Random.Range(0, int.MaxValue)));
    for (int i = 0; i < 100000; i++)
    {
      float2 r = rndg.NextFloat2();
      int2 keys = new int2(Mathf.FloorToInt(r[0] * 20), Mathf.FloorToInt(r[1] * 20));
      if (!_cnt1.ContainsKey(keys[0]))
      {
        _cnt1[keys[0]] = 0;
      }

      _cnt1[keys[0]] += 1;
      if (!_cnt2.ContainsKey(keys[1]))
      {
        _cnt2[keys[1]] = 0;
      }

      _cnt2[keys[1]] += 1;
    }

    int width = math.max(_cnt1.Max(kv => kv.Key), _cnt2.Max(kv => kv.Key));
    int height = math.max(_cnt1.Max(kv => kv.Value), _cnt2.Max(kv => kv.Value)) / 10;
    for (int i = -width; i < width; i++)
    {
      {

        float val1 = 0.5f / height;
        if (_cnt1.ContainsKey(i))
        {
          val1 = _cnt1[i] / (float)height;
        }

        Debug.DrawRay(Vector3.right * i / (width / 5f), Vector3.up * val1, Color.blue, 10, false);
      }
      {

        float val2 = 0.5f / height;
        if (_cnt2.ContainsKey(i))
        {
          val2 = _cnt2[i] / (float)height;
        }

        Debug.DrawRay(Vector3.right * i / (width / 5f) + Vector3.forward / 10, Vector3.up * val2, Color.yellow, 10, false);
      }
    }

  }
  [Test]
  public void ReadOnlyCollection() {
    float[] weights = new[] {0f, 1f, 2f, 3f};
    var roWeights = new ReadOnlyCollection<float>(weights.ToArray());
    var metricValues = new Dictionary<string, float>();
    metricValues.Add("key1", 1);
    metricValues.Add("key2", 2);
    metricValues.Add("key3", 3);
    var roMetricValues = new ReadOnlyDictionary<string, float>(metricValues.ToDictionary(kv=>kv.Key,kv=>kv.Value));
    Debug.Log("Pre Modification:");
    Debug.Log(string.Join(",", weights));
    Debug.Log(string.Join(",", roWeights));
    Debug.Log(string.Join(",", metricValues));
    Debug.Log(string.Join(",", roMetricValues));

    weights[2] = -1;
    metricValues["key2"] = -2;

    Debug.Log("Post Modification:");
    Debug.Log(string.Join(",", weights));
    Debug.Log(string.Join(",", roWeights));
    Debug.Log(string.Join(",", metricValues));
    Debug.Log(string.Join(",", roMetricValues));

  }
  

}



