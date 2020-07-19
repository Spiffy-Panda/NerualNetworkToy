using NUnit.Framework;
using SpiffyLibrary;
using SpiffyLibrary.MachineLearning;
using Unity.Barracuda;
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
    prevLayer = mb.Dense(MLP_Tensor.LayerNames.Hidden, inputLayer, tca.Alloc(new TensorShape(1, 1)), tca.Alloc(new TensorShape(1, 1)));
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
    Debug.Assert(true); // Just getting here is good enough
  }

  [Test]
  public void MLP_Shape()
  {
    TensorCachingAllocator tca = new TensorCachingAllocator();
    int inputSize = 2;
    int outputSize = 3;
    MLP_Tensor mlp = new MLP_Tensor(inputSize, outputSize, hiddenSize: 5);
    IWorker worker = WorkerFactory.CreateWorker(mlp.model, WorkerFactory.Device.GPU);
    Tensor input = tca.Alloc(new TensorShape(1, 1, 1, inputSize));
    for (int i = 0; i < inputSize; i++)
    {
      input[i] = i;
    }
    IWorker ex = worker.Execute(input);
    ex.FlushSchedule(true);
    Tensor output = ex.PeekOutput();
    for (int i = 0; i < outputSize; i++)
    {
      Debug.Log($"output[{i}] = {output[i]}");
    }
    tca.Dispose();
    ex.Dispose();
    Debug.Assert(true);
  }
  [Test]
  public void MLP_Calc()
  {
    TensorCachingAllocator tca = new TensorCachingAllocator();
    int inputSize = 2;
    int outputSize = 3;
    int hiddenSize = 2;
    MLP_Tensor mlp = new MLP_Tensor(inputSize, outputSize, hiddenSize);
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

    string HiddenLayer = MLP_Tensor.LayerNames.Hidden;
    IWorker worker = WorkerFactory.CreateWorker(mlp.model, new string[] { HiddenLayer }, WorkerFactory.Device.GPU);
    Tensor inTensor = tca.Alloc(new TensorShape(1, 1, 1, inputSize));
    for (int i = 0; i < inputSize; i++)
    {
      inTensor[i] = i;
      Debug.Log($"input[{i}] = {inTensor[i]}");
    }
    IWorker ex = worker.Execute(inTensor);
    ex.FlushSchedule(true);


    Tensor hTensor = ex.PeekOutput(HiddenLayer);
    Debug.Assert(hTensor.length == hiddenSize);
    for (int i = 0; i < hTensor.length; i++)
    {
      Debug.Log($"hidden1[{i}] = {hTensor[i]}");
    }
    Tensor output = ex.PeekOutput();
    Debug.Assert(output.length == outputSize);
    for (int i = 0; i < output.length; i++)
    {
      Debug.Log($"output[{i}] = {output[i]}");
    }
    
    for (int iHNode = 0; iHNode < hiddenSize; iHNode++)
    {
      string str = "";
      float sum = 0;
      for (int iINode = 0; iINode < inputSize; iINode++)
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
}



