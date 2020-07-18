using System.Linq.Expressions;
using NUnit.Framework;
using Unity.Barracuda;
using UnityEngine;

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
  public void ModelBuilderTest()
  {
    TensorCachingAllocator tca = new TensorCachingAllocator();
    ModelBuilder mb = new ModelBuilder();
    Model.Input inputLayer = mb.Input("Input", new int[] { -1, 1, 1, 1 });
    Layer prevLayer = null;
    prevLayer = mb.Dense("hidden1", inputLayer, tca.Alloc(new TensorShape(1, 1)), tca.Alloc(new TensorShape(1, 1)));
    prevLayer.weights[0] = 1;
    prevLayer.weights[1] = 1;
    Debug.Log(prevLayer.weights.Length + ": "+string.Join(",", prevLayer.weights));
    for (int i = 0; i < prevLayer.datasets.Length; i++) {
      Debug.Log(prevLayer.datasets[i].name +":"+ prevLayer.datasets[i].offset);
      
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
    var input = tca.Alloc(new TensorShape(4, 1, 1, 1));
    for (int i = 0; i < 4; i++) {
      input[i] = i;
    }
    IWorker ex = worker.Execute(input);
    ex.FlushSchedule(true);
    Tensor output = ex.PeekOutput();
    for (int i = 0; i < 4; i++) {
      Debug.Log($"output[i] = {output[i]}");
    }
    tca.Dispose();
    ex.Dispose();
    Debug.Assert(true); // Just getting here is good enough
  }

}



