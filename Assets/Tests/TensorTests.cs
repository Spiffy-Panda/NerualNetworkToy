﻿using NUnit.Framework;
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
    X.Dispose();
    W.Dispose();
    Y.Dispose();
    Debug.Assert(true); // Just getting here is good enough
  }

  [Test]
  public void ModelBuilderTest()
  {

    ModelBuilder mb = new ModelBuilder();
    Model.Input inputLayer = mb.Input("Input", new int[] { -1, 1, 1, 4 });
    Layer hiddenDenseLayer = mb.Dense("hidden1", inputLayer, new Tensor(new TensorShape(4, 4)), new Tensor(new TensorShape(1, 4)));
    Layer hiddenActiveLayer = mb.Relu("hiddenAct", hiddenDenseLayer);
    Layer outputDenseLayer = mb.Dense("output", hiddenActiveLayer, new Tensor(new TensorShape(4, 4)), new Tensor(new TensorShape(1, 4)));
    Layer outputActiveLayer = mb.Relu("outputActive", outputDenseLayer);
    mb.Output(outputActiveLayer);
    IWorker worker = WorkerFactory.CreateWorker(mb.model, WorkerFactory.Device.GPU);
    IWorker ex = worker.Execute(new Tensor(new TensorShape(1, 1, 1, 4)));
    ex.FlushSchedule(true);
    Debug.Log(ex.PeekOutput());
    Debug.Assert(true); // Just getting here is good enough
  }

}


