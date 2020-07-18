using System;
using System.Collections;
using System.Collections.Generic;
using Unity.Barracuda;
using UnityEngine;
using Unity.Mathematics;
using UnityEngine.Rendering.VirtualTexturing.Procedural;
using Random = Unity.Mathematics.Random;

public abstract class MLP
{
  protected Func<float4, float4> Activation { get; private set; }
  

  public abstract void Copy(MLP other);
  public abstract void Clear();

  public abstract void Mutate(ref GaussianGenerator rnd, float learningRate);

  public abstract float4 Execute(float4 input);

  private static float4 relu(float4 x) { return math.max(float4.zero, x); }
  public abstract float GetWeight(int layer, int srcNode, int dstNode);
}

public class MLP_Tensor : MLP { 
  public const int kInputSize = 4;
  public const int kHiddenSize = 4;
  public const int kOutputSize = 4;
  public SharedArrayTensorData data;
  public Tensor weights_o_h1;
  public Tensor weights_h1_i;
  public Tensor bais_o;
  public Tensor bais_h1;
  public Model model;

  public Layer.FusedActivation activation = Layer.FusedActivation.Sigmoid;

  public MLP_Tensor() {
    return;
    ModelBuilder mb = new ModelBuilder();
    var inputLayer = mb.Input("Input", new int[]{-1,1,1,4});
    Layer hiddenDenseLayer = mb.Dense("hidden1", inputLayer, weights_h1_i, bais_h1);
    Layer hiddenActiveLayer = mb.Relu("hiddenAct", hiddenDenseLayer);
    Layer outputDenseLayer = mb.Dense("output", hiddenActiveLayer, weights_o_h1, bais_o);
    Layer outputActiveLayer = mb.Relu("outputActive",outputDenseLayer);
    mb.Output(outputActiveLayer);
    model = mb.model;
    IWorker worker = WorkerFactory.CreateWorker(model, WorkerFactory.Device.GPU);


  }
  public override void Copy(MLP other) { throw new NotImplementedException(); }

  public override void Clear()
  {
    return;
    for (int idx = 0; idx < weights_h1_i.length; idx++)
      weights_h1_i[idx] =0;
    for (int idx = 0; idx < weights_o_h1.length; idx++)
      weights_o_h1[idx] =0;
    for (int idx = 0; idx < bais_o.length; idx++)
      bais_o[idx] = 0;
    for (int idx = 0; idx < bais_h1.length; idx++)
      bais_h1[idx] = 0;
  }
   
  public override void Mutate(ref GaussianGenerator rnd, float learningRate) {
    return;
    for (int idx = 0; idx < weights_h1_i.length; idx++)
      weights_h1_i[idx] += rnd.NextFloat1() * learningRate;
    for (int idx = 0; idx < weights_o_h1.length; idx++)
      weights_o_h1[idx] += rnd.NextFloat1() * learningRate;
    for (int idx = 0; idx < bais_o.length; idx++)
      bais_o[idx] += rnd.NextFloat1() * learningRate;
    for (int idx = 0; idx < bais_h1.length; idx++)
      bais_h1[idx] += rnd.NextFloat1() * learningRate;
  }

  public override float4 Execute(float4 input) { throw new NotImplementedException(); }
  public override float GetWeight(int layer, int srcNode, int dstNode) { throw new NotImplementedException(); }
}