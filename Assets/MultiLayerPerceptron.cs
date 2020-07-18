using System;
using System.Collections;
using System.Collections.Generic;
using Unity.Barracuda;
using UnityEngine;
using Unity.Mathematics;
using UnityEngine.Rendering.VirtualTexturing.Procedural;
using Random = Unity.Mathematics.Random;

public enum ActivationType {
  Relu,
  Atan
};

public abstract class MLP_2I_2O
{
  protected Func<float4, float4> Activation { get; private set; }
  public MLP_2I_2O() => SetActivation(ActivationType.Relu);

  public void SetActivation(ActivationType func)
  {
    switch (func)
    {
      case ActivationType.Relu:
        Activation = relu;
        break;
      case ActivationType.Atan:
        Activation = math.atan;
        break;
    }
  }

  public abstract void Copy(MLP_2I_2O other);
  public abstract void Clear();

  public abstract void Mutate(ref GaussianGenerator rnd, float learningRate);

  public abstract float2 Execute(float2 input);

  private static float4 relu(float4 x) { return math.max(float4.zero, x); }
}
public abstract class MLP_4I_4O
{
  protected Func<float4, float4> Activation { get; private set; }
  public MLP_4I_4O() => SetActivation(ActivationType.Relu);

  public void SetActivation(ActivationType func)
  {
    switch (func)
    {
      case ActivationType.Relu:
        Activation = relu;
        break;
      case ActivationType.Atan:
        Activation = math.atan;
        break;
    }
  }

  public abstract void Copy(MLP_4I_4O other);
  public abstract void Clear();

  public abstract void Mutate(ref GaussianGenerator rnd, float learningRate);

  public abstract float4 Execute(float4 input);

  private static float4 relu(float4 x) { return math.max(float4.zero, x); }
}
public class MLP_2I_3x4H_2O: MLP_2I_2O
{
  public const int kInputSize = 2;
  public const int kHiddenSize = 4;
  public const int kOutputSize = 2;

  public float2x4 weights_o_h3;
  public float4x4 weights_h3_h2;
  public float4x4 weights_h2_h1;
  public float4x2 weights_h1_i;
  public float4x3 bias_h;
  public float2 bias_h2o;

  public override void Copy(MLP_2I_2O other) {
    Debug.Assert(other.GetType() == GetType(), "MLP_2I_2O copy missmatch.");
    Copy((MLP_2I_3x4H_2O)other);
  }

  public override void Clear()
  {
    for (int iCol = 0; iCol < kOutputSize; iCol++)
      weights_o_h3[iCol] = new float2();
    for (int iCol = 0; iCol < kOutputSize; iCol++)
      weights_h3_h2[iCol] = new float4();
    for (int iCol = 0; iCol < kOutputSize; iCol++)
      weights_h2_h1[iCol] = new float4();
    for (int iCol = 0; iCol < kInputSize; iCol++)
      weights_h1_i[iCol] = new float4();
    bias_h = new float4x3();
    bias_h2o = new float2();
  }

  private void Copy(MLP_2I_3x4H_2O other)
  {
    weights_o_h3  = other.weights_o_h3 ;
    weights_h3_h2 = other.weights_h3_h2;
    weights_h2_h1 = other.weights_h2_h1;
    weights_h1_i  = other.weights_h1_i ;
    bias_h        = other.bias_h       ;
    bias_h2o      = other.bias_h2o;
  }
  public override void Mutate(ref GaussianGenerator rnd, float learningRate)
  {
    float lr = learningRate;

    for (int iCol = 0; iCol < kOutputSize; iCol++)
      weights_o_h3[iCol] = lr * rnd.NextFloat2();
    for (int iCol = 0; iCol < kOutputSize; iCol++)
      weights_h3_h2[iCol] = lr * rnd.NextFloat4();
    for (int iCol = 0; iCol < kOutputSize; iCol++)
      weights_h2_h1[iCol] = lr * rnd.NextFloat4();
    for (int iCol = 0; iCol < kInputSize; iCol++)
      weights_h1_i[iCol] = lr * rnd.NextFloat4();
    bias_h = lr * rnd.NextFloat4x3();
    bias_h2o = lr * rnd.NextFloat2();
  }

  public override float2 Execute(float2 input) {

    float4 value = math.mul(weights_h1_i, input) + bias_h[0];
    value = Activation(value);
    value = math.mul(weights_h2_h1, value) + bias_h[1];
    value = Activation(value);
    value = math.mul(weights_h3_h2, value) + bias_h[2];
    value = Activation(value);
    return math.mul(weights_o_h3, value) + bias_h2o;
  }
}
public class MLP_2I_4H_2O : MLP_2I_2O
{
  public const int kInputSize = 2;
  public const int kHiddenSize = 4;
  public const int kOutputSize = 2;

  public float2x4 weights_o_h1;
  public float4x2 weights_h1_i;
  public float4 bias_h;
  public float2 bias_h2o;

  public override void Clear()
  {
    for (int iCol = 0; iCol < kInputSize; iCol++)
      weights_h1_i[iCol] = new float4();
    for (int iCol = 0; iCol < kHiddenSize; iCol++)
      weights_o_h1[iCol] = new float2();
    bias_h = new float4();
    bias_h2o = new float2();

  }

  public override void Copy(MLP_2I_2O other)
  {
    Debug.Assert(other.GetType() == GetType(), $"MLP_2I_2O copy missmatch. {other.GetType()} vs {GetType()}");
    Copy((MLP_2I_4H_2O)other);
  }
  public void Copy(MLP_2I_4H_2O other)
  {
    weights_h1_i = other.weights_h1_i;
    weights_o_h1 = other.weights_o_h1;
    bias_h = other.bias_h;
    bias_h2o = other.bias_h2o;
  }
  public override void Mutate(ref GaussianGenerator rnd, float learningRate)
  {
    float lr = learningRate;
    for (int iCol = 0; iCol < kInputSize; iCol++)
      weights_h1_i[iCol] += lr * rnd.NextFloat4();
    for (int iCol = 0; iCol < kHiddenSize; iCol++)
      weights_o_h1[iCol] += lr * rnd.NextFloat2();
    bias_h += lr * rnd.NextFloat4();
    bias_h2o += lr * rnd.NextFloat2();
  }

  public override float2 Execute(float2 input)
  {
    var value = math.mul(weights_h1_i, input) + bias_h;
    value = Activation(value);
    return math.mul(weights_o_h1, value) + bias_h2o;
  }

  public float4[] GetHiddenValues(float2 input, bool activate = true)
  {
    float4[] result = new float4[1];
    result[0] = math.mul(weights_h1_i, input) + bias_h;
    if (activate)
      result[0] = Activation(result[0]);
    return result;
  }
}

public class MLP_4I_4O_Tensor : MLP_4I_4O {

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

  public MLP_4I_4O_Tensor() {
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
  public override void Copy(MLP_4I_4O other) { throw new NotImplementedException(); }

  public override void Clear()
  {
    for (int idx = 0; idx < weights_h1_i.length; idx++)
      weights_h1_i[idx] =0;
    for (int idx = 0; idx < weights_o_h1.length; idx++)
      weights_o_h1[idx] =0;
    for (int idx = 0; idx < bais_o.length; idx++)
      bais_o[idx] = 0;
    for (int idx = 0; idx < bais_h1.length; idx++)
      bais_h1[idx] = 0;
  }

  public override void Mutate(ref GaussianGenerator rnd, float learningRate)
  {
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
}

public class MLP_4I_4H_4O : MLP_4I_4O
{
  public const int kInputSize = 4;
  public const int kHiddenSize = 4;
  public const int kOutputSize = 4;

  public float4x4 weights_o_h1;
  public float4x4 weights_h1_i;
  public float4 bias_h;
  public float4 bias_h2o;

  public override void Clear()
  {
    for (int iCol = 0; iCol < kInputSize; iCol++)
      weights_h1_i[iCol] = new float4();
    for (int iCol = 0; iCol < kHiddenSize; iCol++)
      weights_o_h1[iCol] = new float4();
    bias_h = new float4();
    bias_h2o = new float4();

  }

  public override void Copy(MLP_4I_4O other)
  {
    Debug.Assert(other.GetType() == GetType(), $"MLP_4I_4O copy missmatch. {other.GetType()} vs {GetType()}");
    Copy((MLP_4I_4H_4O)other);
  }
  public void Copy(MLP_4I_4H_4O other)
  {
    weights_h1_i = other.weights_h1_i;
    weights_o_h1 = other.weights_o_h1;
    bias_h = other.bias_h;
    bias_h2o = other.bias_h2o;
  }
  public override void Mutate(ref GaussianGenerator rnd, float learningRate)
  {
    float lr = learningRate;
    for (int iCol = 0; iCol < kInputSize; iCol++)
      weights_h1_i[iCol] += lr * rnd.NextFloat4();
    for (int iCol = 0; iCol < kHiddenSize; iCol++)
      weights_o_h1[iCol] += lr * rnd.NextFloat4();
    bias_h += lr * rnd.NextFloat4();
    bias_h2o += lr * rnd.NextFloat4();
  }

  public override float4 Execute(float4 input)
  {
    var value = math.mul(weights_h1_i, input) + bias_h;
    value = Activation(value);
    return math.mul(weights_o_h1, value) + bias_h2o;
  }

  public float4[] GetHiddenValues(float4 input, bool activate = true)
  {
    float4[] result = new float4[1];
    result[0] = math.mul(weights_h1_i, input) + bias_h;
    if (activate)
      result[0] = Activation(result[0]);
    return result;
  }
}
