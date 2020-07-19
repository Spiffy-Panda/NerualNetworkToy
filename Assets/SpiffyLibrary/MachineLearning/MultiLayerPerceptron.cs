using System;
using System.Collections.Generic;
using Unity.Barracuda;
using UnityEngine;

namespace SpiffyLibrary.MachineLearning
{

  public abstract class MLP
  {
    public abstract void Copy(MLP other);
    public abstract void Clear();

    public abstract void Mutate(ref GaussianGenerator rnd, float learningRate);


    public abstract float GetWeight(string layerName, int srcNode, int dstNode);
    public abstract float GetBias(string layerName, int dstNode);
  }

  public class MLP_Tensor : MLP
  {
    public static class LayerNames
    {
      public const string Input = "Input";
      public const string Hidden = "Hidden";
      public static string HiddenActive => Hidden + "_Active";
      public const string Output = "Output";
      public static string OutputActive => Output+ "_Active";
    }
    public readonly int _hiddenSize;
    public readonly int _inputSize;
    public readonly int _outputSize;
    public readonly Model model;

    public Layer.FusedActivation activation = Layer.FusedActivation.Sigmoid;

    public MLP_Tensor(int inputSize = 4, int outputSize = 4, int hiddenSize = 4)
    {
      _inputSize = inputSize;
      _hiddenSize = hiddenSize;
      _outputSize = outputSize;
      TensorCachingAllocator tca = new TensorCachingAllocator();

      ModelBuilder mb = new ModelBuilder();
      Model.Input inputLayer = mb.Input(LayerNames.Input, new int[] { -1, 1, 1, _inputSize });
      Layer hiddenDenseLayer = mb.Dense(LayerNames.Hidden, inputLayer, tca.Alloc(new TensorShape(_inputSize, _hiddenSize)), tca.Alloc(new TensorShape(1, _hiddenSize)));
      Layer hiddenActiveLayer = mb.Relu(LayerNames.HiddenActive, hiddenDenseLayer);
      Layer outputDenseLayer = mb.Dense(LayerNames.Output, hiddenActiveLayer, tca.Alloc(new TensorShape(_hiddenSize,_outputSize)), tca.Alloc(new TensorShape(1, _outputSize)));
      Layer outputActiveLayer = mb.Relu(LayerNames.OutputActive, outputDenseLayer);
      mb.Output(outputActiveLayer);
      model = mb.model;
      tca.Dispose();
    }
    public override void Copy(MLP other) { throw new NotImplementedException(); }

    public override void Clear()
    {
      foreach (Layer layer in model.layers)
      {
        for (int iWeight = 0; iWeight < layer.weights.Length; iWeight++)
        {
          layer.weights[iWeight] = 0;
        }
      }
    }

    public override void Mutate(ref GaussianGenerator rnd, float learningRate)
    {

      foreach (Layer layer in model.layers)
      {
        for (int iWeight = 0; iWeight < layer.weights.Length; iWeight++)
        {
          layer.weights[iWeight] += rnd.NextFloat1() * learningRate;
        }
      }
    }


    public override float GetWeight(string layerName, int srcNode, int dstNode)
    {
      foreach (Layer layer in model.layers)
      {
        if (layer.name == layerName)
        {
          TensorShape wShape = layer.datasets[0].shape;
          int idx = dstNode + wShape.flatWidth * srcNode;
          Debug.Assert(idx < layer.datasets[0].length);
          return layer.weights[idx];
        }
      }
      throw new KeyNotFoundException($"Layer \"{layerName}\" not found.");
    }
    public override float GetBias(string layerName, int dstNode)
    {
      foreach (Layer layer in model.layers)
      {
        if (layer.name == layerName)
        {
          Debug.Assert(dstNode < layer.datasets[1].length);
          long idx = dstNode + layer.datasets[1].offset;
          return layer.weights[idx];
        }
      }
      throw new KeyNotFoundException($"Layer \"{layerName}\" not found.");
    }
  }
}