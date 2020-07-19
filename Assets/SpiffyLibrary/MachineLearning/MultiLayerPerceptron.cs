using System;
using System.Collections.Generic;
using Unity.Barracuda;
using UnityEngine;

namespace SpiffyLibrary.MachineLearning
{
  public class MLP_Tensor 
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

    public static Layer MBActivationByName(ref ModelBuilder mb, string name, object input, Layer.FusedActivation activation) {
      switch (activation)
      {
        case Layer.FusedActivation.Exp:
          return mb.Exp(name, input);
        case Layer.FusedActivation.Log:
          return mb.Log(name, input);
        case Layer.FusedActivation.Neg:
          return mb.Neg(name, input);
        case Layer.FusedActivation.None:
          return mb.Identity(name, input);
        case Layer.FusedActivation.Relu:
          return mb.Relu(name, input);
        case Layer.FusedActivation.Relu6:
          return mb.Relu6(name, input);
        case Layer.FusedActivation.Sigmoid:
          return mb.Sigmoid(name, input);
        case Layer.FusedActivation.Sqrt:
          return mb.Sqrt(name, input);
        case Layer.FusedActivation.Swish:
          return mb.Swish(name, input);
        case Layer.FusedActivation.Tanh:
          return mb.Tanh(name, input);
        default:
          throw new KeyNotFoundException();
      }
    }

    public MLP_Tensor(int inputSize = 4, int outputSize = 4, int hiddenSize = 4, Layer.FusedActivation activation = Layer.FusedActivation.Relu)
    {
      _inputSize = inputSize;
      _hiddenSize = hiddenSize;
      _outputSize = outputSize;
      TensorCachingAllocator tca = new TensorCachingAllocator();

      ModelBuilder mb = new ModelBuilder();
      Model.Input inputLayer = mb.Input(LayerNames.Input, new int[] { -1, 1, 1, _inputSize });
      Layer hiddenDenseLayer = mb.Dense(LayerNames.Hidden, inputLayer, tca.Alloc(new TensorShape(_inputSize, _hiddenSize)), tca.Alloc(new TensorShape(1, _hiddenSize)));

      Layer hiddenActiveLayer = MBActivationByName(ref mb, LayerNames.HiddenActive, hiddenDenseLayer, activation);
      Layer outputDenseLayer  = mb.Dense(LayerNames.Output, hiddenActiveLayer, tca.Alloc(new TensorShape(_hiddenSize,_outputSize)), tca.Alloc(new TensorShape(1, _outputSize)));
      Layer outputActiveLayer = MBActivationByName(ref mb, LayerNames.OutputActive, outputDenseLayer, activation);
      mb.Output(outputActiveLayer);
      model = mb.model;
      tca.Dispose();
    }

    public void Copy(MLP_Tensor other)
    {
      for (int iLayer = 0; iLayer < model.layers.Count; iLayer++)
      {
        Debug.Assert(model.layers[iLayer].weights.Length == other.model.layers[iLayer].weights.Length);
        Array.Copy(other.model.layers[iLayer].weights,model.layers[iLayer].weights,model.layers[iLayer].weights.Length);
      }
    }

    public void Clear()
    {
      foreach (Layer layer in model.layers)
      {
        for (int iWeight = 0; iWeight < layer.weights.Length; iWeight++)
        {
          layer.weights[iWeight] = 0;
        }
      }
    }

    public void Mutate(ref GaussianGenerator rnd, float learningRate)
    {

      foreach (Layer layer in model.layers)
      {
        for (int iWeight = 0; iWeight < layer.weights.Length; iWeight++)
        {
          layer.weights[iWeight] += rnd.NextFloat1() * learningRate;
        }
      }
    }

    public TensorShape GetLayerShape(string layerName)
    {
      foreach (Layer layer in model.layers)
      {
        if (layer.name == layerName)
        {
          if(layer.datasets.Length == 2)
            return layer.datasets[0].shape;
          else 
            throw new ArgumentException($"Layer \"{layerName}\" is not a dense layer.");
        }
      }
      throw new KeyNotFoundException($"Layer \"{layerName}\" not found.");
    }

    public float GetWeight(string layerName, int srcNode, int dstNode)
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
    public float GetBias(string layerName, int dstNode)
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