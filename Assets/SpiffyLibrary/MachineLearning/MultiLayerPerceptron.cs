using System;
using System.Collections.Generic;
using Unity.Barracuda;
using UnityEngine;

namespace SpiffyLibrary.MachineLearning
{
  public class MultiLayerPerception 
  {
    public static class LayerNames
    {
      public const string Input = "Input";
      public const string Hidden = "Hidden";
      public static string HiddenActive => Hidden + "_Active";
      public const string Output = "Output";
      public static string OutputActive => Output+ "_Active";
    }
    public struct Shape {
      public int hiddenSize;
      public int inputSize;
      public int outputSize;
      public int WeightCount => (inputSize +1 )* hiddenSize + (hiddenSize +1)* outputSize;
      public override string ToString() => $"<I{inputSize},H{hiddenSize},O{outputSize}:{WeightCount}>";
    } 
    public readonly Model model;

    public readonly Shape _shape;
    // Readonly as in _weights is the same size and instance, but not content.
    private readonly float[] m_cache;

    private void PrepareCache() {
      int totIdx = 0;
      try {
        foreach (Layer layer in model.layers) {
          foreach (float w in layer.weights) {
            m_cache[totIdx++] = w;
          }
        }
      }
      catch (Exception e) {
        Debug.LogError($"{m_cache.Length} with shape {_shape}\n{e.Message}");
        throw;
      }
      Debug.Assert(totIdx == m_cache.Length);
    }

    public float[] GetReadonlyWeights() => m_cache;

    public void LoadWeights(float[] otherArray)
    {
      int totIdx = 0;
      Debug.Assert(otherArray.Length == _shape.WeightCount);
      Array.Copy(otherArray,m_cache,_shape.WeightCount);
      foreach (Layer layer in model.layers)
      {
        for (int i = 0; i < layer.weights.Length; i++)
        {
          layer.weights[i] = m_cache[totIdx++];
        }
      }
      Debug.Assert(totIdx == m_cache.Length);
    }
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

    public MultiLayerPerception(Shape shape, Layer.FusedActivation activation = Layer.FusedActivation.Relu) {
      _shape = shape;
      ModelBuilder mb = new ModelBuilder();
      m_cache = new float[_shape.WeightCount];
      { // Build the model
        TensorCachingAllocator tca = new TensorCachingAllocator(); 
        string prevLayerName = "[ERROR]NOT_INITIALIZED";
        prevLayerName = mb.Input(LayerNames.Input, new int[] { -1, 1, 1, _shape.inputSize }).name;
        prevLayerName = mb.Dense(LayerNames.Hidden, prevLayerName, tca.Alloc(new TensorShape(_shape.inputSize, _shape.hiddenSize)), tca.Alloc(new TensorShape(1, _shape.hiddenSize))).name;
        prevLayerName = MBActivationByName(ref mb, LayerNames.HiddenActive, prevLayerName, activation).name;
        prevLayerName = mb.Dense(LayerNames.Output, prevLayerName, tca.Alloc(new TensorShape(_shape.hiddenSize, _shape.outputSize)), tca.Alloc(new TensorShape(1, _shape.outputSize))).name;
        prevLayerName = MBActivationByName(ref mb, LayerNames.OutputActive, prevLayerName, activation).name;
        tca.Dispose();
        Debug.Assert(prevLayerName == mb.Output(prevLayerName));
        model = mb.model;
      }
      PrepareCache();
    }

    public void Copy(MultiLayerPerception other)
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
      PrepareCache();
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
      PrepareCache();
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