using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using SpiffyLibrary;
using SpiffyLibrary.MachineLearning;
using Unity.Barracuda;
using UnityEngine;
using Random = Unity.Mathematics.Random;

public class MutationManager : MonoBehaviour {
  private static MutationManager _inst;
  public static MutationManager Inst
  {
    get
    {
      if (_inst == null)
      {
        _inst = FindObjectOfType<MutationManager>();
      }

      return _inst;
    }
  }
  private GaussianGenerator _rndg = null;
  public MoveSimParams _simParams = MoveSimParams.GetDefault();
  public float _learnRate = 0.1f;

  public float[] GetZeroWeights() {
    MultiLayerPerception mlp = new MultiLayerPerception(_simParams.mlpShape,Layer.FusedActivation.Relu6);
    int[] secs = mlp.GetWeightSections();
    float[] result = new float[secs.Length];
    for (int iWeight = 0; iWeight < result.Length; iWeight++) {
      result[iWeight] = (secs[iWeight] % 10 == 2) ? 3 : 0;
    }

    return result;
  }
  public static float[] DefaultGeneratorBase(int count, Func<float> nxtFloat)
  {
    float[] weights =(Inst != null)?Inst.GetZeroWeights():new float[count];
    Mutate(weights,nxtFloat);
    return weights;
  }

  public static void Mutate(float[] weights, Func<float> nxtFloat) {

    for (int iWeight = 0; iWeight < weights.Length; iWeight++)
      weights[iWeight] = weights[iWeight] + nxtFloat();
  }

  public float[] DefaultGenerator() => DefaultGeneratorBase(_simParams.mlpShape.WeightCount, _rndg.NextFloat1);

  public float[] MutateExisting(IReadOnlyCollection<float> parent) {
    float[] result = parent.ToArray();
    Mutate(result, ()=>_learnRate * _rndg.NextFloat1());
    return result;
  }
  public void Start() {
    _rndg = new GaussianGenerator(new Random((uint) UnityEngine.Random.Range(0, int.MaxValue))); 

  }
}
