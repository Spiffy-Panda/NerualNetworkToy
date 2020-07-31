using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using SpiffyLibrary;
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

  public static float[] DefaultGeneratorBase(int count, Func<float> nxtFloat)
  {
    float[] weights = new float[count];
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
