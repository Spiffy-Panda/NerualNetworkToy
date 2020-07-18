using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.Assertions;
using Random = Unity.Mathematics.Random;

[RequireComponent(typeof(EvaluationWeight))]
public class AcademyMove : MonoBehaviour {
  private GaussianGenerator _rndg = null;
  private Random _rndu;
  public int _iterations = 1000;
  public int _testPerRating = 25;
  public float _dt= 0.01f;
  public MLP_4I_4O _BestBrain => _bestBrain;
  private MLP_4I_4O _bestBrain;
  private MLP_4I_4O _otherBrain;
  public float3 _bestMetrics;
  public float _bestCost;
  private float4[] _actBuffer;
  [Range(0.0000001f, 2)]
  public float _initLearnRate = 2;
  [Range(0.0000001f, .1f)]
  public float _mutaLearnRate = 0.1f;

  public bool _isGenerating = true;
  public bool _isMutatingBest = false;
  public float2 _actionSpaceMin = new float2(0, -1);
  public float2 _actionSpaceMax = new float2(1,  1);

  public Func<float3, float> Cost=>GetComponent<EvaluationWeight>().Apply;
  public event Action<float3, float> RatingFinished;
  public event Action<float3, float> NewBestFound;
  public float3 _stateMin = new float3(-5, -5, -math.PI);
  public float3 _stateMax = new float3( 5,  5,  math.PI);
  public void Start() {
    _rndu = new Random((uint) UnityEngine.Random.Range(0, int.MaxValue));
    _rndg = new GaussianGenerator(new Random((uint)UnityEngine.Random.Range(0, int.MaxValue)));
    _actBuffer = new float4[_iterations];
    
    _bestBrain= new MLP_4I_4O_Tensor();
    _otherBrain = new MLP_4I_4O_Tensor();
    _bestBrain.Mutate(ref _rndg,1);
    _bestMetrics=  Rate(_bestBrain);
    _bestCost = Cost(_bestMetrics);
    GetComponent<EvaluationWeight>().ValueChanged+= () => _bestCost = Cost(_bestMetrics);
  }
  private float3 Rate(MLP_4I_4O brain) {
    float3 result = 0;
    for (int i = 0; i < _testPerRating; i++) {
      float3 state = _rndu.NextFloat3(_stateMin,_stateMax);
      float2 tgt = _rndu.NextFloat2(_stateMin.xy, _stateMax.xy);
      result += Rate(brain, state, tgt);
    }
    return result / _testPerRating;
  }


  private float3 Rate(MLP_4I_4O brain, float3 state, float2 tgt)
  {
    (float3 final, float3 closest)= Run(brain, state, tgt, null, _actBuffer);
    float dst = math.length(tgt - state.xy);
    float closestDistance = math.length(tgt - closest.xy) / dst;
    float stoping = math.length(final.xy - tgt.xy) / dst;
    float turning = Mathf.Abs( _actBuffer.Select(act => (act.y)).Sum() / _iterations );

    return new float3(closestDistance, stoping, turning);
  }

  public float2 Observe(float3 state, float2 tgt) {

    float2 obs;
    float2 dlt = tgt.xy - state.xy;
    float2 dir = new float2(math.cos(state.z), math.sin(state.z));
    obs.x = math.atan2(dlt.y, dlt.x);
    obs.y =math.clamp( math.dot(dlt, dir)/2,0,1);
    return obs;
  }
  private (float3 final, float3 closest) Run(MLP_4I_4O brain, float3 state, float2 tgt, float3[] stateBuffer = null, float4[] actBuffer = null)
  {
    Debug.Assert(stateBuffer == null || _iterations == stateBuffer.Length);
    Debug.Assert(actBuffer== null || _iterations == actBuffer.Length);
    float3 closest = state;
    float closestLength = math.length(tgt-state.xy);
    float2 internState = 0;
    for (int i = 0; i < _iterations; i++)
    {
      float2 obs = Observe(state,tgt);
      float4 actIntern = brain.Execute(new float4(obs,internState));
      float2 act = actIntern.xy;
      internState = actIntern.zw;
      float2 dir = new float2(math.cos(state.z), math.sin(state.z));
      float dst = math.length(tgt - state.xy);
      if ( dst < closestLength)
      {
        closestLength = dst ;
        closest = state;
      }

      act = math.clamp(act, _actionSpaceMin, _actionSpaceMax);


      state.z += act.y * _dt;
      state.xy += dir * act.x * _dt;
      if(stateBuffer != null)
        stateBuffer[i] = state;
      if (actBuffer != null)
        actBuffer[i] = actIntern;
    }

    return (state, closest);
  }
  
  private void Update() {
    if(_isGenerating)
    {
      if (!_isMutatingBest)
        _otherBrain.Clear();
      else
        _otherBrain.Copy(_bestBrain);
      _otherBrain.Mutate(ref _rndg, _isMutatingBest ? _mutaLearnRate: _initLearnRate );
      float3 metrics = Rate(_otherBrain);

      float cost = Cost(metrics);
      RatingFinished?.Invoke(metrics,cost);
      if (cost < _bestCost) {
        _bestBrain.Copy(_otherBrain);
        _bestMetrics = metrics;
        _bestCost = cost;
        NewBestFound?.Invoke(_bestMetrics,_bestCost);

      }
    }
  }

  public void PreviewBest(float3 state, float2 target,float3[] stateBuffer= null, float4[] actBuffer = null)
  {
    Debug.Assert(stateBuffer == null || stateBuffer.Length == _iterations);
    Debug.Assert(actBuffer== null || actBuffer.Length == _iterations);
    Run(_bestBrain, state, target, stateBuffer, actBuffer);
  }

}
