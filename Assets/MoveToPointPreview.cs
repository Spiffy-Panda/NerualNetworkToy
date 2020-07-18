﻿using System.Collections;
using System.Collections.Generic;
using Unity.Mathematics;
using UnityEngine;
using Random = Unity.Mathematics.Random;

[RequireComponent(typeof(PeriodicUpdate),typeof(AcademyMove))]
public class MoveToPointPreview : MonoBehaviour {
  private Random _rndu;
  private AcademyMove _Academy => GetComponent<AcademyMove>();
  public PeriodicUpdate _updater => GetComponent<PeriodicUpdate>();
  private float3[] _stateBuffer;
  private float4[] _actBuffer;
  public float2 _targetPosition;
  private int _iterations => _Academy._iterations;

  public bool _moveTarget = true;
  public float3 _stateMin = new float3(-5, -5, -math.PI);
  public float3 _stateMax = new float3(5, 5, math.PI);
  // Start is called before the first frame update
  void Start() {
    _rndu = new Random((uint)UnityEngine.Random.Range(0, int.MaxValue));
    _updater.Updated += OnPeriodicUpdate;
    _stateBuffer = new float3[_iterations];
    _actBuffer = new float4[_iterations];
  }

  public float _startAngle;

  public int CurrentIndex => (int) (_updater.TimeNormalized * _iterations);
  public float3 CurState => (_stateBuffer == null || _stateBuffer.Length != _iterations)?0:_stateBuffer[CurrentIndex];
  void OnPeriodicUpdate() {
    if (_moveTarget)
      _targetPosition = _rndu.NextFloat2(_stateMin.xy, _stateMax.xy);

    float3 state = 0;
    if (_moveTarget)
      _startAngle = _rndu.NextFloat(_stateMin.z, _stateMax.z);
    state.z = _startAngle;
    _Academy.PreviewBest(state,_targetPosition,_stateBuffer,_actBuffer);
    _NetDraw._TestMLP = _Academy._BestBrain as MLP;
    _NetDraw.FindBounds();
  }

  Toughts _NetDraw => FindObjectOfType<Toughts>();
  public void Update() {
    float2 extState = _Academy.Observe(CurState, _targetPosition);
    float2 intState = (CurrentIndex > 0) ? 0 : extState; 
    _NetDraw._observe = new float4(extState,intState);
    _NetDraw.SetAllDirty();
  }

  public void OnDrawGizmos() {
    if (_stateBuffer == null || _stateBuffer.Length != _iterations)
      return;
    

    Gizmos.color = Color.yellow;
    Gizmos.DrawWireSphere((Vector2)_stateBuffer[0].xy, 0.2f);
    Gizmos.DrawSphere((Vector2)CurState.xy, 0.1f);
    Gizmos.DrawRay((Vector2)CurState.xy,new Vector3(Mathf.Cos(CurState.z), Mathf.Sin(CurState.z)));
    Gizmos.color = Color.red;
    Gizmos.DrawSphere((Vector2)_targetPosition.xy,0.2f);
    Gizmos.color = Color.green;
    for (int i = 1; i < _stateBuffer.Length; i++)
    {
     Gizmos.DrawLine((Vector2)_stateBuffer[i].xy, (Vector2)_stateBuffer[i - 1].xy);
    }
  }
}