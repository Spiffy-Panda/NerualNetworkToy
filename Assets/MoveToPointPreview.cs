using System.Collections;
using System.Collections.Generic;
using System.Linq;
using SpiffyLibrary;
using SpiffyLibrary.MachineLearning;
using Unity.Mathematics;
using UnityEngine;
using Unity.Barracuda;
using Random = Unity.Mathematics.Random;

[RequireComponent(typeof(PeriodicUpdate))]
public class MoveToPointPreview : MonoBehaviour {
  private Random _rndu;
  public PeriodicUpdate _updater => GetComponent<PeriodicUpdate>();
  private float3[] _stateBuffer;
  private float2[] _actBuffer;
  public float2 _targetPosition;

  public bool _moveTarget = true;
  private MoveSimParams _simParams = MoveSimParams.GetDefault();
  // Start is called before the first frame update
  void Start() {
    _rndu = new Random((uint)UnityEngine.Random.Range(0, int.MaxValue));
    _updater.Updated += OnPeriodicUpdate;
    _stateBuffer = new float3[_simParams.iterations];
    _actBuffer = new float2[_simParams.iterations];
  }

  public float _startAngle;

  public int CurrentIndex => (int) (_updater.TimeNormalized * _simParams.iterations);
  public float3 CurState => (_stateBuffer == null || _stateBuffer.Length != _simParams.iterations) ?0:_stateBuffer[CurrentIndex];
  void OnPeriodicUpdate() {
    if (GeneBankManager.Inst.GenomeCount <= 0)
      return;

    float3 state = MoveContext.GetRandomState(in _simParams);
    ParetoGeneBank.Genome gi =GeneBankManager.Inst.GetMinMetricGenome(ClosestApproachMetric.MetricName);
    Debug.Log(gi);
    MultiLayerPerception mlp = new MultiLayerPerception(_simParams.mlpShape, Layer.FusedActivation.Relu6);
    mlp.LoadWeights(gi._weights.ToArray());
    IWorker worker = WorkerFactory.CreateWorker(WorkerFactory.Type.Auto,mlp.model,false);
    float2 obs = AcademyMove.Observe(state);
    Debug.Log(_simParams);
    Tensor inTensor = new Tensor(1,_simParams.mlpShape.inputSize);
    int runIdx = 0;
    
    for (int i = 0; i < _simParams.iterations; i++) {
      for (int iINode = 0; iINode < _simParams.mlpShape.inputSize; iINode++) 
        inTensor[runIdx, iINode] = (iINode < 2)? obs[iINode]: 0;

      worker.SetInput(inTensor);
      worker.Execute().FlushSchedule(true);
      using (Tensor outTensor = worker.PeekOutput()) {
        float2 act = 0;
        Debug.Assert(0 <= outTensor[runIdx, 0] && outTensor[runIdx, 0] <= 6);
        Debug.Assert(0 <= outTensor[runIdx, 1] && outTensor[runIdx, 1] <= 6);
        act.x = math.remap(0, 6, 0, 1, outTensor[runIdx, 0]);
        act.y = math.remap(0, 6, -1, 1, outTensor[runIdx, 1]);

        inTensor[runIdx, 2] = outTensor[runIdx, 2];
        inTensor[runIdx, 3] = outTensor[runIdx, 3];
        float2 dir = new float2(math.cos(state.z), math.sin(state.z));
        act = math.clamp(act, _simParams.actionSpaceMin, _simParams.actionSpaceMax);
        _actBuffer[i] = act;
        state.z += act.y * _simParams.dt;
        state.xy += dir * act.x * _simParams.dt;
      }
      _stateBuffer[i] = state;
    }
    worker.Dispose();
    inTensor.Dispose();
    //_Academy.PreviewBest(state,_targetPosition,_stateBuffer,_actBuffer);
    //_NetDraw._TestMLP = _Academy._BestBrain as MLP;
  }

  Toughts _NetDraw => FindObjectOfType<Toughts>();
  public float3 _curState = 0;
  public void Update() {
    _curState = CurState;
    //Tensor obsTensor = new Tensor(new int[] {1, 1, 1, 4});
    //
    //float2 obs = AcademyMove.Observe(new float3(_targetPosition-CurState.xy,CurState.z));
    //obsTensor[0] = obs[0];
    //obsTensor[1] = obs[1];
    //_NetDraw._observe = obsTensor.ToReadOnlyArray();
    //_NetDraw.SetAllDirty();
    //obsTensor.Dispose();
  }

  public void OnDrawGizmos() {
    if (_stateBuffer == null || _stateBuffer.Length != _simParams.iterations)
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
