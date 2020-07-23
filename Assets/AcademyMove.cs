using SpiffyLibrary;
using SpiffyLibrary.MachineLearning;
using System;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Timers;
using Unity.Barracuda;
using Unity.Mathematics;
using UnityEngine;
using Debug = UnityEngine.Debug;
using Random = Unity.Mathematics.Random;


public struct MoveSimParams
{
  public int iterations;
  public int runCount;
  public float dt;
  public float2 actionSpaceMin;
  public float2 actionSpaceMax;
  public override string ToString()
  {
    return $"iterations:{iterations}\n" +
     $"runCount:{runCount}\n" +
     $"dt:{dt}\n" +
     $"actionSpace:{actionSpaceMin}-{actionSpaceMax}";
  }
}
public class MoveContext
{
  private static int _idAllocator = 0;
  public readonly int _id;
  public readonly MoveSimParams _simParams;
  public MultiLayerPerception _mlpModel;
  public float[] Weights => _mlpModel.GetReadonlyWeights();
  public GaussianGenerator _rndn => GaussianGenerator.Inst;
  public static Random _rndu = new Random((uint)UnityEngine.Random.Range(0, Int32.MaxValue));
  public StringBuilder _log = new StringBuilder();
  public Dictionary<string,float> _metrics;
  public IWorker _worker;
  public Tensor inTensor;
  public IEnumerator _runCoro;
  public bool Finished { get; private set; } = false;
  public float Progress = 0;
  public MoveContext(MoveSimParams simParams, MultiLayerPerception.Shape shape, float[] _weights, float3 stateMin, float3 stateMax)
  {
    _id = _idAllocator++;
    _log.AppendLine($"Created with ID {_id}.");
    _mlpModel = new MultiLayerPerception(shape, Layer.FusedActivation.Relu6);
    _mlpModel.LoadWeights(_weights);
    _simParams = simParams;
    _runCoro = RateThread(stateMin, stateMax);
    _log.AppendLine($"Constructer finished.");
  }

  public void Start()
  {
    _log.AppendLine($"Start Called");
    _log.AppendLine("Creating Worker");
    inTensor = AcademyMove.TensorAllocator.Alloc(new TensorShape(_simParams.runCount, _mlpModel._shape.inputSize));
    inTensor.name = "inTensor_"+_id;


    
    _worker= WorkerFactory.CreateWorker(WorkerFactory.Type.ComputePrecompiled, _mlpModel.model,false);
    //_worker = WorkerFactory.CreateWorker(_mlpModel.model, WorkerFactory.Device.GPU);
    _log.AppendLine("Worker Created");

  }

  public bool Tick() => _runCoro.MoveNext(); 
   
  private IEnumerator RateThread(float3 stateMin, float3 stateMax)
  {
    _log.AppendLine("RateThread Started.");
    var runMetrics = new List<MetricInfo>[_simParams.runCount];
    for (int iRun = 0; iRun < _simParams.runCount; iRun++) {
      runMetrics[iRun] = new List<MetricInfo>();
      runMetrics[iRun].Add(new ClosestApproachMetric());
      runMetrics[iRun].Add(new FinalDistanceMetric());
      runMetrics[iRun].Add(new OverRotationMetric());
    }
      
    yield return null;

    float3[] states = new float3[_simParams.runCount];

    for (int iRun = 0; iRun < _simParams.runCount; iRun++)
      states[iRun] = _rndu.NextFloat3(stateMin, stateMax);
    _log.AppendLine("Variables Initialized.");
    for (int i = 0; i < _simParams.iterations; i++) {
      try {
        for (int iRun = 0; iRun < _simParams.runCount; iRun++) {
          float2 obs = AcademyMove.Observe(states[iRun]);
          inTensor[iRun, 0] = obs.x;
          inTensor[iRun, 1] = obs.y;
        }
      }
      catch (Exception e) {
        _log.AppendLine(e.Message);
        Debug.Log(e.StackTrace);
        throw;
      }

      _worker.SetInput(inTensor);
      yield return null;
      _worker.Execute().FlushSchedule(true);
      while (_worker.scheduleProgress < 1)
        yield return null;


      using (Tensor outTensor = _worker.PeekOutput()) {
        outTensor.name = "outTensor";
        outTensor.PrepareCacheForAccess(false);
        for (int iRun = 0; iRun < _simParams.runCount; iRun++) {
          float2 act = 0;
          Debug.Assert(0 <= outTensor[iRun, 0] && outTensor[iRun, 0] <= 6);
          Debug.Assert(0 <= outTensor[iRun, 1] && outTensor[iRun, 1] <= 6);
          act.x = math.remap(0, 6, 0, 1, outTensor[iRun, 0]);
          act.y = math.remap(0, 6, -1, 1, outTensor[iRun, 1]);
          ;
          inTensor[iRun, 2] = outTensor[iRun, 2];
          inTensor[iRun, 3] = outTensor[iRun, 3];
          float2 dir = new float2(math.cos(states[iRun].z), math.sin(states[iRun].z));
          act = math.clamp(act, _simParams.actionSpaceMin, _simParams.actionSpaceMax);
          states[iRun].z += act.y * _simParams.dt;
          states[iRun].xy += dir * act.x * _simParams.dt;

          foreach (var metric in runMetrics[iRun]) {
            metric.EvalIteractionTick(states[iRun], act);
          }
        }
      }

      Progress = (i+1f) / _simParams.iterations;
    }

    _log.AppendLine("Iterations Over.");

    _metrics = runMetrics[0].ToDictionary(mi => mi.Name, mi=>0f);
    for (int iRun = 0; iRun < _simParams.runCount; iRun++) {
      foreach (var metricInfo in runMetrics[iRun]) {
        _metrics[metricInfo.Name] += metricInfo.TotalValue;
      }
    }


    _log.AppendLine("Done");
    inTensor.Dispose();
    _worker.Dispose();
    _log.AppendLine("Worker disposed of.");
    Finished = true;

  }

}


[RequireComponent(typeof(EvaluationWeight))]
public class AcademyMove : MonoBehaviour
{
  public static TensorCachingAllocator TensorAllocator { get; private set; }
  private GaussianGenerator _rndg = null;

  public MoveSimParams _simParams = new MoveSimParams
  {
    iterations = 50*5,
    actionSpaceMin = new float2(0, -1),
    actionSpaceMax = new float2(1, 1),
    dt = 1/ 50f,
    runCount = 25
  };
  
  [Range(0.0000001f, 2)]
  public float _initLearnRate = 2;
  [Range(0.0000001f, .1f)]
  public float _mutaLearnRate = 0.1f;

  public bool _isGenerating = true;
  public bool _isMutatingBest = false;

  public MultiLayerPerception.Shape _mlpShape = new MultiLayerPerception.Shape { inputSize = 4, hiddenSize = 4, outputSize = 4 };
  public float3 _stateMin = new float3(-5, -5, -math.PI);
  public float3 _stateMax = new float3(5, 5, math.PI);
  public List<MoveContext> _currentGeneration = new List<MoveContext>();
  public int _generationSize = 100;
  private int _curGeneration = -1;
  public void Start()
  {
    TensorAllocator = new TensorCachingAllocator();
    _rndg = new GaussianGenerator(new Random((uint)UnityEngine.Random.Range(0, int.MaxValue)));
  }

  public void OnDisable() => TensorAllocator.Dispose();

  public static float2 Observe(float3 state)
  {

    float2 dlt = 0 - state.xy;
    float2 dir = new float2(math.cos(state.z), math.sin(state.z));
    return new float2(math.atan2(dlt.y, dlt.x), math.clamp(math.dot(dlt, dir) / 2, 0, 1));
  }
  
  private void Update()
  {
    if (_currentGeneration.Count == 0) {
      _curGeneration++;
      Debug.Log($"Creating Generation: {_curGeneration}");
      for (int iContext = 0; iContext < _generationSize; iContext++) {
        float[] weights = new float[_mlpShape.WeightCount];
        for (int iWeight = 0; iWeight < weights.Length; iWeight++)
          weights[iWeight] = _rndg.NextFloat1();
        _currentGeneration.Add(new MoveContext(_simParams, _mlpShape, weights, _stateMin, _stateMax));
        _currentGeneration[_currentGeneration.Count-1].Start();
      }
    } else{
      Stopwatch stopwatch = new Stopwatch();
      stopwatch.Start();
      int tickCount = 0;
      for (var index = 0; index < 10000; index++) {
        var context = _currentGeneration[index%_currentGeneration.Count];
        if(context.Finished)
          continue;
        if (!context.Tick()) {
          GeneBankManager.Inst.Evaluate(context.Weights, context._metrics);
        }

        tickCount++;
        if (stopwatch.ElapsedMilliseconds > 100)
          break;
      }

      _currentGeneration.RemoveAll(context => context.Finished);
    }
  }

}
