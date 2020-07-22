using SpiffyLibrary;
using SpiffyLibrary.MachineLearning;
using System;
using System.Collections;
using System.Diagnostics;
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
  private float4[] _actBuffer;
  public GaussianGenerator _rndn => GaussianGenerator.Inst;
  public static Random _rndu = new Random((uint)UnityEngine.Random.Range(0, Int32.MaxValue));
  public StringBuilder _log = new StringBuilder();
  public float3 _metrics;
  public IWorker _worker;
  public Tensor inTensor;
  public IEnumerator _runCoro;
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
    AcademyMove.m_ops.Prepare(inTensor);


    
    _worker= WorkerFactory.CreateWorker(WorkerFactory.Type.ComputePrecompiled, _mlpModel.model,false);
    //_worker = WorkerFactory.CreateWorker(_mlpModel.model, WorkerFactory.Device.GPU);
    _log.AppendLine("Worker Created");

  }

  public bool Tick() => _runCoro.MoveNext(); 
   
  private IEnumerator RateThread(float3 stateMin, float3 stateMax)
  {
    _log.AppendLine("RateThread Started.");
    _actBuffer = new float4[_simParams.iterations];
    float3[] runMetrics = new float3[_simParams.runCount];
    yield return null;

    float3[] states = new float3[_simParams.runCount];

    for (int iRun = 0; iRun < _simParams.runCount; iRun++)
    {
      states[iRun] = _rndu.NextFloat3(stateMin, stateMax);
      runMetrics[iRun] = new float3(float.PositiveInfinity, float.NaN, 0);
    }
    _log.AppendLine("Variables Initialized.");
    for (int i = 0; i < _simParams.iterations; i++)
    {
      if(i%100==0)
        yield return null;
      try {
        for (int iRun = 0; iRun < _simParams.runCount; iRun++)
        {
          float2 obs = AcademyMove.Observe(states[iRun]);
          inTensor[iRun, 0] = obs.x;
          inTensor[iRun, 1] = obs.y;
        }
      }
      catch (Exception e) {
        _log.AppendLine(e.Message);
        throw;
      }

      _worker.Execute(inTensor);
      while (_worker.scheduleProgress < 1)
      {
        yield return null;
      }


      using (Tensor outTensor = _worker.PeekOutput()) {
        outTensor.PrepareCacheForAccess(false);
        for (int iFrame = 0; iFrame < 3; iFrame++) {
         // yield return null;
        }
        for (int iRun = 0; iRun < _simParams.runCount; iRun++)
        {
          float2 act = 0;
          act.x = outTensor[iRun, 0];
          act.y = outTensor[iRun, 1];
          inTensor[iRun, 2] = outTensor[iRun, 2];
          inTensor[iRun, 3] = outTensor[iRun, 3];
          float2 dir = new float2(math.cos(states[iRun].z), math.sin(states[iRun].z));
          act = math.clamp(act, _simParams.actionSpaceMin, _simParams.actionSpaceMax);
          states[iRun].z += act.y * _simParams.dt;
          states[iRun].xy += dir * act.x * _simParams.dt;

          {
            float dst = math.length(states[iRun].xy);
            if (dst < runMetrics[iRun].x)
            {
              runMetrics[iRun].x = dst;
            }

            runMetrics[iRun].z += math.abs(states[iRun].z);
          }
        }
      }
    }

    _log.AppendLine("Iterations Over.");
    for (int iRun = 0; iRun < _simParams.runCount; iRun++)
    {
      float dst = math.length(states[iRun].xy);
      runMetrics[iRun].y = dst;
    }

    _metrics = 0;
    _log.AppendLine(string.Join("\n", runMetrics));
    for (int iRun = 0; iRun < _simParams.runCount; iRun++)
    {
      _metrics += runMetrics[iRun] / _simParams.runCount;
    }


    _log.AppendLine("Done");
    inTensor.Dispose();
    _worker.Dispose();
    _log.AppendLine("Worker disposed of.");


  }
#if false
  private (float3 final, float3 closest) Run(Model model, float3 state, float2 tgt, float3[] stateBuffer = null, float4[] actBuffer = null)
  {
    Debug.Assert(stateBuffer == null || _simParams.iterations == stateBuffer.Length);
    Debug.Assert(actBuffer== null || _simParams.iterations == actBuffer.Length);
    float3 closest = state;
    float closestLength = math.length(tgt-state.xy);
    Tensor inTensor = AcademyMove.TensorAllocator.Alloc(new TensorShape(1, 4));
    inTensor[0, 2] = 0;
    inTensor[0, 3] = 0; 

    using (IWorker oneshotSyncWorker =
      WorkerFactory.CreateWorker(model, WorkerFactory.Device.GPU)) {

      for (int i = 0; i < _simParams.iterations; i++) {
        //AcademyMove.Observe(ref inTensor, state, tgt);
        oneshotSyncWorker.Execute(inTensor).FlushSchedule();
        float2 act = 0;
        using (Tensor outTensor = oneshotSyncWorker.PeekOutput())
        {
          act.x = outTensor[0,0];
          act.y = outTensor[0,1];
          inTensor[0, 2] = outTensor[0, 2];
          inTensor[0, 3] = outTensor[0, 3];
        }
        float2 dir = new float2(math.cos(state.z), math.sin(state.z));
        float dst = math.length(tgt - state.xy);
        if (dst < closestLength) {
          closestLength = dst;
          closest = state;
        }

        act = math.clamp(act, _simParams.actionSpaceMin, _simParams.actionSpaceMax);


        state.z += act.y * _simParams.dt;
        state.xy += dir * act.x * _simParams.dt;
        if (stateBuffer != null)
          stateBuffer[i] = state;
        if (actBuffer != null)
          actBuffer[i] = new float4(act, inTensor[0, 2], inTensor[0, 3]);
      }
      inTensor.Dispose();
    }

    return (state, closest);
  }
#endif

}


[RequireComponent(typeof(EvaluationWeight))]
public class AcademyMove : MonoBehaviour
{
  private static AcademyMove _inst;
  private TensorCachingAllocator _alloc;
  public static TensorCachingAllocator TensorAllocator => _inst._alloc;
  public static IOps m_ops = null;
  private GaussianGenerator _rndg = null;
  private Random _rndu;

  public MoveSimParams _simParams = new MoveSimParams
  {
    iterations = 10000,
    actionSpaceMin = new float2(0, -1),
    actionSpaceMax = new float2(1, 1),
    dt = 0.01f,
    runCount = 25
  };

  public float3 _bestMetrics;
  public float _bestCost;
  [Range(0.0000001f, 2)]
  public float _initLearnRate = 2;
  [Range(0.0000001f, .1f)]
  public float _mutaLearnRate = 0.1f;

  public bool _isGenerating = true;
  public bool _isMutatingBest = false;
  public MultiLayerPerception.Shape _mlpShape = new MultiLayerPerception.Shape { inputSize = 4, hiddenSize = 4, outputSize = 4 };
  public Func<float3, float> Cost => GetComponent<EvaluationWeight>().Apply;
  public event Action<float3, float> RatingFinished;
  public event Action<float3, float> NewBestFound;
  public float3 _stateMin = new float3(-5, -5, -math.PI);
  public float3 _stateMax = new float3(5, 5, math.PI);
  public MoveContext _context;
  public void Start()
  {
    _inst = this;
    _alloc = new TensorCachingAllocator();
    m_ops = new ReferenceComputeOps(ComputeShaderSingleton.Instance.referenceKernels, _alloc);
    _rndu = new Random((uint)UnityEngine.Random.Range(0, int.MaxValue));
    _rndg = new GaussianGenerator(new Random((uint)UnityEngine.Random.Range(0, int.MaxValue)));
    float[] weights = new float[_mlpShape.WeightCount];
    for (int iWeight = 0; iWeight < weights.Length; iWeight++)
      weights[iWeight] = _rndg.NextFloat1();
    _context = new MoveContext(_simParams, _mlpShape, weights, _stateMin, _stateMax);
    _bestMetrics = float.PositiveInfinity;
    _bestCost = Cost(_bestMetrics);

    GetComponent<EvaluationWeight>().ValueChanged += () => _bestCost = Cost(_bestMetrics);
  }

  public static float2 Observe(float3 state)
  {

    float2 dlt = 0 - state.xy;
    float2 dir = new float2(math.cos(state.z), math.sin(state.z));
    return new float2(math.atan2(dlt.y, dlt.x), math.clamp(math.dot(dlt, dir) / 2, 0, 1));
  }

  public bool _dbgContextComplete;
  public Stopwatch _dbgStopwatch;
  public Task _task;
  private void Update()
  {
    if (_isGenerating)
    {
    }

    if (_dbgContextComplete)
    {
      if (Time.frameCount % 10 == 0)
      {
        Debug.Log($"{_context._log}");
      }

      return;
    }
    if (Time.frameCount < 10)
    {
      Debug.Log(".");
    }
    else if (Time.frameCount == 10)
    {
      Debug.Log($"Starting with params:\n{_simParams}");
      _dbgStopwatch = new Stopwatch();
      _dbgStopwatch.Start();
      _context.Start();
      _task = new Task(() => {
        while (_context.Tick()) ;
      });
      _task.Start();
    }
    else if (_task.IsCompleted)
    {
      _dbgStopwatch.Stop();
      Debug.Log($"[{_dbgStopwatch.Elapsed}]Context{_context._id}:{_context._metrics}: {string.Join(",", _context.Weights)}");
      Debug.Log($"{_context._log}");
      _dbgContextComplete = true;
    }

  }

  public void PreviewBest(float3 state, float2 target, float3[] stateBuffer = null, float4[] actBuffer = null)
  {
  }

}
