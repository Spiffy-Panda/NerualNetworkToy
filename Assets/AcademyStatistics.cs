using System;
using System.Collections;
using System.Collections.Generic;
using SpiffyLibrary.MachineLearning;
using Unity.Mathematics;
using UnityEngine;
[RequireComponent(typeof(AcademyMove))]
public class AcademyStatistics : MonoBehaviour
{
  private AcademyMove _Academy => GetComponent<AcademyMove>();
  public float3 _smoothedMetrics = float3.zero;
  
  public float _smoothingAlpha = 0.01f;
  public float3 _minMetrics = Single.PositiveInfinity;
  public float3 _maxMetrics = Single.NegativeInfinity;

  public int _generationCount = 0;

  public List<float3> _bestHistory = new List<float3>();
  public string _AgentJson = "";
  // Start is called before the first frame update
  void Start() {
    //_Academy.RatingFinished += OnRatingFinished;
    //_Academy.NewBestFound += OnNewBestFound;
    _bestHistory = new List<float3>();
  }

  private void OnNewBestFound(float3 metrics, float cost)
  {
    _bestHistory.Add(metrics);

    //var mlp = _Academy._BestBrain as MultiLayerPerception;
    //_AgentJson = JsonUtility.ToJson(mlp);
  }

  // Update is called once per frame
  void OnRatingFinished(float3 metrics, float cost) {
    _generationCount++;
    _minMetrics = math.min(_minMetrics, metrics);
    _maxMetrics = math.max(_maxMetrics, metrics);
    _smoothedMetrics = _smoothedMetrics * (1 - _smoothingAlpha) + metrics* _smoothingAlpha;
  }

  
}
