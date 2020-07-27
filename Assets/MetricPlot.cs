using System.Collections;
using System.Collections.Generic;
using Unity.Mathematics;
using UnityEngine;

public class MetricPlot : MonoBehaviour {
  public Mesh _markerMesh = null;
  public Material _material = null;
  public Dictionary<int, float3> _dataPoints = new Dictionary<int, float3>();
  public float3 _minValue;
  public float3 _maxValue;
  public string[] _metrics = new[] {ClosestApproachMetric.MetricName,FinalDistanceMetric.MetricName,OverRotationMetric.MetricName};
  private void Start() {
    FindObjectOfType<AcademyMove>()._NewGeneration += OnGeneration;

  }

  private void OnGeneration() {
    _dataPoints.Clear();
    _minValue = float.PositiveInfinity;
    _maxValue = float.NegativeInfinity;
    var genomes = GeneBankManager.Inst.GetAllGenome();
    foreach (var genome in genomes) {
      int id = genome._id;
      float3 metric = -1;
      for (int iMetric = 0; iMetric < _metrics.Length; iMetric++) {
        metric[iMetric] = genome._metrics[_metrics[iMetric]];
      }
      _minValue = math.min(_minValue, metric);
      _maxValue = math.max(_maxValue, metric);
      _dataPoints.Add(id, metric);
    }
  }

  private void Update() {
    Debug.Assert(_dataPoints != null && _markerMesh != null && _material !=null);
    List<Matrix4x4> TRSs  = new List<Matrix4x4> ();
    foreach (var idPoint in _dataPoints) {
      float3 pnt = math.unlerp(_minValue, _maxValue, idPoint.Value);
      TRSs.Add(transform.localToWorldMatrix*Matrix4x4.TRS(pnt,Quaternion.identity, 0.1f * Vector3.one));
    }
    Graphics.DrawMeshInstanced(_markerMesh,0,_material,TRSs);

  }
}
