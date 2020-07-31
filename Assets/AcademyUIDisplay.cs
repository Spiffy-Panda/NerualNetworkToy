using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.Events;
using UnityEngine.UI;
using Random = Unity.Mathematics.Random;

[RequireComponent(typeof(AcademyStatistics),typeof(AcademyMove))]
public class AcademyUIDisplay : MonoBehaviour {
  private AcademyStatistics AcadStats => GetComponent<AcademyStatistics>();
  private AcademyMove Academy => GetComponent<AcademyMove>();
  private EvaluationWeight EvalWeights => GetComponent<EvaluationWeight>();
  public Text _textDisplay;
  public BarycentricSlider _outBestsMetrics;
  public Font _font;
  public bool _bestHistroyDirty = false;
  public string[] _metricNames = new[] {
    "Closet Point",
    "Stopping",
    "Turning",
  };
  

  // Start is called before the first frame update
  void Start()
  {
    var rng = new Random((uint)UnityEngine.Random.Range(0, int.MaxValue));
    string[] allFonts = Font.GetOSInstalledFontNames();
    _font = Font.CreateDynamicFontFromOSFont("Trebuchet MS",6);
    _textDisplay.font = _font;
    //Academy.NewBestFound += (m,c)=>_bestHistroyDirty = true;
    
  }

  // Update is called once per frame
  void Update() {
    if (_bestHistroyDirty) {
      int cnt = 0;
        _outBestsMetrics._extraPoints.Clear();
      foreach (var metrics in AcadStats._bestHistory)
      {
        float3 rawMetrics = metrics;
        float3 invNormMetrics = 1 - math.unlerp(AcadStats._minMetrics, AcadStats._maxMetrics, rawMetrics);
        Color c = Color.HSVToRGB(0.15f * cnt++, 1, 1);
        bool badInput = false;
        for(int iDim =0;iDim < 3; iDim++) {
          badInput |= (float.IsNaN(invNormMetrics[iDim]) || float.IsInfinity(invNormMetrics[iDim])) ;
        }

        if (badInput) {
          Debug.LogError($"Best History had NAN: {Time.frameCount}({Time.timeSinceLevelLoad}): \n" +
                         $" Raw: {rawMetrics}\n" +
                         $" InvNorm: {invNormMetrics}\n" +
                         $" Min: {AcadStats._minMetrics} \n " +
                         $" Max: {AcadStats._maxMetrics} \n ");
          continue;
        }
          float3 bPos = BarycentricSlider.ToBarycentric(invNormMetrics);
        _outBestsMetrics._extraPoints.Add((bPos, c, .5f));
      }
    }

    StringBuilder sb = new StringBuilder();
    sb.AppendLine($"Move to point Academy");
    sb.AppendLine($"Agents Tested: {AcadStats._generationCount}");

    if (EvalWeights) {
      float3 w = EvalWeights.Weights;
      bool3 favoring = w > 0.5f;
      Debug.Assert(_metricNames.Length == 3);
      if (math.any(favoring)) {
        for (int i = 0; i < _metricNames.Length; i++) {
          if (favoring[i])
          {
            sb.AppendLine($"Favoring Metric: {_metricNames[i]}");
          }
        }
      } else {
        sb.AppendLine($"Favoring Metric: None");
      }
      
      sb.AppendLine($"Metric Weights: {w.x:0.00}, {w.y:0.00}, {w.z:0.00}");

      float3 rMetrics = 0;//Academy._bestMetrics;

      sb.AppendLine($"Best Metrics:");

      for (int i = 0; i < _metricNames.Length; i++) {

        sb.AppendLine($"\t{_metricNames[i], -20} {rMetrics[i],10:0.00000}");
      }

      float3 mi = AcadStats._minMetrics;
      sb.AppendLine($"Min Metric: {mi.x:0.00}, {mi.y:0.00}, {mi.z:0.00}");
      float3 ma = AcadStats._maxMetrics;
      sb.AppendLine($"Max Metric: {ma.x:0.00}, {ma.y:0.00}, {ma.z:0.00}");



    }

    _textDisplay.text = sb.ToString();

    if(_outBestsMetrics) {
      float3 rawMetrics = 0;//Academy._bestMetrics;
      float3 invNormMetrics = 1- math.unlerp(AcadStats._minMetrics, AcadStats._maxMetrics, rawMetrics);
      float3 bryMetrics = invNormMetrics / Vector3.Dot(invNormMetrics, Vector3.one);
      if (!float.IsNaN(bryMetrics.x)) {

        _outBestsMetrics.Value = bryMetrics;
      }
    }
  }
}
