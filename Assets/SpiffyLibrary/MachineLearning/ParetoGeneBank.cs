using System;
using System.Collections;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Linq;
using Unity.Mathematics;
using UnityEngine;
using Random = UnityEngine.Random;

namespace SpiffyLibrary.MachineLearning {


  public abstract class MetricInfo
  {
    public abstract string Name { get; }
    public float TotalValue { get; set; }
    // TODO: make the state and action not relay on order
    public abstract void EvalIteractionTick(float3 state, float2 action);

  }
  public class ParetoGeneBank
  {

    public struct MetricRecord
    {
      public readonly string MetricName;
      public float MinKnownValue;
      public float MaxKnownValue;
    }

    public class GeneInfo
    {
      private static int _idAllocator = 0;
      public readonly int _id;
      public readonly ReadOnlyCollection<float> _weights;
      public readonly ReadOnlyDictionary<string, float> _metrics;

      public GeneInfo(float[] weights, Dictionary<string, float> metrics)
      {
        _id = _idAllocator++;
        _weights = new ReadOnlyCollection<float>(weights.ToArray());
        _metrics = new ReadOnlyDictionary<string, float>(metrics.ToDictionary(kv => kv.Key, kv => kv.Value));
      }

      public string GetYamlEntry()
      {
        var metricEntries = _metrics.Select(kv => $"\t\t{kv.Key}: {kv.Value}");
        return ($"- OriginalRunID: {_id}\n" +
               $"\tMetric:\n{string.Join("\n", metricEntries)}\n" +
               $"\tWeights: [{string.Join(", ", _weights)}]").Replace("\t", SoftTab);
      }
      public override string ToString() => $"{_id}: {string.Join(",", _metrics.Select(kv=>$"({kv.Key},{kv.Value})"))}";
    }

    public Dictionary<string, MetricRecord> _metricRecords = new Dictionary<string, MetricRecord>();
    private List<GeneInfo> _frontier = new List<GeneInfo>();
    public ReadOnlyCollection<GeneInfo> Frontier => new ReadOnlyCollection<GeneInfo>(_frontier);
    public int GenomeCount => _frontier.Count;
    public event Action<GeneInfo> _GeneRemovedFromPool;
    public event Action<GeneInfo> _GeneAddedToPool;
    public const string SoftTab = "  ";

    public bool Evaluate(float[] weights, Dictionary<string, float> metrics)
    {
      bool shouldAdd = true;
      foreach (var geneInfo in _frontier)
      {
        bool dominatesNew = true;
        foreach (var kvMetric in geneInfo._metrics)
        {
          if (metrics[kvMetric.Key] < kvMetric.Value)
          {
            dominatesNew = false;
            break;
          }
        }

        if (dominatesNew)
        {
          shouldAdd = false;
          break;
        }
      }

      if (!shouldAdd)
        return false;
      var newGene = new GeneInfo(weights, metrics);
      for (var iGene = _frontier.Count - 1; iGene >= 0; iGene--)
      {
        bool dominatesOld = true;
        foreach (var kvMetric in _frontier[iGene]._metrics)
        {
          // reverse of last time
          if (metrics[kvMetric.Key] > kvMetric.Value)
          {
            dominatesOld = false;
            //Debug.Log($"(n{newGene._id} vs. o{_frontier[iGene]._id}) kept because {kvMetric.Key}: {metrics[kvMetric.Key]} < {kvMetric.Value}");
            break;
          }
        }
        if (dominatesOld)
        {
          var removed = _frontier[iGene];
          _frontier.RemoveAt(iGene);
          _GeneRemovedFromPool?.Invoke(removed);
        }
      }
      _frontier.Add(newGene);
      _GeneAddedToPool?.Invoke(_frontier[_frontier.Count - 1]);
      return true;
    }

    public GeneInfo GetRandomGenome() => _frontier[Random.Range(0, _frontier.Count)];

    public string GetPythonDict() {

      string pythonDictEntry(ParetoGeneBank.GeneInfo gi)
      {
        var metricEntries = gi._metrics.Select(kv => $"\"{kv.Key}\":{kv.Value}");
        return $"\"{gi._id}\": {{{string.Join(",", metricEntries)}}}";
      }
      return "{\n" + string.Join(",\n", _frontier.Select(pythonDictEntry)) + "}";
    }

    // Yaml is more human readable then json, so quick custom output.
    public string GetYAML() {
     
      return "data:\n"+ string.Join("\n", _frontier.Select(gi=>gi.GetYamlEntry())).Replace("\t",SoftTab);
    }

    public int ClearGeneEntries() {
      int result = _frontier.Count;
      _frontier.Clear();
      return result;
    }
  }
}
