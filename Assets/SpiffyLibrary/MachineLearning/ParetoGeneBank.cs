using System;
using System.Collections;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Linq;
using System.Text;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.PlayerLoop;
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

    public class Genome
    {
      private static int _idAllocator = 0;
      public readonly int _id;
      public readonly ReadOnlyCollection<float> _weights;
      public readonly ReadOnlyDictionary<string, float> _metrics;

      public Genome(float[] weights, Dictionary<string, float> metrics)
      {
        _id = _idAllocator++;
        _weights = new ReadOnlyCollection<float>(weights.ToArray());
        _metrics = new ReadOnlyDictionary<string, float>(metrics.ToDictionary(kv => kv.Key, kv => kv.Value));
      }

      public string GetYamlEntry()
      {
        var metricEntries = _metrics.Select(kv => $"\t\t{kv.Key}: {kv.Value:R}");
        return ($"- OriginalRunID: {_id}\n" +
               $"\tMetric:\n{string.Join("\n", metricEntries)}\n" +
               $"\tWeights: [{string.Join(", ", _weights.Select(f=>f.ToString("R")))}]").Replace("\t", SoftTab);
      }

      public static Genome ParseYaml(IReadOnlyCollection<string> yaml) {
        bool inMetrics = false;
        int id = -1;
        float[] weights = null;
        Dictionary<string,float> metrics = new Dictionary<string, float>();
        foreach (string line in yaml) {

          if (line.Contains("OriginalRunID")) {
            Debug.Assert(int.TryParse(line.Split(':')[1],out id));
            inMetrics = false;
          }

          if (line.Contains("Weights")) {
            string wArray = line.Split(':')[1].Trim(" []".ToCharArray());
            weights = wArray.Split(',').Select(str => float.Parse(str.Trim())).ToArray();
            inMetrics = false;
          }

          if (inMetrics) {
            string[] kv = line.Trim().Split(':');
            metrics[kv[0]] = float.Parse(kv[1]);
          }
          if (line.Contains("Metric")) {
            inMetrics = true;
          }
        }
        Genome result = new Genome(weights,metrics);
        return result;
      }
      public override string ToString() => $"{_id}: {string.Join(",", _metrics.Select(kv=>$"({kv.Key},{kv.Value})"))}";
    }

    public Dictionary<string, MetricRecord> _metricRecords = new Dictionary<string, MetricRecord>();
    private List<Genome> _frontier = new List<Genome>();
    public ReadOnlyCollection<Genome> Frontier => new ReadOnlyCollection<Genome>(_frontier);
    public int GenomeCount => _frontier.Count;
    public event Action<Genome> _GeneRemovedFromPool;
    public event Action<Genome> _GeneAddedToPool;
    public const string SoftTab = "  ";

    public bool Evaluate(Genome gi) => Evaluate(gi._weights.ToArray(), new Dictionary<string, float>(gi._metrics)); 
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
      var newGene = new Genome(weights, metrics);
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

    public Genome GetRandomGenome() => _frontier[Random.Range(0, _frontier.Count)];

    public string GetPythonDict() {

      string pythonDictEntry(ParetoGeneBank.Genome gi)
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

    public static Genome[] ParseYaml(string yaml) {
      List<Genome> result = new List<Genome>();
      int start= yaml.IndexOf("\n", StringComparison.Ordinal)+1;
      string giArrayYml = yaml.Substring(start);
      string[] lines = giArrayYml.Split('\n');
      List<string> buffer = new List<string>();
      for (int iLine = 0; iLine < lines.Length; iLine++) {
        if (lines[iLine].StartsWith("-") && buffer.Count >0)
        {
          result.Add(Genome.ParseYaml(buffer));
          buffer.Clear();
        }

        buffer.Add(lines[iLine]);
      }

      if (buffer.Count > 0)
        result.Add(Genome.ParseYaml(buffer));

      return result.ToArray();
    }
  }
}
