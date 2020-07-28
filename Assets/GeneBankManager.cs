using SpiffyLibrary.MachineLearning;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Google.Protobuf.Reflection;
using UnityEngine;
// Based on a Pareto Frontier.
// Which is the collection of points that 
// No metric can be better off without making at least one metric better.
// See:
// https://en.wikipedia.org/wiki/Pareto_efficiency
public class GeneBankManager : MonoBehaviour
{
  private ParetoGeneBank _geneBank = new ParetoGeneBank();
  public static GeneBankManager _inst;
  public bool _logChangesToFile = true;
  public bool _logChangesToConsole = false;
  public StreamWriter _addLog;
  public StreamWriter _remLog;
  public static GeneBankManager Inst
  {
    get
    {
      if (_inst == null)
      {
        _inst = FindObjectOfType<GeneBankManager>();
      }

      return _inst;
    }
  }
  private string runTag = null;
  public PreloadGeneData _preloadData;
  private void LogAdd(ParetoGeneBank.Genome gi) => _addLog.WriteLine(gi.GetYamlEntry());
  private void LogRemove(ParetoGeneBank.Genome gi) => _remLog.WriteLine(gi.GetYamlEntry());
  public int _dbgGenomeCount = -1;
  public int GenomeCount => _geneBank.GenomeCount;
  public bool LoadDataOnStart = false;
  public void Start()
  {
    runTag = $"Run{DateTime.Now:MMMdd_HHmm}";
    if (_logChangesToFile)
    {
      _addLog = new StreamWriter($"OutputData/{runTag}_Added.yaml");
      _remLog = new StreamWriter($"OutputData/{runTag}_Removed.yaml");
      _addLog.WriteLine("data:");
      _remLog.WriteLine("data:");
      _geneBank._GeneAddedToPool     += LogAdd;
      _geneBank._GeneRemovedFromPool += LogRemove;
    }

    if (_logChangesToConsole)
    {
      _geneBank._GeneAddedToPool += (gi) => Debug.Log($"+{gi}");
      _geneBank._GeneRemovedFromPool += (gi) => Debug.Log($"-{gi}");
    }

    _geneBank._GeneAddedToPool += (gi) => _dbgGenomeCount = _geneBank.GenomeCount;
    _geneBank._GeneRemovedFromPool += (gi) => _dbgGenomeCount = _geneBank.GenomeCount;

    if (_preloadData != null && LoadDataOnStart)
    {
      var preloadGenomes = _preloadData.GetData();
      foreach (var gi in _preloadData.GetData()) {
        _geneBank.Evaluate(gi);
      }
      Debug.Log($"Done Preloaded. Genebank Size: {_geneBank.GenomeCount} of {preloadGenomes.Length} from preload.");
    }
  }

  public bool Evaluate(float[] weights, Dictionary<string, float> metrics)
  {
    bool result = _geneBank.Evaluate(weights, metrics);
    return result;
  }

  public ParetoGeneBank.Genome GetRandomGenome()
  {
    return _geneBank.GetRandomGenome();
  }
  public ParetoGeneBank.Genome GetGenomeByID(int id)
  {
    var result = _geneBank.Frontier.FirstOrDefault(gi => gi._id == id);
    return result;
  }

  public ParetoGeneBank.Genome GetMinMetricGenome(string metricName, int idx) {
    return _geneBank.Frontier.OrderBy(gi => gi._metrics[metricName]).ElementAt(idx);
  }


  public ParetoGeneBank.Genome GetMinMetricGenome(string metricName)
  {
    return _geneBank.Frontier.Aggregate((giA, giB) => giA._metrics[metricName] < giB._metrics[metricName] ? giA : giB);
  }

  public ParetoGeneBank.Genome[] GetAllGenome()
  {
    return _geneBank.Frontier.ToArray();
  }

  private void OnDisable()
  {
    if (_addLog != null)
    {
      _geneBank._GeneAddedToPool -= LogAdd;
      _addLog.Close();
      _addLog = null;
    }
    if (_remLog != null)
    {
      _geneBank._GeneRemovedFromPool -= LogRemove;
      _remLog.Close();
      _remLog = null;
    }
    string ymlFrountier = _geneBank.GetYAML();
    Debug.Log(ymlFrountier);

    if (_logChangesToFile)
      File.WriteAllText($"OutputData/{runTag}_FinalGenes.yaml", ymlFrountier);
  }

}
