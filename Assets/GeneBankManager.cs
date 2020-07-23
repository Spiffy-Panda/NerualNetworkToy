using SpiffyLibrary.MachineLearning;
using System;
using System.Collections.Generic;
using System.IO;
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

  private void LogAdd(ParetoGeneBank.GeneInfo gi) => _addLog.WriteLine(gi.GetYamlEntry());
  private void LogRemove(ParetoGeneBank.GeneInfo gi) => _remLog.WriteLine(gi.GetYamlEntry());

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

  }

  public bool Evaluate(float[] weights, Dictionary<string, float> metrics)
  {
    bool result = _geneBank.Evaluate(weights, metrics);
    return result;
  }

  public ParetoGeneBank.GeneInfo GetRandomGenome()
  {
    return _geneBank.GetRandomGenome();
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
    File.WriteAllText($"OutputData/{runTag}_FinalGenes.yaml", ymlFrountier);
  }

}
