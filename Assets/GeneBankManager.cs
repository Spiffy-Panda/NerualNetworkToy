using System;
using System.Collections;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Data;
using System.Linq;
using SpiffyLibrary.MachineLearning;
using UnityEditor.Build.Reporting;
using UnityEngine;
// Based on a Pareto Frontier.
// Which is the collection of points that 
// No metric can be better off without making at least one metric better.
// See:
// https://en.wikipedia.org/wiki/Pareto_efficiency
public class GeneBankManager : MonoBehaviour {
  ParetoGeneBank _geneBank = new ParetoGeneBank();
  public static GeneBankManager _inst;

  public static GeneBankManager Inst {
    get {
      if (_inst == null)
        _inst = FindObjectOfType<GeneBankManager>();
      return _inst;
    }
  }

  public void Start() {
    _geneBank._GeneAddedToPool += (gi) => Debug.Log($"+{gi}");
    _geneBank._GeneRemovedFromPool += (gi) => Debug.Log($"-{gi}");
  }

  public bool Evaluate(float[] weights, Dictionary<string, float> metrics)
  {
    bool result = _geneBank.Evaluate(weights, metrics);
    return result;
  }

  public ParetoGeneBank.GeneInfo GetRandomGenome() => _geneBank.GetRandomGenome();

  private void OnDisable() {
    string ymlFrountier = _geneBank.GetYAML();
    Debug.Log(ymlFrountier);
    System.IO.File.WriteAllText($"OutputData/Run_{DateTime.Now:MMMdd_HHmm}_Genes.yaml", ymlFrountier);
  }

}
