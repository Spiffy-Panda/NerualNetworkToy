using System;
using System.Collections;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.IO;
using System.Linq;
using SpiffyLibrary.MachineLearning;
using UnityEngine;

[CreateAssetMenu(fileName = "PreloadGeneData", menuName = "ScriptableObjects/PreloadGeneData", order = 1)]
public class PreloadGeneData: ScriptableObject {
  [Multiline(50)]
  public string _rawData;


  public ParetoGeneBank.Genome[] GetData() {
    return ParetoGeneBank.ParseYaml(_rawData);
  }
  [ContextMenu("Load Most Recent From Output")]
  public void LoadMostRecentFromOutput() {
    string folder = "OutputData";
    string file = "*_FinalGenes.yaml";
    var potentalFilenames = System.IO.Directory.GetFiles(folder,file).Select(fname => new FileInfo(fname)).OrderByDescending(fi=>fi.CreationTime).ToArray();
    Debug.Log(string.Join("\n", potentalFilenames.Select(fi=>fi.Name+fi.CreationTime)));
    string filename = potentalFilenames[0].FullName;
    string content = File.ReadAllText(filename);
    _rawData = content;
  }
}
