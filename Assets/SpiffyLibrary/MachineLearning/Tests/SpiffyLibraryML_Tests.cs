using NUnit.Framework;
using SpiffyLibrary.MachineLearning;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using UnityEngine;
using Debug = UnityEngine.Debug;
using Random = UnityEngine.Random;

namespace SpiffyLibraryML.Tests
{
  public class SpiffyLibraryML_Tests
  {
    [Test]
    public void ParetoGeneBank()
    {
      StringBuilder sb = new StringBuilder();
      ParetoGeneBank geneBank = new ParetoGeneBank();
      geneBank._GeneAddedToPool += (gi) => sb.Append($"[Gene Added: {gi}]");
      geneBank._GeneRemovedFromPool += (gi) => sb.Append($"[Gene Removed: {gi}]");
      Dictionary<string, float> metrics = new Dictionary<string, float>();
      float[] dummyWeights = new float[3];

      void logGenebank_() => sb.AppendLine($"Genebank Contains: {string.Join(", ", geneBank.Frontier.Select(gi => $"<{gi}>"))}");
      void addTestPoint_(float x, float y)
      {
        metrics["x"] = x; metrics["y"] = y;
        sb.Append($"Adding ({x},{y}): ");
        sb.AppendLine($"returned {geneBank.Evaluate(dummyWeights, metrics)}");
      }
      try
      {
        sb.AppendLine($"Double add");
        addTestPoint_(1, 1);
        addTestPoint_(1, 1);
        logGenebank_();
        Assert.AreEqual(geneBank.GenomeCount, 1);

        sb.AppendLine($"Dominate point, start curve");
        addTestPoint_(.5f, .9f);
        logGenebank_();
        Assert.AreEqual(geneBank.GenomeCount, 1);

        sb.AppendLine($"Rest of Init Curve");
        addTestPoint_(.9f, .5f);
        addTestPoint_(.6f, .6f);
        logGenebank_();
        Assert.AreEqual(geneBank.GenomeCount, 3);

        sb.AppendLine($"Domintates all, barely");
        addTestPoint_(.5f, .5f);
        addTestPoint_(.5f, .5f);
        logGenebank_();
        Assert.AreEqual(geneBank.GenomeCount, 1);

        sb.AppendLine($"Domintates all, Clearly.");
        addTestPoint_(.4f, .4f);
        logGenebank_();
        Assert.AreEqual(geneBank.GenomeCount, 1);

        Debug.Log($"Clearing Gene Entries (removed:{geneBank.ClearGeneEntries()})");
        logGenebank_();
        Assert.AreEqual(geneBank.GenomeCount, 0);

        Debug.Log($"Adding Random Points.");
        for (int iRand = 0; iRand < 100; iRand++)
        {
          for(int iWeight =0;iWeight< dummyWeights.Length;iWeight++)
            dummyWeights[iWeight] = Random.value;
          addTestPoint_(Random.value, Random.value);
        }
        logGenebank_();
        Debug.Log(DateTime.Now.ToString("MMMdd_HHmm"));
        System.IO.File.WriteAllText("OutputData/TestML_ParetoGenebank.yaml",geneBank.GetYAML());
      }
      catch (Exception e)
      {
        sb.Insert(0, $"{e.Message}\n");
        sb.AppendLine(e.StackTrace);
        Debug.Assert(false, sb);
      }
      Debug.Log(sb);
    }
  }

}
