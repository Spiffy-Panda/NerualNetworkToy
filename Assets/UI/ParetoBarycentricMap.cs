﻿using System.Collections;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using SpiffyLibrary.UIElements;
using Unity.Mathematics;
using UnityEngine.UIElements;
using Random = UnityEngine.Random;

namespace  ProjectUI {
  using BarycentricSlider = SpiffyLibrary.UIElements.BarycentricSlider;
  public class ParetoBarycentricMap : BarycentricSlider
  {
    public new class UxmlFactory : UxmlFactory<ParetoBarycentricMap, Image.UxmlTraits>{}

    public int Value_ID { get; private set; } = -1;
    private int _subDiv =5;
    Dictionary<int, float3> metricVecLookup = new Dictionary<int, float3>();
    int GetClosestGenome(float3 metricBS) => metricVecLookup.Aggregate((kvCur, kvNxt) =>
        (math.dot(kvCur.Value, metricBS) < math.dot(kvNxt.Value, metricBS)) ? kvCur : kvNxt
      ).Key;


    protected override void CalcAdditionalValues() {
      Value_ID = GetClosestGenome(_value_bs);
    }

    protected override void OnGenerateVisualContent(MeshGenerationContext cxt) {

      List<Vertex> vertices = new List<Vertex>();
      List<ushort> indices = new List<ushort>();
      List<int3> curTri = new List<int3>();

      Color[] clrByDepth = new Color[_subDiv + 1];
      for (int iClr = 0; iClr < clrByDepth.Length; iClr++) {
        float t = iClr / (float)(clrByDepth.Length);
        clrByDepth[iClr] = Color.HSVToRGB(t, 1, 1);
      }
      for (int iTriVtx = 0; iTriVtx < 3; iTriVtx++)
      {
        Vertex vtx = new Vertex();
        vtx.position = (Vector2)T_LsFromNs(TriangleMat[iTriVtx]);
        vtx.tint = clrByDepth[_subDiv]; // it circular
        vertices.Add(vtx);
      }
      curTri.Add(new int3(2,1,0));

      for (int iDiv =0;iDiv < _subDiv;iDiv++)
      {
        List<int3> nxtTri = new List<int3>();
        foreach (var outerTri in curTri) {
          int vtxOffset = vertices.Count;
          for (int iTriVtx = 0; iTriVtx < 3; iTriVtx++) {
            var pntLS = Vector3.zero;
            for (int iEdge = 0; iEdge < 3; iEdge++) {
              pntLS += (iEdge == iTriVtx) ? Vector3.zero: vertices[outerTri[iEdge]].position;

            }
            pntLS /= 2;
            Vertex vtx = new Vertex();
            vtx.position = pntLS;
            vtx.tint = clrByDepth[iDiv]; 
            vertices.Add(vtx);
          }

          for (int iTriVtx = 0; iTriVtx < 3; iTriVtx++) {
            // Broad cast it to xyz;
            int3 newTri = outerTri[iTriVtx];
            if (iTriVtx == 0)
              newTri.yz = vtxOffset + new int2(2, 1);
            else if (iTriVtx == 1) 
              newTri.xz = vtxOffset + new int2(2, 0);
            else if (iTriVtx == 2) 
              newTri.xy = vtxOffset + new int2(1, 0);
            nxtTri.Add(newTri);
          }
          nxtTri.Add(vtxOffset + new int3(0,1,2));

        }
        curTri = nxtTri;
      }
      foreach (var tri in curTri)
      {
        indices.AddRange(new ushort[] { (ushort)tri.x, (ushort)tri.y, (ushort)tri.z });
        //indices.AddRange(new ushort[] { (ushort)tri.z, (ushort)tri.y, (ushort)tri.x });
      }

      if (GeneBankManager.Inst && GeneBankManager.Inst.GenomeCount > 0) 
        ApplyGenomeColors(vertices);

      AddPoint(vertices, indices, T_NsFromBs(_value_bs), math.cmin(layout.size / 30), Color.white);

      MeshWriteData meshData = cxt.Allocate(vertices.Count, indices.Count);
      meshData.SetAllVertices(vertices.ToArray());
      meshData.SetAllIndices(indices.ToArray());
    }

    public string[] _metricNames = new[] {
      ClosestApproachMetric.MetricName,
      FinalDistanceMetric.MetricName,
      OverRotationMetric.MetricName
    };
    public void ApplyGenomeColors(List<Vertex> vertices) {
      var genomes = GeneBankManager.Inst.GetAllGenome();
      Dictionary<int, Color> clrLookup = new Dictionary<int, Color>();
      metricVecLookup.Clear();
      float hue = 0;
      float3 minMetric = float.PositiveInfinity;
      float3 maxMetric = float.NegativeInfinity;
      foreach (var genome in genomes) {
        clrLookup[genome._id] = Color.HSVToRGB(hue%1f,1,1);
        hue += 0.618f;
        float3 metricVector = new float3();
        for (int iMetric = 0; iMetric < _metricNames.Length; iMetric++)
          metricVector[iMetric] = genome._metrics[_metricNames[iMetric]];
        metricVecLookup[genome._id] = metricVector;
        minMetric = math.min(minMetric, metricVecLookup[genome._id]);
        maxMetric = math.max(maxMetric, metricVecLookup[genome._id]);
      }
      
      foreach (var key in metricVecLookup.Keys.ToArray()) {
        float3 val = math.unlerp(minMetric, maxMetric, metricVecLookup[key]);
        metricVecLookup[key] = val;
      }
      for (int iVtx = 0; iVtx < vertices.Count; iVtx++) {
        var vtx= vertices[iVtx];
        var pntBS = T_BsFromNs(T_NsFromLs(((float3)vtx.position).xy));

        var closestID = GetClosestGenome(pntBS);
        vtx.tint = clrLookup[closestID];
        vertices[iVtx] = vtx;
      }
    }
  }
}