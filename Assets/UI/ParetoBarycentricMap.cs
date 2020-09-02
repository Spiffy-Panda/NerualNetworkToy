using System.Collections;
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
    private int _subDiv =7;
    Dictionary<int, float3> metricVecLookup = new Dictionary<int, float3>();
    int GetClosestGenome(float3 metricBS) => metricVecLookup.Aggregate((kvCur, kvNxt) =>
        (math.dot(kvCur.Value, metricBS) < math.dot(kvNxt.Value, metricBS)) ? kvCur : kvNxt
      ).Key;


    protected override void CalcAdditionalValues() {
      Value_ID = GetClosestGenome(_value_bs);
    }

    protected override void OnGenerateVisualContent(MeshGenerationContext cxt) {
      var mp = new MeshParts();
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
        mp.vertices.Add(vtx);
      }
      curTri.Add(new int3(2,1,0));

      for (int iDiv =0;iDiv < _subDiv;iDiv++)
      {
        List<int3> nxtTri = new List<int3>();
        foreach (var outerTri in curTri) {
          int vtxOffset = mp.vertices.Count;
          for (int iTriVtx = 0; iTriVtx < 3; iTriVtx++) {
            var pntLS = Vector3.zero;
            for (int iEdge = 0; iEdge < 3; iEdge++) {
              pntLS += (iEdge == iTriVtx) ? Vector3.zero: mp.vertices[outerTri[iEdge]].position;

            }
            pntLS /= 2;
            Vertex vtx = new Vertex();
            vtx.position = pntLS;
            vtx.tint = clrByDepth[iDiv];
            mp.vertices.Add(vtx);
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
        mp.triangles.Add(tri);
        //indices.AddRange(new ushort[] { (ushort)tri.z, (ushort)tri.y, (ushort)tri.x });
      }

      if (GeneBankManager.Inst && GeneBankManager.Inst.GenomeCount > 0) 
        ApplyGenomeColors(mp.vertices);

      AddPoint(mp, T_NsFromBs(_value_bs), math.cmin(layout.size / 30), Color.white);
      
      MeshWriteData meshData = cxt.Allocate(mp.vertices.Count, mp.triangles.Count * 3);
      meshData.SetAllVertices(mp.vertices.ToArray());
      meshData.SetAllIndices(mp.GetIndices());
    }

    public string[] _metricNames = new[] {
      ClosestApproachMetric.MetricName,
      FinalDistanceMetric.MetricName,
      OverRotationMetric.MetricName
    };
    Dictionary<int, Color> clrLookup = new Dictionary<int, Color>();
    private const float _invGoldenRatio = 0.618034f;
    public void ApplyGenomeColors(List<Vertex> vertices) {
      var genomes = GeneBankManager.Inst.GetAllGenome();
      metricVecLookup.Clear();
      List<int> freeKeys = new List<int>();
      HashSet<int> curKeys = new HashSet<int>(genomes.Select(gi => gi._id));
      foreach (var kv in clrLookup) {
        if(!curKeys.Contains(kv.Key))
          freeKeys.Add(kv.Key);
      }

      float3 minMetric = float.PositiveInfinity;
      float3 maxMetric = float.NegativeInfinity;
      foreach (var genome in genomes) {

        if (!clrLookup.ContainsKey(genome._id)) {
          if (freeKeys.Count > 0) {
            int oldKey = freeKeys[0];
            clrLookup[genome._id] = clrLookup[oldKey];
            clrLookup.Remove(oldKey);
            freeKeys.RemoveAt(0);
          } else {
            clrLookup[genome._id] = Color.HSVToRGB((clrLookup.Count* _invGoldenRatio )% 1f,1,1);
          }
        }
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
