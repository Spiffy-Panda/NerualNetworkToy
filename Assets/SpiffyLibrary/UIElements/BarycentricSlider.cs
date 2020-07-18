using System;
using System.Collections.Generic;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.UIElements;

namespace SpiffyLibrary.UIElements {
using static Unity.Mathematics.math;
  public class BarycentricSlider : VisualElement {
    public new class UxmlFactory : UxmlFactory<BarycentricSlider, UxmlTraits> {
    }

    public float3 Value {
      get => _value_bs;
      set {
        Debug.Assert(Mathf.Approximately(csum(value), 1), "Barycentric: Sum of components must equal 1");
        _value_bs = value;
      }
    }

    public readonly float2x3 TriangleMat = new float2x3(new float2(0.5f, 0), new float2(0, 1), new float2(1, 1));

    public float2 _mpos_ns = 0.5f;
    public float3 _value_bs = 1f / 3;
    public Color[] _pallet = new Color[4] {
      TurboColorMap.Map(0.1f),
      TurboColorMap.Map(0.5f),
      TurboColorMap.Map(0.9f),
      Color.white
    };


    public float2 T_LsFromNs(float2 ns) => ns * (float2)layout.size;
    public float2 T_NsFromLs(float2 ls) => ls / (float2)layout.size;
    public float2 T_NsFromBs(float3 bs) => mul(TriangleMat, bs);

    /// <summary>
    /// Barycentric Space from Normal Space
    /// </summary>
    /// <param name="ns">normalized space</param>
    /// <returns>ns in Barycentric space</returns>

    public float3 T_BsFromNs(float2 ns)
    {
      // We contruct a matrix using the 3 points of the triangle padded by z=1.
      float3x3 mat = new float3x3();
      mat.c0 = new float3(TriangleMat.c0, 1);
      mat.c1 = new float3(TriangleMat.c1, 1);
      mat.c2 = new float3(TriangleMat.c2, 1);

      // We pad ns with a z=1 so we get the 3d point that component sum to 1.
      float3 ns1 = new float3(ns, 1);

      // This is the inverse of the mul used in NsFromBS, which can be seen as a weighted average.
      return  mul(inverse(mat), ns1);
    }
    public BarycentricSlider()
    {
      generateVisualContent += OnGenerateVisualContent;
      RegisterCallback<MouseDownEvent>(OnClick);
    }

    void OnClick(MouseDownEvent evt) {
      var mpos_ls = evt.localMousePosition;
      _mpos_ns = T_NsFromLs(mpos_ls);
      bool valid = _mpos_ns.y > abs(_mpos_ns.x * 2 - 1);
      if (valid) {
        _value_bs = T_BsFromNs(_mpos_ns);
        MarkDirtyRepaint();
      }

    }
    void OnGenerateVisualContent(MeshGenerationContext cxt)
    {
      List<Vertex> vertices = new List<Vertex>();
      List<ushort> indices = new List<ushort>();

      { // Background Triangle
        for (int iTriVtx = 0; iTriVtx < 3; iTriVtx++) {
          Vertex vtx = new Vertex();
          vtx.position = (Vector2)T_LsFromNs(TriangleMat[iTriVtx]);
          vtx.tint = _pallet[iTriVtx];
          vertices.Add(vtx);
        }
        indices.AddRange(new ushort[]{ 2,1,0});
      }


      void AddPoint(float2 pnt_ns, float2 size ,Color clr) {
        float2 pnt_ls = T_LsFromNs(pnt_ns);
        int vtxOffset = (int)vertices.Count;
        float2[] pos = new float2[] {
          pnt_ls,
          pnt_ls + new float2(-1,-1) * size,
          pnt_ls + new float2(-1, 1) * size,
          pnt_ls + new float2( 1, 1) * size,
          pnt_ls + new float2( 1,-1) * size,
        };
        for (int iPointVtx = 0; iPointVtx < pos.Length; iPointVtx++)
        {
          Vertex vtx = new Vertex();
          vtx.tint = clr;
          vtx.position = (Vector2)pos[iPointVtx];
          vertices.Add(vtx);
        }

        indices.Add((ushort)(vtxOffset + 1));
        indices.Add((ushort)(vtxOffset + 0));
        indices.Add((ushort)(vtxOffset + 2));

        indices.Add((ushort)(vtxOffset + 3));
        indices.Add((ushort)(vtxOffset + 0));
        indices.Add((ushort)(vtxOffset + 4));
      }
      AddPoint(T_NsFromBs(_value_bs), cmin(layout.size / 30), Color.white);

      MeshWriteData meshData = cxt.Allocate(vertices.Count, indices.Count);
      meshData.SetAllVertices(vertices.ToArray());
      meshData.SetAllIndices(indices.ToArray());
    }
    
  }
}



