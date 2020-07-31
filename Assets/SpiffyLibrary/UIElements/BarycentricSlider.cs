using System;
using System.Collections.Generic;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.UIElements;

namespace SpiffyLibrary.UIElements {
  using static Unity.Mathematics.math;
  public class MeshParts {
    public List<Vertex> vertices = new List<Vertex>();
    public List<int3> triangles = new List<int3>();
    public int IndicesCount => triangles.Count * 3;
    public ushort[] GetIndices() {
      ushort[] result = new ushort[triangles.Count * 3];
      for (var iTri = 0; iTri < triangles.Count; iTri++) { 
        for (int iDim = 0; iDim < 3; iDim++) {
          result[iTri* 3 + iDim] = (ushort)triangles[iTri][iDim];
        }
      }
      return result;
    }
  }
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

    public Action Clicked; 
    public BarycentricSlider()
    {
      generateVisualContent += OnGenerateVisualContent;
      RegisterCallback<MouseDownEvent>(OnClick);
    }

    protected virtual void CalcAdditionalValues() { }

    protected virtual void OnClick(MouseDownEvent evt) {
      var mpos_ls = evt.localMousePosition;
      _mpos_ns = T_NsFromLs(mpos_ls);
      bool valid = _mpos_ns.y > abs(_mpos_ns.x * 2 - 1);
      if (valid) {
        _value_bs = T_BsFromNs(_mpos_ns);
        CalcAdditionalValues();
        Clicked?.Invoke();
        MarkDirtyRepaint();
      }

    }
    protected void AddPoint(MeshParts mp, float2 pnt_ns, float2 size ,Color clr) {
      float2 pnt_ls = T_LsFromNs(pnt_ns);
      int vtxOffset = (int)mp.vertices.Count;
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
        mp.vertices.Add(vtx);
      }
      mp.triangles.Add(vtxOffset + new int3(1, 0, 2));
      mp.triangles.Add(vtxOffset + new int3(3, 0, 4));
      }
    protected virtual void OnGenerateVisualContent(MeshGenerationContext cxt)
    {
      MeshParts mp = new MeshParts();
      { // Background Triangle
        for (int iTriVtx = 0; iTriVtx < 3; iTriVtx++) {
          Vertex vtx = new Vertex();
          vtx.position = (Vector2)T_LsFromNs(TriangleMat[iTriVtx]);
          vtx.tint = _pallet[iTriVtx];
          mp.vertices.Add(vtx);
        }
        mp.triangles.Add(new int3(2,1,0));
      }

      AddPoint(mp,T_NsFromBs(_value_bs), cmin(layout.size / 30), Color.white);


      MeshWriteData meshData = cxt.Allocate(mp.vertices.Count, mp.triangles.Count * 3);
      meshData.SetAllVertices(mp.vertices.ToArray());
      meshData.SetAllIndices(mp.GetIndices());
    }
    
  }
}



