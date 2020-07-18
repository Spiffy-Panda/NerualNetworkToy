using System;
using System.Collections;
using System.Collections.Generic;
using System.Data.Common;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.EventSystems;
using UnityEngine.UI;

public class BarycentricSlider : Graphic,IPointerDownHandler {
  public bool _isAcceptingInput;
  private Mesh mesh;
  public Rect rect => rectTransform.rect;
  private float2 point = -100000;
  public Color[] _triangeColors = new Color[3];
  public float3 _value = 1f / 3;
  public event Action Clicked;
  public float _mainPointSize = 1f;

  public List<(float3 baryPoint, Color color, float size)> _extraPoints = new List<(float3 baryPoint, Color color, float size)>();

  public float3 Value {
    get { return _value; }
    set {
      Debug.Assert(math.abs(math.csum(value) - 1) < 0.0001f, "Barycentric Slider: value components must sum to 1.");
      if (math.csum(math.abs(value - _value)) > 0.0001f) {
        _value = value;
        SetVerticesDirty();
      }
    }
  }

  public static float3 ToBarycentric(float3 nonNegVector) {
    Debug.Assert(math.all(nonNegVector >= 0), $"ToBarycentric: nonNegVector must be positive or zero, {nonNegVector}");
    return nonNegVector / math.dot(nonNegVector, 1);
  }

  public float2x3 _triangleRegion_US;
  public float2x3 CalcTriangleUiPos() {
    var result = new float2x3();
    for (int iV = 0; iV < 3; iV++) {
      float theta = ((iV) * Mathf.PI * 2 / 3) + (Mathf.PI / 2);
      Vector2 dir = new Vector2(Mathf.Cos(theta), Mathf.Sin(theta));
      result[iV] = rect.center + Vector2.Scale(dir, rect.size / 2);
    }

    return result;
  }
  public float2 BaryToUI(float3 baryPos) {

    float2 result = 0;
    for (int iV = 0; iV < 3; iV++)
    {
      result += _triangleRegion_US[iV] * baryPos[iV];
    }

    return result;
  }
  protected override void OnPopulateMesh(VertexHelper vh) {
    vh.Clear();
    if (point.x < rect.xMin) {
      point = rect.center;

    }
    int offset = vh.currentVertCount;
    _triangleRegion_US = CalcTriangleUiPos();
    point = BaryToUI(Value);
    for (int i = 0; i < 3; i++) {
      vh.AddVert((Vector2)_triangleRegion_US[i], _triangeColors[i], Vector2.zero);
      
    }
    vh.AddTriangle(offset + 0, offset + 1, offset + 2);
    if (_extraPoints == null)
      _extraPoints = new List<(float3 baryPoint, Color color, float size)>();
    foreach (var pnt in _extraPoints) {

      offset = vh.currentVertCount;
      float2x4 tickVtxBase = new float2x4(1, 0, -1, 0,
                                          0, 1, 0, -1);
      tickVtxBase = math.mul(float2x2.Scale(rect.width, rect.height) / 20, tickVtxBase);
      for (int iVtx = 0; iVtx < 4; iVtx++)
        vh.AddVert((Vector2)(BaryToUI(pnt.baryPoint) + pnt.size * tickVtxBase[iVtx]), pnt.color, Vector2.zero);

      vh.AddTriangle(offset + 0, offset + 1, offset + 2);
      vh.AddTriangle(offset + 2, offset + 3, offset + 0);
    }
    
    {
      offset = vh.currentVertCount;
      float2x4 tickVtxBase  =new float2x4(1,0,-1, 0,
                                          0,1, 0,-1);
      tickVtxBase = math.mul(float2x2.Scale(rect.width, rect.height) / 20, tickVtxBase);
      for(int iVtx =0;iVtx < 4;iVtx++)
        vh.AddVert((Vector2)(point + _mainPointSize * tickVtxBase[iVtx]),Color.white,Vector2.zero);

      vh.AddTriangle(offset + 0, offset + 1, offset + 2);
      vh.AddTriangle(offset + 2, offset + 3, offset + 0);
    }
  }
  
  public void OnPointerDown(PointerEventData eventData) {
    if (!_isAcceptingInput) {

      this.UpdateGeometry();
      return;
    }
    Vector3[] wsArray = new Vector3[4];
    rectTransform.GetWorldCorners(wsArray);
    Rect ws = Rect.MinMaxRect(wsArray[0].x, wsArray[0].y, wsArray[2].x, wsArray[2].y);
    Vector2 ns = Rect.PointToNormalized(ws, (Vector2)eventData.position);
    point = Rect.NormalizedToPoint(rect, ns);
    Vector2 nsCent = ns - Vector2.one / 2;
    float dir = (-Mathf.Atan2(nsCent.y, nsCent.x) + 2*Mathf.PI + Mathf.PI/2)%(Mathf.PI*2);
    float mag = nsCent.magnitude*2;

    {
      Vector3 baryCenter = Vector3.one / 3;
      Vector3 u = Vector3.right - baryCenter;
      Vector3 v = Vector3.Cross(u, baryCenter.normalized);
      Vector3 c = baryCenter + u * Mathf.Cos(dir) * mag + v * Mathf.Sin(dir) * mag;

      Vector3 d = Vector3.Max(Vector3.zero, c);
      d = ToBarycentric(d);
      _value = d;
      Clicked?.Invoke();
    }
    this.UpdateGeometry();
  }
}
