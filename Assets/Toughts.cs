using System;
using System.Collections;
using System.Collections.Generic;
using System.Transactions;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.UI;
using UnityEngine.UIElements;
using Random = Unity.Mathematics.Random;

public class Toughts : Graphic {
  private MoveToPointPreview _preview;
  private AcademyMove _academy;

  protected override void OnPopulateMesh(VertexHelper vh) {
    base.OnPopulateMesh(vh);
    vh.Clear();
    float2x4 rBase = new float2x4(0,1,1,0,
                                    0,0,1,1);
    float2 rectScale = rectTransform.rect.size;
    float2 rectPos = rectTransform.rect.position;
    float aspect = rectTransform.rect.width / rectTransform.rect.height;
    Func<float2,float2> T = (vtx) => (rectScale * vtx + rectPos);
    void macroAddRect(float2 pos, float2 size, Color clr)
    {
      int vtxc= vh.currentVertCount;
      for (int iVtx = 0; iVtx < 4; iVtx++)
      {
        vh.AddVert((Vector2)T( size*rBase[iVtx] + pos), clr, Vector2.zero);

      }
      vh.AddTriangle(vtxc+0, vtxc+1, vtxc+ 2);
      vh.AddTriangle(vtxc+3, vtxc+0, vtxc+2);
    }

    int hiddenLayerCount = 1;

    float2 nodeSize = new float2(0.2f/aspect, 0.2f);
    float2 macGetNodePos(int layer, int idx) {
      float2 usable = 1 - nodeSize;
      float2 nPos;
      nPos.x = layer / (2f + hiddenLayerCount - 1); //hidden + input and output
      nPos.y = 1- idx / 3f;
      return usable * nPos;
    }

    // actually turbo, but jet is a common color map thats name stands out
    Func<float, Color> jet = TurboColorMap.Map;
    macroAddRect(0, 1, Color.gray);

    return; //TODO: when MLP_Tensor workes finish port function

    var mlp = _TestMLP;
    for (int iINode = 0; iINode < 4; iINode++) {
      macroAddRect(macGetNodePos(0, iINode), nodeSize, jet(math.unlerp(obsMin[iINode], obsMax[iINode], _observe[0])));
    }

    // TODO: Replace with new hidden value check.
    float4 hv = 1;// mlp.GetHiddenValues(_observe, false)[0];
    float4 hva = 0;// mlp.GetHiddenValues(_observe, true)[0];
    hv = math.unlerp(hiddenBounds.c0, hiddenBounds.c1, hv);
    for (int iDim = 0; iDim < 4; iDim++)
    {
      macroAddRect(macGetNodePos(1, iDim), nodeSize, jet(hv[iDim]));
      macroAddRect(macGetNodePos(1, iDim)+ nodeSize / 4, nodeSize/2, jet(hva[iDim]));
    }

    float4 ov = math.unlerp(outBounds.c0, outBounds.c1, mlp.Execute(_observe));
    for (int iONode = 0; iONode < 4; iONode++)
      macroAddRect(macGetNodePos(2, iONode), nodeSize, jet(ov[iONode]));

    void drwLine(float2 u, float2 v, float width, Color clr) {

      float2 dlt = v - u;
      float2 S = new float2(math.length(rectScale * dlt), math.cmin(rectScale) * width);
      dlt = rectScale * dlt;
      float2x2 R = float2x2.Rotate(math.atan2(dlt.y, dlt.x));
      float2x2 RS= math.mul(R,float2x2.Scale(S));
      Func<float2, float2> T2 = (vtx) => (math.mul(RS, vtx)  + T(u));
       
      int vtxc = vh.currentVertCount;
      for (int iVtx = 0; iVtx < 4; iVtx++)
      {
        vh.AddVert((Vector2)T2(rBase[iVtx]-new float2(0,0.5f)), clr, Vector2.zero);

      }
      vh.AddTriangle(vtxc + 0, vtxc + 1, vtxc + 2);
      vh.AddTriangle(vtxc + 3, vtxc + 0, vtxc + 2);

    }

    float2 xBuf = nodeSize/2;
    xBuf.y = 0;  
    for (int iINode = 0; iINode < 4; iINode++)
    {
      for (int iWNode = 0; iWNode < 4; iWNode++)
      {
        float2 posI = macGetNodePos(0, iINode) + nodeSize / 2;
        float2 posW = macGetNodePos(1, iWNode) + nodeSize / 2;

        float t = 0.5f + mlp.GetWeight(1,iINode,iWNode) / (2);

        drwLine(posI+xBuf, posW -xBuf, 0.025f, jet(t));
      }
    }
    for (int iWNode = 0; iWNode < 4; iWNode++)
    {
      for (int iONode = 0; iONode < 4; iONode++)
      {
        float2 posW = macGetNodePos(1, iWNode) + nodeSize / 2;
        float2 posO = macGetNodePos(2, iONode) + nodeSize / 2;

        float t = 0.5f + mlp.GetWeight(1, iWNode, iONode) ;

        drwLine(posW + xBuf, posO- xBuf, 0.025f, jet(t));
      }
    }
    /*
    var midPoint = testU / 2 + testV / 2;
    drwLine(testU, testV, 0.05f, Color.red);
    drwLine(midPoint, testV, 0.05f, Color.green);
    drwLine(testU,testU / 2 + testV / 2, 0.05f, Color.blue);
    float2 ps = 0.025f;
    ps.x /= aspect;
    macroAddRect(testU-ps/2, ps, Color.cyan);
    macroAddRect(testV - ps / 2, ps, Color.yellow);
    macroAddRect(midPoint - ps / 2, ps, Color.magenta);
    */
  }

  public float2 testU = new float2(0, 0.3f);
  public float2 testV = new float2(0.1f, 0.1f);
  public float2 _obsMin = new float2(-Mathf.PI, 0);
  public float2 _obsMax = new float2(Mathf.PI,5);
  public float4 obsMin => new float4(_obsMin, outBounds.c0.zw);
  public float4 obsMax => new float4(_obsMax, outBounds.c1.zw);
  public float4x2 hiddenBounds = 0;
  public float4x2 outBounds = 0;
  public float4 _observe = new float4(.5f, .33f, .5f, .33f);


  public string _AgentJson = "";
  [ContextMenu("LoadAgent")]
  public void LoadAgent() { _testMLP = JsonUtility.FromJson<MLP>(_AgentJson); }
  private MLP _testMLP;


  public MLP _TestMLP
  {
    get {
      if (_testMLP == null) {
        _testMLP = new MLP_Tensor();
        Random _rndu = new Random((uint)UnityEngine.Random.Range(0, int.MaxValue));
        GaussianGenerator _rndn = new GaussianGenerator(_rndu);
        _testMLP.Mutate(ref _rndn, 1);
      }
      return _testMLP;
    }
    set { _testMLP = value; }
  }

  [ContextMenu("Mark Dirty")]
  void MarkDirty() {
    UpdateGeometry();
  }


  [ContextMenu("Find Bounds")]
  public void FindBounds()
  {
    Random _rndu = new Random((uint)UnityEngine.Random.Range(0, int.MaxValue));
    var mlp = _TestMLP;
    var hbnds = new float4x2(float.PositiveInfinity, float.NegativeInfinity);
    var obnds = new float4x2(float.PositiveInfinity, float.NegativeInfinity);
    for (int iSample = 0; iSample < 20; iSample++) {
      float4 obs=0;
      obs.xy = _rndu.NextFloat2(_obsMin, _obsMax);
      obs.zw = _rndu.NextFloat2(-2, 2);

      // TODO: Get intermediate tensors;
      float4 hv = 1;// mlp.GetHiddenValues(obs, true)[0];
      hbnds.c0 = math.min(hbnds.c0, hv);
      hbnds.c1 = math.max(hbnds.c1, hv);

      float4 ov = mlp.Execute(obs);
      obnds.c0 = math.min(obnds.c0, ov);
      obnds.c1 = math.max(obnds.c1, ov);
    }

    hiddenBounds = hbnds + new float4x2(-.001f, .001f);
    outBounds = obnds + new float4x2(-.001f, .001f);
  }
}
