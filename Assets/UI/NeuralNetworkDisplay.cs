using SpiffyLibrary;
using SpiffyLibrary.MachineLearning;
using System;
using System.Linq;
using SpiffyLibrary.UIElements;
using Unity.Barracuda;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.UIElements;
using Random = Unity.Mathematics.Random;

namespace ProjectUI
{
  public class NeuralNetworkDisplay : VisualElement{
    public new class UxmlFactory : UxmlFactory<NeuralNetworkDisplay, UxmlTraits>{}

    private readonly MoveToPointPreview _preview;
    private readonly AcademyMove _academy;

    // we are using relu6
    public float2 _actNodeMinMax = new float2(0, 6);

    // just enough to show overflow
    public float2 _rawNodeMinMax = new float2(-1, 7);
    public float[] _observe = new float[] {.5f, .33f, .5f, .33f, 0.1f};

    private static readonly float2x4 RectCourners = new float2x4(0, 1, 1, 0, 0, 0, 1, 1);

    private readonly string[] _extraLayers = new string[] {
      MultiLayerPerception.LayerNames.Hidden,
      MultiLayerPerception.LayerNames.HiddenActive,
      MultiLayerPerception.LayerNames.Output
    };

    float2 NodeSize => new float2(0.2f / (RectScale.x / RectScale.y), 0.2f);
    private float2 RectScale => layout.size;

    Color RawNodeColor(float r) => TurboColorMap.Map(math.unlerp(_rawNodeMinMax.x, _rawNodeMinMax.y, r));
    Color ActNodeColor(float a) => TurboColorMap.Map(math.unlerp(_actNodeMinMax.x, _actNodeMinMax.y, a));

    public string _MLPJson = "";
    public int3 _testNNShape = new int3(3, 3, 3);

    [ContextMenu("LoadAgent")]
    public void LoadAgent() { throw new NotImplementedException(); }

    private MultiLayerPerception _testMLP;


    public MultiLayerPerception _TestMLP {
      get {
        if (_testMLP == null) {
          Debug.Log("No model yet.");
        }
        return _testMLP;
      }
      set {
        _testMLP = value; 
        MarkDirtyRepaint();
      }
    }


    public NeuralNetworkDisplay() {

      generateVisualContent += OnGenerateVisualContent;
      RegisterCallback<MouseDownEvent>(OnClick);
    }

    private void OnClick(MouseDownEvent evt) {

    }


    private void AddRect(MeshParts mp, float2 pos, float2 size, Color clr) {
      Func<float2, float2> T = (vtx) => (RectScale * vtx);
      int vtxOffset = mp.vertices.Count;
      for (int iVtx = 0; iVtx < 4; iVtx++) {
        mp.vertices.Add(new Vertex {
          position = (Vector2) T(size * RectCourners[iVtx] + pos),
          tint = clr,
          uv = Vector2.zero
        });

      }

      mp.triangles.Add(vtxOffset + new int3(2, 1, 0));
      mp.triangles.Add(vtxOffset + new int3(0, 1, 2));
      mp.triangles.Add(vtxOffset + new int3(3, 0, 2));
    }

    private void DrawLine(MeshParts mp, float2 u, float2 v, float width, Color clr) {

      float2 dlt = v - u;
      float2 S = new float2(math.length(RectScale * dlt), math.cmin(RectScale) * width);
      dlt = RectScale * dlt;
      float2x2 R = float2x2.Rotate(math.atan2(dlt.y, dlt.x));
      float2x2 RS = math.mul(R, float2x2.Scale(S));
      Func<float2, float2> T = (vtx) => (RectScale * vtx);
      Func<float2, float2> T2 = (vtx) => (math.mul(RS, vtx) + T(u));

      int vtxOffset = mp.vertices.Count;
      for (int iVtx = 0; iVtx < 4; iVtx++)
      {
        mp.vertices.Add(new Vertex
        {
          position = (Vector2)T2(RectCourners[iVtx] - new float2(0, 0.5f)),
          tint = clr,
          uv = Vector2.zero
        });

      }
      mp.triangles.Add(vtxOffset + new int3(0, 1, 2));
      mp.triangles.Add(vtxOffset + new int3(3, 0, 2));
      

    }

    private float2 GetNodePos(int layer, int idx) {
      int[] layerSizes = new int[] {_TestMLP._shape.inputSize, _TestMLP._shape.hiddenSize, _TestMLP._shape.outputSize};
      float2 usable = 1 - NodeSize;
      float2 nPos;
      nPos.x = layer / (2f);
      nPos.y = 1 - idx / (layerSizes[layer] - 1f);
      return usable * nPos;
    }
    private void OnGenerateVisualContent(MeshGenerationContext cxt) {
      MeshParts mp = new MeshParts();
      MultiLayerPerception mlp = _TestMLP;
      AddRect(mp, 0, 1, Color.gray);
      if (mlp != null) {

        using (IWorker oneshotSyncWorker =
          WorkerFactory.CreateWorker(_testMLP.model, _extraLayers, WorkerFactory.Device.GPU)) {

          using (Tensor obsTensor = new Tensor(new TensorShape(1, mlp._shape.inputSize))) {
            if (_observe.Length < mlp._shape.inputSize)
              _observe = new float[mlp._shape.inputSize];
            for (int iINode = 0; iINode < mlp._shape.inputSize; iINode++) {
              obsTensor[iINode] = _observe[iINode];
            }

            oneshotSyncWorker.Execute(obsTensor).FlushSchedule();
          }

          for (int iINode = 0; iINode < mlp._shape.inputSize; iINode++) {
            AddRect(mp, GetNodePos(0, iINode), NodeSize, ActNodeColor(_observe[iINode]));
          }


          using (Tensor hvr = oneshotSyncWorker.PeekOutput(MultiLayerPerception.LayerNames.Hidden)) {
            using (Tensor hva = oneshotSyncWorker.PeekOutput(MultiLayerPerception.LayerNames.HiddenActive)) {
              for (int iHNode = 0; iHNode < mlp._shape.hiddenSize; iHNode++) {
                AddRect(mp, GetNodePos(1, iHNode), NodeSize, RawNodeColor(hvr[iHNode]));
                AddRect(mp, GetNodePos(1, iHNode) + new float2(0.5f, 0) * NodeSize, new float2(0.5f, 1) * NodeSize,
                  ActNodeColor(hva[iHNode]));
              }
            }
          }

          using (Tensor ovr = oneshotSyncWorker.PeekOutput(MultiLayerPerception.LayerNames.Output)) {
            using (Tensor ova = oneshotSyncWorker.PeekOutput()) {
              for (int iONode = 0; iONode < mlp._shape.outputSize; iONode++) {
                AddRect(mp, GetNodePos(2, iONode), NodeSize, RawNodeColor(ovr[iONode]));
                AddRect(mp, GetNodePos(2, iONode) + new float2(0.5f, 0) * NodeSize, new float2(0.5f, 1) * NodeSize,
                  ActNodeColor(ova[iONode]));
              }
            }
          }
        }

      string[] layerNames = new string[]
        {MultiLayerPerception.LayerNames.Hidden, MultiLayerPerception.LayerNames.Output};
      float2 xBuf = NodeSize / 2;
      xBuf.y = 0;
      int prvLayer = 0;
      int curLayer = 1;
      foreach (string layerName in layerNames) {
        TensorShape tShape = _testMLP.GetLayerShape(layerName);
        for (int iPNode = 0; iPNode < tShape.flatHeight; iPNode++) {
          for (int iCNode = 0; iCNode < tShape.flatWidth; iCNode++) {
            float2 posI = GetNodePos(prvLayer, iPNode) + NodeSize / 2;
            float2 posW = GetNodePos(curLayer, iCNode) + NodeSize / 2;

            float t = 0.5f + mlp.GetWeight(layerName, iPNode, iCNode);
            DrawLine(mp, posI + xBuf, posW - xBuf, 0.025f, TurboColorMap.Map(t));
          }
        }

        prvLayer = curLayer;
        curLayer++;
        }
      }
      MeshWriteData meshData = cxt.Allocate(mp.vertices.Count, mp.IndicesCount);
      if (meshData.vertexCount > 0) {
        meshData.SetAllVertices(mp.vertices.ToArray());
        meshData.SetAllIndices(mp.GetIndices());
      }
    }

    [ContextMenu("Mark Dirty")]
    private void MarkDirtyContextMenu() { MarkDirtyRepaint();}

  }
}