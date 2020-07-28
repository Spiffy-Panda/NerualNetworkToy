using System.Collections;
using System.Collections.Generic;
using ProjectUI;
using SpiffyLibrary;
using UnityEngine;
using UnityEngine.UIElements;

public class ProjectUIInterface : MonoBehaviour {
  public VisualElement _root;

  public RenderTexture _texture;

  private ParetoBarycentricMap baryMap;
  // Start is called before the first frame update
  void Start() {
    _root = GetComponent<UIDocument>().rootVisualElement;
    baryMap = _root.Q(name: "ParetoBarycentricMap") as ParetoBarycentricMap;
    if(baryMap!=null)
      baryMap.Clicked +=Clicked;
  }

  private void Clicked() {
    var preview = FindObjectOfType<MoveToPointPreview>();
    if (baryMap.Value_ID >= 0)
      preview.SelectedGeneId = baryMap.Value_ID;
    preview.GetComponent<PeriodicUpdate>()?.MarkTrigger();
  }

  // Update is called once per frame
    void Update()
    {
        
    }
}
