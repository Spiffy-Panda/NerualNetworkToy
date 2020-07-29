using System.Collections;
using System.Collections.Generic;
using ProjectUI;
using SpiffyLibrary;
using UnityEngine;
using UnityEngine.UIElements;

public static class UIStrings {
  public static class Names {
    public const string PreviewSelection = "PreviewSelection";
    public const string SelectedParent = "SelectedParent";
  }
}

public class ProjectUIInterface : MonoBehaviour {
  public VisualElement _root;

  public RenderTexture _texture;

  private ParetoBarycentricMap _previewSelectionMap;
  private ParetoBarycentricMap _parentSelectionMap;
  // Start is called before the first frame update
  void Start() {
    _root = GetComponent<UIDocument>().rootVisualElement;
    _previewSelectionMap = _root.Q(name: UIStrings.Names.PreviewSelection) as ParetoBarycentricMap;
    _parentSelectionMap  = _root.Q(name: UIStrings.Names.SelectedParent) as ParetoBarycentricMap;

    if(_previewSelectionMap!=null)
      _previewSelectionMap.Clicked +=PreviewMapClicked;
    if (_parentSelectionMap != null)
      _parentSelectionMap.Clicked += ParentMapClicked;
  }

  private void PreviewMapClicked()
  {
    var preview = FindObjectOfType<MoveToPointPreview>();
    if (_previewSelectionMap.Value_ID >= 0)
      preview.SelectedGeneId = _previewSelectionMap.Value_ID;
    preview.GetComponent<PeriodicUpdate>()?.MarkTrigger();
  }
  private void ParentMapClicked()
  {
    var preview = FindObjectOfType<MoveToPointPreview>();
    if (_previewSelectionMap.Value_ID >= 0)
      preview.SelectedGeneId = _previewSelectionMap.Value_ID;
    preview.GetComponent<PeriodicUpdate>()?.MarkTrigger();
  }

  // Update is called once per frame
  void Update()
    {
        
    }
}
