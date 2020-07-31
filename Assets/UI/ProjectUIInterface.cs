using System.Collections.ObjectModel;
using SpiffyLibrary;
using SpiffyLibrary.MachineLearning;
using System.Linq;
using Unity.Barracuda;
using UnityEngine;
using UnityEngine.UIElements;

namespace ProjectUI
{

  using static ProjectUI.Generated.MainUI2Names;

  public class ProjectUIInterface : MonoBehaviour {
    public AcademyMove _academy;
    public GeneBankManager _genebank => GeneBankManager.Inst;
    public VisualElement _root;

    public RenderTexture _texture;

    private ParetoBarycentricMap _previewSelectionMap;
    private ParetoBarycentricMap _parentSelectionMap;
    private NeuralNetworkDisplay _nnDisplay;
    private Button _btnApply;
    private Button _btnStop;
    private Button _btnTabRandomSearch;
    private Button _btnTabRandomParent;
    private Button _btnTabNoEvolution;
    private Button _btnTabSelectedParent;
    private TextElement _txtTabContent;

    // In case something dominates selected parent
    private ReadOnlyCollection<float> _weightsOfSelected = null;
    public enum EvolutionTab {
      None,
      RandomSearch,
      SelectedParent,
      RandomParent
    };
    
    public EvolutionTab _currentTab { get; private set; } = EvolutionTab.None;
    // Start is called before the first frame update
    private void Start()
    {
      _root = GetComponent<UIDocument>().rootVisualElement;
      _previewSelectionMap = _root.Q<ParetoBarycentricMap>(name: Names.PreviewSelection);
      _parentSelectionMap = _root.Q<ParetoBarycentricMap>(name: Names.SelectedParent);
      _nnDisplay = _root.Q<NeuralNetworkDisplay>(name: Names.NeuralNetworkDisplay);
      _btnApply = _root.Q<Button>(name: Names.btnApply);
      _btnStop = _root.Q<Button>(name: Names.btnStop);
      _btnTabRandomSearch = _root.Q<Button>(name: Names.btnTabRandomSearch);
      _btnTabRandomParent = _root.Q<Button>(name: Names.btnTabRandomParent);
      _btnTabNoEvolution = _root.Q<Button>(name: Names.btnTabNoEvolution);
      _btnTabSelectedParent = _root.Q<Button>(name: Names.btnTabSelectedParent);
      _txtTabContent = _root.Q<TextElement>(name: Names.txtTabContent);

      _btnApply.clicked += ApplyClicked;
      _btnStop.clicked += () => _academy._isGenerating = false;
      _btnTabNoEvolution   .clicked += () => SelectTab(EvolutionTab.None);
      _btnTabRandomSearch  .clicked += () => SelectTab(EvolutionTab.RandomSearch);
      _btnTabSelectedParent.clicked += () => SelectTab(EvolutionTab.SelectedParent);
      _btnTabRandomParent  .clicked += () => SelectTab(EvolutionTab.RandomParent);
      if (_previewSelectionMap != null)
      {
        _previewSelectionMap.Clicked += PreviewMapClicked;
      }

      if (_parentSelectionMap != null)
      {
        _parentSelectionMap.Clicked += ParentMapClicked;
      }
    }

    private void SelectTab(EvolutionTab newTab) {
      _currentTab = newTab;
      _txtTabContent.text = newTab.ToString();
    }
    private void ApplyClicked()
    {
      switch (_currentTab)
      {
        case EvolutionTab.None:
          _academy._isGenerating = false;
          break;
        case EvolutionTab.RandomSearch:
          _academy._pickParent = MutationManager.Inst.DefaultGenerator;
          _academy._isGenerating = true;
          break;
        case EvolutionTab.RandomParent:
          _academy._pickParent = ()=> MutationManager.Inst.MutateExisting(_genebank.GetRandomGenome()._weights);
          _academy._isGenerating = true;
          break;
        case EvolutionTab.SelectedParent:
          if (_weightsOfSelected != null) {
            _academy._pickParent = () => MutationManager.Inst.MutateExisting(_weightsOfSelected);
            _academy._isGenerating = true;
          } else {
            _txtTabContent.text += "\nFAIL: Could not create generator.";
            _academy._isGenerating = false;
          }
          break;
        default:
          break;
      }
    }

    private void PreviewMapClicked()
    {
      MoveToPointPreview preview = FindObjectOfType<MoveToPointPreview>();
      if (_previewSelectionMap.Value_ID >= 0)
      {
        preview.SelectedGeneId = _previewSelectionMap.Value_ID;
      }

      preview.GetComponent<PeriodicUpdate>()?.MarkToTrigger();

      Debug.Log("PreviewClicked");
      ParetoGeneBank.Genome gi = GeneBankManager.Inst.GetGenomeByID(_previewSelectionMap.Value_ID);
      if (gi == null)
      {
        return;
      }

      MultiLayerPerception mlp = new MultiLayerPerception(MoveSimParams.GetDefault().mlpShape, Layer.FusedActivation.Relu6);
      mlp.LoadWeights(gi._weights.ToArray());
      _nnDisplay._TestMLP = mlp;
      _nnDisplay.MarkDirtyRepaint();
      Debug.Log("SetNNDisp");
    }
    private void ParentMapClicked()
    {

      MoveToPointPreview preview = FindObjectOfType<MoveToPointPreview>();
      if (_parentSelectionMap.Value_ID >= 0)
      {
        preview.SelectedGeneId = _parentSelectionMap.Value_ID;
      }

      _weightsOfSelected = _genebank.GetGenomeByID(_parentSelectionMap.Value_ID)._weights;
      preview.GetComponent<PeriodicUpdate>()?.MarkToTrigger();
      Debug.Log("ParentClicked");
    }

    // Update is called once per frame
    private void Update()
    {

    }
  }

}