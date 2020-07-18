using System;
using System.Collections;
using System.Collections.Generic;
using Unity.Mathematics;
using UnityEngine;

public class EvaluationWeight : MonoBehaviour
{
  [SerializeField]
  private BarycentricSlider _ratingWeightUi;
  [SerializeField]
  private  float3 _inspectorWeight = 1;

  public float3 Weights { get; private set; } = float3.zero;

  public event Action ValueChanged;
  public void Start() {
    if(_ratingWeightUi)
      _ratingWeightUi.Clicked += UIClicked;
    _ratingWeightUi._value = _inspectorWeight;
  }

  public void Update() {
    if (math.all(_inspectorWeight != Weights)) {
      Weights = _inspectorWeight;
      ValueChanged?.Invoke();
    }
  }
  

  private void UIClicked()
  {
    Weights = _ratingWeightUi._value;
    _inspectorWeight = Weights;
    ValueChanged?.Invoke();
  }

  public float Apply(float3 raw) {
    if (math.all(Weights == float3.zero))
      Weights = _inspectorWeight;
    return math.dot(Weights, raw);
  }
}
