using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class PeriodicUpdate : MonoBehaviour
{
  public float _period = 1;
  public float TimeIntoPeriod { get; private set; }
  public float TimeNormalized => TimeIntoPeriod / _period;
  public event Action Updated;
  // Update is called once per frame
  void Update()
  {
    TimeIntoPeriod += Time.deltaTime;
    if (TimeIntoPeriod > _period) {
      TimeIntoPeriod -= _period;
      Updated?.Invoke();
    }
      
  }
}
