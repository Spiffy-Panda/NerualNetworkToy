using System;
using UnityEngine;

namespace SpiffyLibrary
{
  public class PeriodicUpdate : MonoBehaviour
  {
    public float _period = 1;
    public float TimeIntoPeriod { get; private set; }
    public float TimeNormalized => TimeIntoPeriod / _period;

    public event Action Updated;

    // Update is called once per frame
    private void Update()
    {
      TimeIntoPeriod += Time.deltaTime;
      if (TimeIntoPeriod >= _period)
      {
        TimeIntoPeriod -= _period;
        Updated?.Invoke();
      }

    }

    public void MarkToTrigger() {
      if (TimeIntoPeriod < _period)
        TimeIntoPeriod = _period;
    }
  }
}