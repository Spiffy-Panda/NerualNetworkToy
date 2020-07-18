using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class TeleportingTarget : MonoBehaviour {
  public Rect _bounds;

  public float _period = 2;

  public float _nxtTime = -1;
    // Start is called before the first frame update
  void Start() {
  }

  void Update() {
    if (Time.timeSinceLevelLoad > _nxtTime) { 
      transform.position = Rect.NormalizedToPoint(_bounds,new Vector2(Random.value,Random.value));
      _nxtTime = Time.timeSinceLevelLoad + _period;
    }
  }
}
