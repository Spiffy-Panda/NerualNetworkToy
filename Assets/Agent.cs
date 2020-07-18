using System.Collections;
using System.Collections.Generic;
using Unity.Mathematics;
using UnityEngine;

public class Agent : MonoBehaviour {
  public Vector2 _target;
  public MLP_2I_4H_2O _brain;

  public float4 _state;

  public float _thougth;


  // Update is called once per frame
  void Update() {
    Vector2 delta = _target - (Vector2)transform.position;
    float bearing = Mathf.Atan2(delta.y, delta.x) ;
    _state.x = bearing / Mathf.PI;
    _state.y = delta.magnitude / 10;
    Debug.DrawLine(transform.position, transform.right * delta.magnitude);
    Debug.DrawRay( transform.right, transform.up * bearing);
  }
}
