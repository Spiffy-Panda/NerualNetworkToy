using System;
using System.Collections;
using System.Collections.Generic;
using SpiffyLibrary.MachineLearning;
using Unity.Mathematics;
using UnityEngine;
using Random = Unity.Mathematics.Random;


public class ClosestApproachMetric : MetricInfo
{
  public static string MetricName => typeof(ClosestApproachMetric).Name;
  public override string Name => MetricName;
  public ClosestApproachMetric() { TotalValue = float.PositiveInfinity;}

  public override void EvalIteractionTick(float3 state, float2 action)
  {
    float dst = math.length(state.xy);
    if (dst < TotalValue)
      TotalValue = dst;
  }
}
public class FinalDistanceMetric : MetricInfo
{
  public static string MetricName => typeof(FinalDistanceMetric).Name;
  public override string Name => MetricName;

  public override void EvalIteractionTick(float3 state, float2 action)
  {
    float dst = math.length(state.xy);
    TotalValue = dst;
  }
}
public class OverRotationMetric : MetricInfo
{
  public static string MetricName => typeof(OverRotationMetric).Name;
  public override string Name => MetricName;
  private float raw = 0;
  private int count = 0;
  public override void EvalIteractionTick(float3 state, float2 action) {
    raw = (count * raw + action.y) / (count + 1);
    count += 1;
    TotalValue = Math.Abs(raw);
  }
}
