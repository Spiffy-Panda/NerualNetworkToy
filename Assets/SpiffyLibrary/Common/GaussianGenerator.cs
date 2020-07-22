using System;
using System.Collections.Generic;
using System.Linq;
using Unity.Mathematics;
using UnityEngine;
using static Unity.Mathematics.math;
using Random = Unity.Mathematics.Random;

namespace SpiffyLibrary
{

  public class GaussianGenerator
  {
    public static GaussianGenerator Inst { get; } = new GaussianGenerator(new Random((uint)UnityEngine.Random.Range(0,int.MaxValue)));
    private Random _uniform;
    private const float epsilon = Single.Epsilon;
    private const float tau = math.PI * 2;
    private float _spare;

    public GaussianGenerator(Random uniform)
    {
      _uniform = uniform;
    }

    public float NextFloat1()
    {
      if (float.IsNaN(_spare))
      {
        float2 tmp = NextFloat2();
        _spare = tmp.x;
        return tmp.y;
      }
      else
      {
        float result = _spare;
        _spare = float.NaN;
        return result;
      }
    }
    public float2 NextFloat2()
    {
      float2 s;
      do
      {
        s = _uniform.NextFloat2();
      }
      while (s.x <= epsilon);//happens rarely but 0 breaks log

      return sqrt(-2.0f * log(s.x)) * new float2(cos(tau * s.y), sin(tau * s.y));
    }

    public float3 NextFloat3()
    {
      return new float3(NextFloat2(), NextFloat1());
    }

    public float4 NextFloat4()
    {
      return new float4(NextFloat2(), NextFloat2());
    }

    public float4x4 NextFloat4x4()
    {
      return new float4x4(NextFloat4(), NextFloat4(), NextFloat4(), NextFloat4());
    }

    public float3x4 NextFloat3x4()
    {
      return new float3x4(NextFloat3(), NextFloat3(), NextFloat3(), NextFloat3());
    }

    public float2x4 NextFloat2x4()
    {
      return new float2x4(NextFloat2(), NextFloat2(), NextFloat2(), NextFloat2());
    }

    public float4x3 NextFloat4x3()
    {
      return new float4x3(NextFloat4(), NextFloat4(), NextFloat4());
    }

    public float3x3 NextFloat3x3()
    {
      return new float3x3(NextFloat3(), NextFloat3(), NextFloat3());
    }

    public float2x3 NextFloat2x3()
    {
      return new float2x3(NextFloat2(), NextFloat2(), NextFloat2());
    }

    public float4x2 NextFloat4x2()
    {
      return new float4x2(NextFloat4(), NextFloat4());
    }

    public float3x2 NextFloat3x2()
    {
      return new float3x2(NextFloat3(), NextFloat3());
    }

    public float2x2 NextFloat2x2()
    {
      return new float2x2(NextFloat2(), NextFloat2());
    }


  }


}