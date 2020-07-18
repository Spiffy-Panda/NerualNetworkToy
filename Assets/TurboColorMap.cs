﻿using System;
using System.Collections;
using System.Collections.Generic;
using Unity.Mathematics;
using UnityEngine;

// Based on Google's Turbo Color Map
public class TurboColorMap {
  private static float3[] lookupTable = new float3[] {
    new float3(0.18995f, 0.07176f, 0.23217f),
    new float3(0.19483f, 0.08339f, 0.26149f),
    new float3(0.19956f, 0.09498f, 0.29024f),
    new float3(0.20415f, 0.10652f, 0.31844f),
    new float3(0.2086f, 0.11802f, 0.34607f),
    new float3(0.21291f, 0.12947f, 0.37314f),
    new float3(0.21708f, 0.14087f, 0.39964f),
    new float3(0.22111f, 0.15223f, 0.42558f),
    new float3(0.225f, 0.16354f, 0.45096f),
    new float3(0.22875f, 0.17481f, 0.47578f),
    new float3(0.23236f, 0.18603f, 0.50004f),
    new float3(0.23582f, 0.1972f, 0.52373f),
    new float3(0.23915f, 0.20833f, 0.54686f),
    new float3(0.24234f, 0.21941f, 0.56942f),
    new float3(0.24539f, 0.23044f, 0.59142f),
    new float3(0.2483f, 0.24143f, 0.61286f),
    new float3(0.25107f, 0.25237f, 0.63374f),
    new float3(0.25369f, 0.26327f, 0.65406f),
    new float3(0.25618f, 0.27412f, 0.67381f),
    new float3(0.25853f, 0.28492f, 0.693f),
    new float3(0.26074f, 0.29568f, 0.71162f),
    new float3(0.2628f, 0.30639f, 0.72968f),
    new float3(0.26473f, 0.31706f, 0.74718f),
    new float3(0.26652f, 0.32768f, 0.76412f),
    new float3(0.26816f, 0.33825f, 0.7805f),
    new float3(0.26967f, 0.34878f, 0.79631f),
    new float3(0.27103f, 0.35926f, 0.81156f),
    new float3(0.27226f, 0.3697f, 0.82624f),
    new float3(0.27334f, 0.38008f, 0.84037f),
    new float3(0.27429f, 0.39043f, 0.85393f),
    new float3(0.27509f, 0.40072f, 0.86692f),
    new float3(0.27576f, 0.41097f, 0.87936f),
    new float3(0.27628f, 0.42118f, 0.89123f),
    new float3(0.27667f, 0.43134f, 0.90254f),
    new float3(0.27691f, 0.44145f, 0.91328f),
    new float3(0.27701f, 0.45152f, 0.92347f),
    new float3(0.27698f, 0.46153f, 0.93309f),
    new float3(0.2768f, 0.47151f, 0.94214f),
    new float3(0.27648f, 0.48144f, 0.95064f),
    new float3(0.27603f, 0.49132f, 0.95857f),
    new float3(0.27543f, 0.50115f, 0.96594f),
    new float3(0.27469f, 0.51094f, 0.97275f),
    new float3(0.27381f, 0.52069f, 0.97899f),
    new float3(0.27273f, 0.5304f, 0.98461f),
    new float3(0.27106f, 0.54015f, 0.9893f),
    new float3(0.26878f, 0.54995f, 0.99303f),
    new float3(0.26592f, 0.55979f, 0.99583f),
    new float3(0.26252f, 0.56967f, 0.99773f),
    new float3(0.25862f, 0.57958f, 0.99876f),
    new float3(0.25425f, 0.5895f, 0.99896f),
    new float3(0.24946f, 0.59943f, 0.99835f),
    new float3(0.24427f, 0.60937f, 0.99697f),
    new float3(0.23874f, 0.61931f, 0.99485f),
    new float3(0.23288f, 0.62923f, 0.99202f),
    new float3(0.22676f, 0.63913f, 0.98851f),
    new float3(0.22039f, 0.64901f, 0.98436f),
    new float3(0.21382f, 0.65886f, 0.97959f),
    new float3(0.20708f, 0.66866f, 0.97423f),
    new float3(0.20021f, 0.67842f, 0.96833f),
    new float3(0.19326f, 0.68812f, 0.9619f),
    new float3(0.18625f, 0.69775f, 0.95498f),
    new float3(0.17923f, 0.70732f, 0.94761f),
    new float3(0.17223f, 0.7168f, 0.93981f),
    new float3(0.16529f, 0.7262f, 0.93161f),
    new float3(0.15844f, 0.73551f, 0.92305f),
    new float3(0.15173f, 0.74472f, 0.91416f),
    new float3(0.14519f, 0.75381f, 0.90496f),
    new float3(0.13886f, 0.76279f, 0.8955f),
    new float3(0.13278f, 0.77165f, 0.8858f),
    new float3(0.12698f, 0.78037f, 0.8759f),
    new float3(0.12151f, 0.78896f, 0.86581f),
    new float3(0.11639f, 0.7974f, 0.85559f),
    new float3(0.11167f, 0.80569f, 0.84525f),
    new float3(0.10738f, 0.81381f, 0.83484f),
    new float3(0.10357f, 0.82177f, 0.82437f),
    new float3(0.10026f, 0.82955f, 0.81389f),
    new float3(0.0975f, 0.83714f, 0.80342f),
    new float3(0.09532f, 0.84455f, 0.79299f),
    new float3(0.09377f, 0.85175f, 0.78264f),
    new float3(0.09287f, 0.85875f, 0.7724f),
    new float3(0.09267f, 0.86554f, 0.7623f),
    new float3(0.0932f, 0.87211f, 0.75237f),
    new float3(0.09451f, 0.87844f, 0.74265f),
    new float3(0.09662f, 0.88454f, 0.73316f),
    new float3(0.09958f, 0.8904f, 0.72393f),
    new float3(0.10342f, 0.896f, 0.715f),
    new float3(0.10815f, 0.90142f, 0.70599f),
    new float3(0.11374f, 0.90673f, 0.69651f),
    new float3(0.12014f, 0.91193f, 0.6866f),
    new float3(0.12733f, 0.91701f, 0.67627f),
    new float3(0.13526f, 0.92197f, 0.66556f),
    new float3(0.14391f, 0.9268f, 0.65448f),
    new float3(0.15323f, 0.93151f, 0.64308f),
    new float3(0.16319f, 0.93609f, 0.63137f),
    new float3(0.17377f, 0.94053f, 0.61938f),
    new float3(0.18491f, 0.94484f, 0.60713f),
    new float3(0.19659f, 0.94901f, 0.59466f),
    new float3(0.20877f, 0.95304f, 0.58199f),
    new float3(0.22142f, 0.95692f, 0.56914f),
    new float3(0.23449f, 0.96065f, 0.55614f),
    new float3(0.24797f, 0.96423f, 0.54303f),
    new float3(0.2618f, 0.96765f, 0.52981f),
    new float3(0.27597f, 0.97092f, 0.51653f),
    new float3(0.29042f, 0.97403f, 0.50321f),
    new float3(0.30513f, 0.97697f, 0.48987f),
    new float3(0.32006f, 0.97974f, 0.47654f),
    new float3(0.33517f, 0.98234f, 0.46325f),
    new float3(0.35043f, 0.98477f, 0.45002f),
    new float3(0.36581f, 0.98702f, 0.43688f),
    new float3(0.38127f, 0.98909f, 0.42386f),
    new float3(0.39678f, 0.99098f, 0.41098f),
    new float3(0.41229f, 0.99268f, 0.39826f),
    new float3(0.42778f, 0.99419f, 0.38575f),
    new float3(0.44321f, 0.99551f, 0.37345f),
    new float3(0.45854f, 0.99663f, 0.3614f),
    new float3(0.47375f, 0.99755f, 0.34963f),
    new float3(0.48879f, 0.99828f, 0.33816f),
    new float3(0.50362f, 0.99879f, 0.32701f),
    new float3(0.51822f, 0.9991f, 0.31622f),
    new float3(0.53255f, 0.99919f, 0.30581f),
    new float3(0.54658f, 0.99907f, 0.29581f),
    new float3(0.56026f, 0.99873f, 0.28623f),
    new float3(0.57357f, 0.99817f, 0.27712f),
    new float3(0.58646f, 0.99739f, 0.26849f),
    new float3(0.59891f, 0.99638f, 0.26038f),
    new float3(0.61088f, 0.99514f, 0.2528f),
    new float3(0.62233f, 0.99366f, 0.24579f),
    new float3(0.63323f, 0.99195f, 0.23937f),
    new float3(0.64362f, 0.98999f, 0.23356f),
    new float3(0.65394f, 0.98775f, 0.22835f),
    new float3(0.66428f, 0.98524f, 0.2237f),
    new float3(0.67462f, 0.98246f, 0.2196f),
    new float3(0.68494f, 0.97941f, 0.21602f),
    new float3(0.69525f, 0.9761f, 0.21294f),
    new float3(0.70553f, 0.97255f, 0.21032f),
    new float3(0.71577f, 0.96875f, 0.20815f),
    new float3(0.72596f, 0.9647f, 0.2064f),
    new float3(0.7361f, 0.96043f, 0.20504f),
    new float3(0.74617f, 0.95593f, 0.20406f),
    new float3(0.75617f, 0.95121f, 0.20343f),
    new float3(0.76608f, 0.94627f, 0.20311f),
    new float3(0.77591f, 0.94113f, 0.2031f),
    new float3(0.78563f, 0.93579f, 0.20336f),
    new float3(0.79524f, 0.93025f, 0.20386f),
    new float3(0.80473f, 0.92452f, 0.20459f),
    new float3(0.8141f, 0.91861f, 0.20552f),
    new float3(0.82333f, 0.91253f, 0.20663f),
    new float3(0.83241f, 0.90627f, 0.20788f),
    new float3(0.84133f, 0.89986f, 0.20926f),
    new float3(0.8501f, 0.89328f, 0.21074f),
    new float3(0.85868f, 0.88655f, 0.2123f),
    new float3(0.86709f, 0.87968f, 0.21391f),
    new float3(0.8753f, 0.87267f, 0.21555f),
    new float3(0.88331f, 0.86553f, 0.21719f),
    new float3(0.89112f, 0.85826f, 0.2188f),
    new float3(0.8987f, 0.85087f, 0.22038f),
    new float3(0.90605f, 0.84337f, 0.22188f),
    new float3(0.91317f, 0.83576f, 0.22328f),
    new float3(0.92004f, 0.82806f, 0.22456f),
    new float3(0.92666f, 0.82025f, 0.2257f),
    new float3(0.93301f, 0.81236f, 0.22667f),
    new float3(0.93909f, 0.80439f, 0.22744f),
    new float3(0.94489f, 0.79634f, 0.228f),
    new float3(0.95039f, 0.78823f, 0.22831f),
    new float3(0.9556f, 0.78005f, 0.22836f),
    new float3(0.96049f, 0.77181f, 0.22811f),
    new float3(0.96507f, 0.76352f, 0.22754f),
    new float3(0.96931f, 0.75519f, 0.22663f),
    new float3(0.97323f, 0.74682f, 0.22536f),
    new float3(0.97679f, 0.73842f, 0.22369f),
    new float3(0.98f, 0.73f, 0.22161f),
    new float3(0.98289f, 0.7214f, 0.21918f),
    new float3(0.98549f, 0.7125f, 0.2165f),
    new float3(0.98781f, 0.7033f, 0.21358f),
    new float3(0.98986f, 0.69382f, 0.21043f),
    new float3(0.99163f, 0.68408f, 0.20706f),
    new float3(0.99314f, 0.67408f, 0.20348f),
    new float3(0.99438f, 0.66386f, 0.19971f),
    new float3(0.99535f, 0.65341f, 0.19577f),
    new float3(0.99607f, 0.64277f, 0.19165f),
    new float3(0.99654f, 0.63193f, 0.18738f),
    new float3(0.99675f, 0.62093f, 0.18297f),
    new float3(0.99672f, 0.60977f, 0.17842f),
    new float3(0.99644f, 0.59846f, 0.17376f),
    new float3(0.99593f, 0.58703f, 0.16899f),
    new float3(0.99517f, 0.57549f, 0.16412f),
    new float3(0.99419f, 0.56386f, 0.15918f),
    new float3(0.99297f, 0.55214f, 0.15417f),
    new float3(0.99153f, 0.54036f, 0.1491f),
    new float3(0.98987f, 0.52854f, 0.14398f),
    new float3(0.98799f, 0.51667f, 0.13883f),
    new float3(0.9859f, 0.50479f, 0.13367f),
    new float3(0.9836f, 0.49291f, 0.12849f),
    new float3(0.98108f, 0.48104f, 0.12332f),
    new float3(0.97837f, 0.4692f, 0.11817f),
    new float3(0.97545f, 0.4574f, 0.11305f),
    new float3(0.97234f, 0.44565f, 0.10797f),
    new float3(0.96904f, 0.43399f, 0.10294f),
    new float3(0.96555f, 0.42241f, 0.09798f),
    new float3(0.96187f, 0.41093f, 0.0931f),
    new float3(0.95801f, 0.39958f, 0.08831f),
    new float3(0.95398f, 0.38836f, 0.08362f),
    new float3(0.94977f, 0.37729f, 0.07905f),
    new float3(0.94538f, 0.36638f, 0.07461f),
    new float3(0.94084f, 0.35566f, 0.07031f),
    new float3(0.93612f, 0.34513f, 0.06616f),
    new float3(0.93125f, 0.33482f, 0.06218f),
    new float3(0.92623f, 0.32473f, 0.05837f),
    new float3(0.92105f, 0.31489f, 0.05475f),
    new float3(0.91572f, 0.3053f, 0.05134f),
    new float3(0.91024f, 0.29599f, 0.04814f),
    new float3(0.90463f, 0.28696f, 0.04516f),
    new float3(0.89888f, 0.27824f, 0.04243f),
    new float3(0.89298f, 0.26981f, 0.03993f),
    new float3(0.88691f, 0.26152f, 0.03753f),
    new float3(0.88066f, 0.25334f, 0.03521f),
    new float3(0.87422f, 0.24526f, 0.03297f),
    new float3(0.8676f, 0.2373f, 0.03082f),
    new float3(0.86079f, 0.22945f, 0.02875f),
    new float3(0.8538f, 0.2217f, 0.02677f),
    new float3(0.84662f, 0.21407f, 0.02487f),
    new float3(0.83926f, 0.20654f, 0.02305f),
    new float3(0.83172f, 0.19912f, 0.02131f),
    new float3(0.82399f, 0.19182f, 0.01966f),
    new float3(0.81608f, 0.18462f, 0.01809f),
    new float3(0.80799f, 0.17753f, 0.0166f),
    new float3(0.79971f, 0.17055f, 0.0152f),
    new float3(0.79125f, 0.16368f, 0.01387f),
    new float3(0.7826f, 0.15693f, 0.01264f),
    new float3(0.77377f, 0.15028f, 0.01148f),
    new float3(0.76476f, 0.14374f, 0.01041f),
    new float3(0.75556f, 0.13731f, 0.00942f),
    new float3(0.74617f, 0.13098f, 0.00851f),
    new float3(0.73661f, 0.12477f, 0.00769f),
    new float3(0.72686f, 0.11867f, 0.00695f),
    new float3(0.71692f, 0.11268f, 0.00629f),
    new float3(0.7068f, 0.1068f, 0.00571f),
    new float3(0.6965f, 0.10102f, 0.00522f),
    new float3(0.68602f, 0.09536f, 0.00481f),
    new float3(0.67535f, 0.0898f, 0.00449f),
    new float3(0.66449f, 0.08436f, 0.00424f),
    new float3(0.65345f, 0.07902f, 0.00408f),
    new float3(0.64223f, 0.0738f, 0.00401f),
    new float3(0.63082f, 0.06868f, 0.00401f),
    new float3(0.61923f, 0.06367f, 0.0041f),
    new float3(0.60746f, 0.05878f, 0.00427f),
    new float3(0.5955f, 0.05399f, 0.00453f),
    new float3(0.58336f, 0.04931f, 0.00486f),
    new float3(0.57103f, 0.04474f, 0.00529f),
    new float3(0.55852f, 0.04028f, 0.00579f),
    new float3(0.54583f, 0.03593f, 0.00638f),
    new float3(0.53295f, 0.03169f, 0.00705f),
    new float3(0.51989f, 0.02756f, 0.0078f),
    new float3(0.50664f, 0.02354f, 0.00863f),
    new float3(0.49321f, 0.01963f, 0.00955f),
    new float3(0.4796f, 0.01583f, 0.01055f)
  };


  public static Color Map(float ot) {
    float t = Mathf.Clamp01(ot);
    int idx = (int)(lookupTable.Length * t - 0.00001f);
    try {
      float3 vec = lookupTable[idx];
      return new Color(vec.x, vec.y, vec.z);
    }
    catch (Exception ) {
      Debug.Log($"{ot} {t} {idx}");
      return Color.clear;
    }
  }
}
