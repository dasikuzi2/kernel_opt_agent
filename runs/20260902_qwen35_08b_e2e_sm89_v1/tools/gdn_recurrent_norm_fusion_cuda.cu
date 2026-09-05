#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>

namespace {

__device__ __forceinline__ float warp_sum(float value) {
  value += __shfl_down_sync(0xffffffffu, value, 16);
  value += __shfl_down_sync(0xffffffffu, value, 8);
  value += __shfl_down_sync(0xffffffffu, value, 4);
  value += __shfl_down_sync(0xffffffffu, value, 2);
  value += __shfl_down_sync(0xffffffffu, value, 1);
  return value;
}

// Fixed production shape for Qwen3.5-0.8B packed GDN decode:
// B=1, H=HV=16, K=V=128, BF16 recurrent state.
__global__ void gdn_recurrent_norm_fused_sm89(
    const __nv_bfloat16* __restrict__ mixed_qkv,
    const __nv_bfloat16* __restrict__ a,
    const __nv_bfloat16* __restrict__ b,
    const float* __restrict__ A_log,
    const float* __restrict__ dt_bias,
    const __nv_bfloat16* __restrict__ gate,
    const __nv_bfloat16* __restrict__ weight,
    __nv_bfloat16* __restrict__ state,
    const int32_t* __restrict__ state_indices,
    __nv_bfloat16* __restrict__ out,
    float eps) {
  constexpr int H = 16;
  constexpr int HV = 16;
  constexpr int K = 128;
  constexpr int V = 128;
  constexpr float scale = 0.08838834764831845f;

  const int head = blockIdx.x % HV;
  const int token = blockIdx.x / HV;
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int state_idx = state_indices[token];
  __shared__ float q[K];
  __shared__ float k[K];
  __shared__ __nv_bfloat16 recurrent[V];
  __shared__ float reduction[V];

  if (state_idx <= 0) {
    out[(token * HV + head) * V + threadIdx.x] = __float2bfloat16_rn(0.0f);
    return;
  }

  // Warp zero deliberately owns the q/k norm.  Keeping this reduction on one
  // warp avoids the changed state transition observed when generic Triton
  // num_warps was increased.
  if (warp == 0) {
    float qv[4];
    float kv[4];
    float qsum = 0.0f;
    float ksum = 0.0f;
#pragma unroll
    for (int item = 0; item < 4; ++item) {
      const int col = lane + item * 32;
      qv[item] = __bfloat162float(mixed_qkv[token * 6144 + head * K + col]);
      kv[item] = __bfloat162float(
          mixed_qkv[token * 6144 + H * K + head * K + col]);
      qsum += qv[item] * qv[item];
      ksum += kv[item] * kv[item];
    }
    qsum = warp_sum(qsum);
    ksum = warp_sum(ksum);
    const float qinv = rsqrtf(__shfl_sync(0xffffffffu, qsum, 0) + 1.0e-6f);
    const float kinv = rsqrtf(__shfl_sync(0xffffffffu, ksum, 0) + 1.0e-6f);
#pragma unroll
    for (int item = 0; item < 4; ++item) {
      const int col = lane + item * 32;
      q[col] = qv[item] * qinv * scale;
      k[col] = kv[item] * kinv;
    }
  }
  __syncthreads();

  const float x = __bfloat162float(a[token * HV + head]) + dt_bias[head];
  const float softplus = x <= 20.0f ? logf(1.0f + expf(x)) : x;
  const float decay = expf(-expf(A_log[head]) * softplus);
  const float bv = __bfloat162float(b[token * HV + head]);
  const float beta = 1.0f / (1.0f + expf(-bv));
  const int row_base = warp * 32;
  const int state_base = (state_idx * HV + head) * V * K;

  // Each warp owns one stock BV32 slice.  A lane holds four K columns for one
  // row, so state traffic is coalesced and the row dot products use warp
  // shuffles rather than serial scalar loops.
#pragma unroll 1
  for (int local_row = 0; local_row < 32; ++local_row) {
    const int row = row_base + local_row;
    float h[4];
    float dot_k = 0.0f;
#pragma unroll
    for (int item = 0; item < 4; ++item) {
      const int col = lane + item * 32;
      h[item] = __bfloat162float(state[state_base + row * K + col]) * decay;
      dot_k += h[item] * k[col];
    }
    dot_k = warp_sum(dot_k);
    dot_k = __shfl_sync(0xffffffffu, dot_k, 0);
    const float value = __bfloat162float(
        mixed_qkv[token * 6144 + 2 * H * K + head * V + row]);
    const float correction = (value - dot_k) * beta;
    float dot_q = 0.0f;
#pragma unroll
    for (int item = 0; item < 4; ++item) {
      const int col = lane + item * 32;
      h[item] += correction * k[col];
      state[state_base + row * K + col] = __float2bfloat16_rn(h[item]);
      dot_q += h[item] * q[col];
    }
    dot_q = warp_sum(dot_q);
    if (lane == 0) {
      recurrent[row] = __float2bfloat16_rn(dot_q);
    }
  }
  __syncthreads();

  const int col = threadIdx.x;
  const float recurrent_value = __bfloat162float(recurrent[col]);
  reduction[col] = recurrent_value * recurrent_value;
  __syncthreads();
  for (int stride = 64; stride > 0; stride >>= 1) {
    if (col < stride) {
      reduction[col] += reduction[col + stride];
    }
    __syncthreads();
  }
  const float inv_rms = rsqrtf(reduction[0] / 128.0f + eps);
  const float gate_value = __bfloat162float(gate[(token * HV + head) * V + col]);
  const float silu_gate = gate_value / (1.0f + expf(-gate_value));
  const float weight_value = __bfloat162float(weight[col]);
  out[(token * HV + head) * V + col] = __float2bfloat16_rn(
      recurrent_value * inv_rms * weight_value * silu_gate);
}

}  // namespace

extern "C" int launch_gdn_recurrent_norm_fused_sm89(
    const void* mixed_qkv,
    const void* a,
    const void* b,
    const void* A_log,
    const void* dt_bias,
    const void* gate,
    const void* weight,
    void* state,
    const void* state_indices,
    void* out,
    float eps,
    int batch,
    void* stream) {
  gdn_recurrent_norm_fused_sm89<<<batch * 16, 128, 0,
                                  reinterpret_cast<cudaStream_t>(stream)>>>(
      static_cast<const __nv_bfloat16*>(mixed_qkv),
      static_cast<const __nv_bfloat16*>(a),
      static_cast<const __nv_bfloat16*>(b),
      static_cast<const float*>(A_log),
      static_cast<const float*>(dt_bias),
      static_cast<const __nv_bfloat16*>(gate),
      static_cast<const __nv_bfloat16*>(weight),
      static_cast<__nv_bfloat16*>(state),
      static_cast<const int32_t*>(state_indices),
      static_cast<__nv_bfloat16*>(out),
      eps);
  return static_cast<int>(cudaGetLastError());
}
