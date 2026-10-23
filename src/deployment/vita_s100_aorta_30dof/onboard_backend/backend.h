#pragma once
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
#define VITA_BACKEND_ABI 1u
/* No control/publishing API exists. Buffers belong to caller; no clipping. */
enum VitaStatus { VITA_OK=0, VITA_INVALID=1, VITA_BUSY=2, VITA_TIMEOUT=3,
                  VITA_NONFINITE=4, VITA_RUNTIME=5, VITA_POISONED=6 };
enum VitaVelocityBackend { VITA_VELOCITY_ORT=0, VITA_VELOCITY_HBM=1 };
typedef struct { const char* path; const char* sha256; } VitaArtifact;
typedef struct {
 uint32_t abi_version, struct_bytes;
 VitaArtifact front, back, moe, router, pool0, pool1, velocity, velocity_hbm;
 uint32_t timeout_ms; /* positive UCP/worker wait timeout, not a CPU hard deadline */
 uint32_t parallel_router; /* 0 serial reference, 1 persistent worker */
 double deadline_ms; /* diagnostic end-to-end deadline; late outputs still returned */
} VitaConfig;
typedef struct {
 uint32_t abi_version, struct_bytes;
 uint64_t velocity_calls, motion_calls, explicit_resets, rope_resets;
 int64_t motion_step;
 uint32_t poisoned, velocity_hbm_available;
} VitaState;
typedef struct {
 uint32_t abi_version, struct_bytes;
 int64_t step_used, step_next;
 uint64_t velocity_calls, motion_calls, explicit_resets, rope_resets;
 double total_ms, router_ms, frontend_ms, parallel_ms, prepare_ms, moe_ms, backend_ms, output_ms;
 uint32_t deadline_missed, automatic_reset;
} VitaTiming;
/* Hashes required, 64 lowercase hex chars. velocity_hbm optional only if path empty. */
int vita_create(const VitaConfig*,void** handle,char* error,size_t error_bytes);
/* These calls reject concurrent use. Caller also serializes destroy vs other calls. */
int vita_reset(void* handle,char* error,size_t error_bytes);
int vita_state(void* handle,VitaState*,char* error,size_t error_bytes);
int vita_velocity(void* handle,const float* obs,size_t obs_count,int backend,
                  float* actions,size_t action_count,VitaTiming*,char* error,size_t error_bytes);
int vita_motion(void* handle,const float* obs,size_t obs_count,
                float* actions,size_t action_count,
                float* cache,size_t cache_count,int64_t* routes,size_t route_count,
                float* logits,size_t logit_count,VitaTiming*,char* error,size_t error_bytes);
/* On timeout/busy, handle remains LIVE; retry later. Never free active DMA memory. */
int vita_destroy(void* handle,char* error,size_t error_bytes);
const char* vita_build_info(void);
#ifdef __cplusplus
}
#endif
