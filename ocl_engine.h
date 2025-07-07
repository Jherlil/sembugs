#ifndef OCL_ENGINE_H
#define OCL_ENGINE_H
#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifndef NO_OPENCL
int ocl_init(int requested_shaders);
void ocl_cleanup();
size_t ocl_max_shaders();
int ocl_sha256_33(const uint8_t *input, uint8_t *digest);
int ocl_sha256_batch_33(const uint8_t *inputs, size_t n, uint8_t *digests);
int ocl_ripemd160_batch_32(const uint8_t *inputs, size_t n, uint8_t *digests);
#else
static inline int ocl_init(int requested_shaders){ (void)requested_shaders; return 0; }
static inline void ocl_cleanup(){ }
static inline size_t ocl_max_shaders(){ return 0; }
static inline int ocl_sha256_33(const uint8_t *i, uint8_t *o){ (void)i;(void)o; return 0; }
static inline int ocl_sha256_batch_33(const uint8_t *i,size_t n,uint8_t *o){(void)i;(void)n;(void)o;return 0;}
static inline int ocl_ripemd160_batch_32(const uint8_t *i,size_t n,uint8_t *o){(void)i;(void)n;(void)o;return 0;}
#endif

#ifdef __cplusplus
}
#endif

#endif