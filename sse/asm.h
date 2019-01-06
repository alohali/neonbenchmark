#ifndef SSE_ASM_H_
#define SSE_ASM_H_
#include <stdint.h>
void   fast_memcpy(uint8_t *dst, const uint8_t *src, uint64_t len);

#endif