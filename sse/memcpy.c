#include <stdio.h>
#include <stdint.h>

/*
https://software.intel.com/sites/landingpage/IntrinsicsGuide/#expand=4058&techs=SSE,SSE2,SSE3,SSSE3,SSE4_1,SSE4_2
x86 Assembly Guide: http://www.cs.virginia.edu/~evans/cs216/guides/x86.html
https://gcc.gnu.org/onlinedocs/gcc/Extended-Asm.html#x86Operandmodifiers

__m128 x1 = _mm_load_ps(vec1_x);
__m128 y1 = _mm_load_ps(vec1_y);
__m128 z1 = _mm_load_ps(vec1_z);
　
__m128 x2 = _mm_load_ps(vec2_x);
__m128 y2 = _mm_load_ps(vec2_y);
__m128 z2 = _mm_load_ps(vec2_z);
　
__m128 t1 = _mm_mul_ps(x1, x2);
__m128 t2 = _mm_mul_ps(y1, y2);
t1 = _mm_add_ps(t1, t2);
t2 = _mm_mul_ps(z1, z2);
t1 = _mm_add_ps(t1, t2);
　
_mm_store_ps(output, t1);

__m128 x1 = _mm_load_ps(vec1_x);
__m128 y1 = _mm_load_ps(vec1_y);
__m128 z1 = _mm_load_ps(vec1_z);
　
__m128 x2 = _mm_load_ps(vec2_x);
__m128 y2 = _mm_load_ps(vec2_y);
__m128 z2 = _mm_load_ps(vec2_z);
　
_mm_prefetch((const char*)(vec1_x + next), _MM_HINT_NTA);
_mm_prefetch((const char*)(vec1_y + next), _MM_HINT_NTA);
_mm_prefetch((const char*)(vec1_z + next), _MM_HINT_NTA);
　
_mm_prefetch((const char*)(vec2_x + next), _MM_HINT_NTA);
_mm_prefetch((const char*)(vec2_y + next), _MM_HINT_NTA);
_mm_prefetch((const char*)(vec2_z + next), _MM_HINT_NTA);
　
__m128 t1 = _mm_mul_ps(x1, x2);
__m128 t2 = _mm_mul_ps(y1, y2);
t1 = _mm_add_ps(t1, t2);
t2 = _mm_mul_ps(z1, z2);
t1 = _mm_add_ps(t1, t2);
　
_mm_stream_ps(output, t1);
*/
static inline void
fast_memcpy(uint8_t *dst, const uint8_t *src, size_t len)
{
    len = 4;
    size_t zero = 0;
    asm volatile (
                    "mov  %%rcx, %[len] \n"
                    "loop:\n"
                    ".align 8\n"
                    "movdqu (%[src]), %%xmm0\n"
                    "movdqu 16(%[src]), %%xmm1\n"
                    "movdqu 32(%[src]), %%xmm2\n"
                    "movdqu 48(%[src]), %%xmm3\n"
                    "movdqu 64(%[src]), %%xmm4\n"
                    "movdqu 80(%[src]), %%xmm5\n"
                    "movdqu 96(%[src]), %%xmm6\n"
                    "movdqu 112(%[src]), %%xmm7\n"
                    "movdqu %%xmm0, (%[dst])\n"
                    "movdqu %%xmm1, 16(%[dst])\n"
                    "movdqu %%xmm2, 32(%[dst])\n"
                    "movdqu %%xmm3, 48(%[dst])\n"
                    "movdqu %%xmm4, 64(%[dst])\n"
                    "dec %%rcx\n"
                    "movdqu %%xmm5, 80(%[dst])\n"
                    "movdqu %%xmm6, 96(%[dst])\n"
                    "movdqu %%xmm7, 112(%[dst])\n"
                    "cmp %%rcx, %3\n"
                    // "call printf\n"
                    "jne loop\n"
                    :
                    : [src] "r" (src),
                      [dst] "r"(dst),
                      [len] "r"(len),
                      "r"(zero)
                    : "xmm0", "xmm1", "xmm2", "xmm3","xmm4", "xmm5", "xmm6", "xmm7", "memory", "rcx");
}

static void * sse_memcpy(void * to, const void * from, size_t len)
{
     void *retval;
     size_t i;
     retval = to;

     /* PREFETCH has effect even for MOVSB instruction ;) */
     __asm__ __volatile__ (
                          "   prefetchnta (%0)\n"
                          "   prefetchnta 64(%0)\n"
                          "   prefetchnta 128(%0)\n"
                          "   prefetchnta 192(%0)\n"
                          "   prefetchnta 256(%0)\n"
                          : : "r" (from) );

                    __asm__ __volatile__ (
                                         "prefetchnta 320(%0)\n"
                                         "movups (%0), %%xmm0\n"
                                         "movups 16(%0), %%xmm1\n"
                                         "movups 32(%0), %%xmm2\n"
                                         "movups 48(%0), %%xmm3\n"
                                         "movntps %%xmm0, (%1)\n"
                                         "movntps %%xmm1, 16(%1)\n"
                                         "movntps %%xmm2, 32(%1)\n"
                                         "movntps %%xmm3, 48(%1)\n"
                                         :: "r" (from), "r" (to) : "memory");


                    __asm__ __volatile__ (
                                         "prefetchnta 320(%0)\n"
                                         "movaps (%0), %%xmm0\n"
                                         "movaps 16(%0), %%xmm1\n"
                                         "movaps 32(%0), %%xmm2\n"
                                         "movaps 48(%0), %%xmm3\n"
                                         "movntps %%xmm0, (%1)\n"
                                         "movntps %%xmm1, 16(%1)\n"
                                         "movntps %%xmm2, 32(%1)\n"
                                         "movntps %%xmm3, 48(%1)\n"
                                         :: "r" (from), "r" (to) : "memory");

          /* since movntq is weakly-ordered, a "sfence"
           * is needed to become ordered again. */
          __asm__ __volatile__ ("sfence":::"memory");
          /* enables to use FPU */
          __asm__ __volatile__ ("emms":::"memory");
     
}

static inline void
fast_memcpy_intrin(uint8_t *dst, const uint8_t *src, size_t len)
{
    len = len / 16;
    for(int i=0; i<len; i++){

    }
}

uint8_t src[1024] = {0};
uint8_t dst[1024] = {0};

int main(int argc, char **argv)
{
    for(int i=0; i<1024;i++)
        src[i] = i%128;
    fast_memcpy(dst, src, 1024);
    
    for(int i=0; i<1024;i++)
    {   
        if(dst[i]!=src[i]){
            printf("error %d %d %d\n", i, src[i], dst[i]);
            return -1;
        }
    }
    printf("done\n");   
    return 0;
}
