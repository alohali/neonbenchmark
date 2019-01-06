
#include "asm.h"

void   fast_memcpy(uint8_t *dst, const uint8_t *src, uint64_t len)
{
    len = len / 128;
    asm volatile (
                    "mov  %[len], %%rcx  \n"
                    "mov  %[src], %%r8  \n"
                    "mov  %[dst], %%r9  \n"
                    "loop11:\n"
                    ".align 8\n"
                    "movdqu (%%r8), %%xmm0\n"
                    "movdqu 16(%%r8), %%xmm1\n"
                    "movdqu 32(%%r8), %%xmm2\n"
                    "movdqu 48(%%r8), %%xmm3\n"
                    "movdqu 64(%%r8), %%xmm4\n"
                    "movdqu 80(%%r8), %%xmm5\n"
                    "movdqu 96(%%r8), %%xmm6\n"
                    "movdqu 112(%%r8), %%xmm7\n"
                    "add $0x80, %%r8\n"
                    "movdqu %%xmm0, (%%r9)\n"
                    "movdqu %%xmm1, 16(%%r9)\n"
                    "movdqu %%xmm2, 32(%%r9)\n"
                    "movdqu %%xmm3, 48(%%r9)\n"
                    "movdqu %%xmm4, 64(%%r9)\n"
                    "dec %%rcx\n"
                    "movdqu %%xmm5, 80(%%r9)\n"
                    "movdqu %%xmm6, 96(%%r9)\n"
                    "movdqu %%xmm7, 112(%%r9)\n"
                    "add $0x80, %%r9\n"
                    "cmp $0,%%rcx\n"
                    "jne loop11\n"
                    :
                    : [src] "r" (src),
                      [dst] "r"(dst),
                      [len] "r"(len)
                    : "xmm0", "xmm1", "xmm2", "xmm3","xmm4", "xmm5", "xmm6", "xmm7", "memory", "rcx", "r8", "r9");
}