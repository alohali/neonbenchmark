    /*
        memcpy.c  copy from directfb
    */
    /*
    * memcpy.c
    * Copyright (C) 1999-2001 Aaron Holtzman <[email]aholtzma@ess.engr.uvic.ca[/email]>
    *
    * This file is part of mpeg2dec, a free MPEG-2 video stream decoder.
    *
    * mpeg2dec is free software; you can redistribute it and/or modify
    * it under the terms of the GNU General Public License as published by
    * the Free Software Foundation; either version 2 of the License, or
    * (at your option) any later version.
    *
    * mpeg2dec is distributed in the hope that it will be useful,
    * but WITHOUT ANY WARRANTY; without even the implied warranty of
    * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    * GNU General Public License for more details.
    *
    * You should have received a copy of the GNU General Public License
    * along with this program; if not, write to the Free Software
    * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
    */

    #include <sys/time.h>
    #include <time.h>

    #include <stdio.h>
    #include <stdlib.h>
    #include <string.h>

    #include "memcpy.h"
    #include "cpu_accel.h"

    #if defined (ARCH_X86) || defined (ARCH_X86_64) || defined (ARCH_PPC) || (SIZEOF_LONG == 8)
    # define RUN_BENCHMARK  1
    #else
    # define RUN_BENCHMARK  0
    #endif

    #if defined (ARCH_X86) || defined (ARCH_X86_64)

    /* for small memory blocks (<256 bytes) this version is faster */
    #define small_memcpy(to,from,n)\
    {\
    register unsigned long int dummy;\
    __asm__ __volatile__(\
      "rep; movsb"\
      :"=&D"(to), "=&S"(from), "=&c"(dummy)\
      :"0" (to), "1" (from),"2" (n)\
      : "memory");\
    }

    /* linux kernel __memcpy (from: /include/asm/string.h) */
    static inline void * __memcpy(void * to, const void * from, size_t n)
    {
         int d0, d1, d2;

         if ( n < 4 ) {
              small_memcpy(to,from,n);
         }
         else
              __asm__ __volatile__(
                                  "rep ; movsl\n\t"
                                  "testb $2,%b4\n\t"
                                  "je 1f\n\t"
                                  "movsw\n"
                                  "1:\ttestb $1,%b4\n\t"
                                  "je 2f\n\t"
                                  "movsb\n"
                                  "2:"
                                  : "=&c" (d0), "=&D" (d1), "=&S" (d2)
                                  :"0" (n/4), "q" (n),"1" ((long) to),"2" ((long) from)
                                  : "memory");

         return(to);
    }

    #ifdef USE_MMX

    #define MMX_MMREG_SIZE 8

    #define MMX1_MIN_LEN 0x800  /* 2K blocks */
    #define MIN_LEN 0x40  /* 64-byte blocks */

    static void * mmx_memcpy(void * to, const void * from, size_t len)
    {
         void *retval;
         size_t i;
         retval = to;

         if (len >= MMX1_MIN_LEN) {
              register unsigned long int delta;
              /* Align destinition to MMREG_SIZE -boundary */
              delta = ((unsigned long int)to)&(MMX_MMREG_SIZE-1);
              if (delta) {
                   delta=MMX_MMREG_SIZE-delta;
                   len -= delta;
                   small_memcpy(to, from, delta);
              }
              i = len >> 6; /* len/64 */
              len&=63;
              for (; i>0; i--) {
                   __asm__ __volatile__ (
                                        "movq (%0), %%mm0\n"
                                        "movq 8(%0), %%mm1\n"
                                        "movq 16(%0), %%mm2\n"
                                        "movq 24(%0), %%mm3\n"
                                        "movq 32(%0), %%mm4\n"
                                        "movq 40(%0), %%mm5\n"
                                        "movq 48(%0), %%mm6\n"
                                        "movq 56(%0), %%mm7\n"
                                        "movq %%mm0, (%1)\n"
                                        "movq %%mm1, 8(%1)\n"
                                        "movq %%mm2, 16(%1)\n"
                                        "movq %%mm3, 24(%1)\n"
                                        "movq %%mm4, 32(%1)\n"
                                        "movq %%mm5, 40(%1)\n"
                                        "movq %%mm6, 48(%1)\n"
                                        "movq %%mm7, 56(%1)\n"
                                        :: "r" (from), "r" (to) : "memory");
                   from +=64;
                   to   +=64;
              }
              __asm__ __volatile__ ("emms":::"memory");
         }
         /*
          * Now do the tail of the block
          */
         if (len) __memcpy(to, from, len);
         return retval;
    }

    #ifdef USE_SSE

    #define SSE_MMREG_SIZE 16

    static void * mmx2_memcpy(void * to, const void * from, size_t len)
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

         if (len >= MIN_LEN) {
              register unsigned long int delta;
              /* Align destinition to MMREG_SIZE -boundary */
              delta = ((unsigned long int)to)&(MMX_MMREG_SIZE-1);
              if (delta) {
                   delta=MMX_MMREG_SIZE-delta;
                   len -= delta;
                   small_memcpy(to, from, delta);
              }
              i = len >> 6; /* len/64 */
              len&=63;
              for (; i>0; i--) {
                   __asm__ __volatile__ (
                                        "prefetchnta 320(%0)\n"
                                        "movq (%0), %%mm0\n"
                                        "movq 8(%0), %%mm1\n"
                                        "movq 16(%0), %%mm2\n"
                                        "movq 24(%0), %%mm3\n"
                                        "movq 32(%0), %%mm4\n"
                                        "movq 40(%0), %%mm5\n"
                                        "movq 48(%0), %%mm6\n"
                                        "movq 56(%0), %%mm7\n"
                                        "movntq %%mm0, (%1)\n"
                                        "movntq %%mm1, 8(%1)\n"
                                        "movntq %%mm2, 16(%1)\n"
                                        "movntq %%mm3, 24(%1)\n"
                                        "movntq %%mm4, 32(%1)\n"
                                        "movntq %%mm5, 40(%1)\n"
                                        "movntq %%mm6, 48(%1)\n"
                                        "movntq %%mm7, 56(%1)\n"
                                        :: "r" (from), "r" (to) : "memory");
                   from += 64;
                   to   += 64;
              }
              /* since movntq is weakly-ordered, a "sfence"
              * is needed to become ordered again. */
              __asm__ __volatile__ ("sfence":::"memory");
              __asm__ __volatile__ ("emms":::"memory");
         }
         /*
          * Now do the tail of the block
          */
         if (len) __memcpy(to, from, len);
         return retval;
    }

    /* SSE note: i tried to move 128 bytes a time instead of 64 but it
    didn't make any measureable difference. i'm using 64 for the sake of
    simplicity. [MF] */
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

         if (len >= MIN_LEN) {
              register unsigned long int delta;
              /* Align destinition to MMREG_SIZE -boundary */
              delta = ((unsigned long int)to)&(SSE_MMREG_SIZE-1);
              if (delta) {
                   delta=SSE_MMREG_SIZE-delta;
                   len -= delta;
                   small_memcpy(to, from, delta);
              }
              i = len >> 6; /* len/64 */
              len&=63;
              if (((unsigned long)from) & 15)
                   /* if SRC is misaligned */
                   for (; i>0; i--) {
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
                        from += 64;
                        to   += 64;
                   }
              else
                   /*
                      Only if SRC is aligned on 16-byte boundary.
                      It allows to use movaps instead of movups, which required
                      data to be aligned or a general-protection exception (#GP)
                      is generated.
                   */
                   for (; i>0; i--) {
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
                        from += 64;
                        to   += 64;
                   }
              /* since movntq is weakly-ordered, a "sfence"
               * is needed to become ordered again. */
              __asm__ __volatile__ ("sfence":::"memory");
              /* enables to use FPU */
              __asm__ __volatile__ ("emms":::"memory");
         }
         /*
          * Now do the tail of the block
          */
         if (len) __memcpy(to, from, len);
         return retval;
    }

    #endif /* USE_SSE */
    #endif /* USE_MMX */


    static void *linux_kernel_memcpy(void *to, const void *from, size_t len) {
         return __memcpy(to,from,len);
    }

    #endif /* ARCH_X86 */


    #if SIZEOF_LONG == 8

    static void * generic64_memcpy( void * to, const void * from, size_t len )
    {
         register u8   *d = (u8*)to;
         register u8   *s = (u8*)from;
         size_t         n;

         if (len >= 128) {
              unsigned long delta;

              /* Align destination to 8-byte boundary */
              delta = (unsigned long)d & 7;
              if (delta) {
                   len -= 8 - delta;                 

                   if ((unsigned long)d & 1) {
                        *d++ = *s++;
                   }
                   if ((unsigned long)d & 2) {
                        *((u16*)d) = *((u16*)s);
                        d += 2; s += 2;
                   }
                   if ((unsigned long)d & 4) {
                        *((unsigned int*)d) = *((unsigned int*)s);
                        d += 4; s += 4;
                   }
              }
             
              n    = len >> 6;
              len &= 63;
             
              for (; n; n--) {
                   ((u64*)d)[0] = ((u64*)s)[0];
                   ((u64*)d)[1] = ((u64*)s)[1];
                   ((u64*)d)[2] = ((u64*)s)[2];
                   ((u64*)d)[3] = ((u64*)s)[3];
                   ((u64*)d)[4] = ((u64*)s)[4];
                   ((u64*)d)[5] = ((u64*)s)[5];
                   ((u64*)d)[6] = ((u64*)s)[6];
                   ((u64*)d)[7] = ((u64*)s)[7];
                   d += 64; s += 64;
              }
         }
         /*
          * Now do the tail of the block
          */
         if (len) {
              n = len >> 3;
             
              for (; n; n--) {
                   *((u64*)d) = *((u64*)s);
                   d += 8; s += 8;
              }
              if (len & 4) {
                   *((unsigned int*)d) = *((unsigned int*)s);
                   d += 4; s += 4;
              }
              if (len & 2)  {
                   *((u16*)d) = *((u16*)s);
                   d += 2; s += 2;
              }
              if (len & 1)
                   *d = *s;
         }
         
         return to;
    }

    #endif /* SIZEOF_LONG == 8 */



    static struct {
         char                 *name;
         char                 *desc;
         memcpy_func           function;
         unsigned long long    time;
         unsigned int          cpu_require;
         int                   best_count;
    } memcpy_method[] =
    {
         { NULL, NULL, NULL, 0, 0},
         { "libc",     "libc memcpy()",             (memcpy_func) memcpy, 0, 0, 0 },
    #if SIZEOF_LONG == 8
         { "generic64","Generic 64bit memcpy()",    generic64_memcpy, 0, 0, 0},
    #endif /* SIZEOF_LONG == 8 */
    #if defined (ARCH_X86) || defined (ARCH_X86_64)
         { "linux",    "linux kernel memcpy()",     linux_kernel_memcpy, 0, 0, 0},
    #ifdef USE_MMX
         { "mmx",      "MMX optimized memcpy()",    mmx_memcpy, 0, MM_MMX, 0},
    #ifdef USE_SSE
         { "mmxext",   "MMXEXT optimized memcpy()", mmx2_memcpy, 0, MM_MMXEXT, 0},
         { "sse",      "SSE optimized memcpy()",    sse_memcpy, 0, MM_MMXEXT|MM_SSE, 0},
    #endif /* USE_SSE  */
    #endif /* USE_MMX  */
    #endif /* ARCH_X86 */
         { NULL, NULL, NULL, 0, 0}
    };


    #ifdef ARCH_X86
    static inline unsigned long long int rdtsc()
    {
         unsigned long long int x;
         __asm__ volatile (".byte 0x0f, 0x31" : "=A" (x));
         return x;
    }
    #else
    static inline unsigned long long int rdtsc()
    {
         struct timeval tv;

         gettimeofday (&tv, NULL);
         return (tv.tv_sec * 1000000 + tv.tv_usec);
    }
    #endif


    memcpy_func seye_memcpy = (memcpy_func) memcpy;

    void seye_find_best_memcpy( int buf_size)
    {
         unsigned long long t;
         char *buf1, *buf2;
         int i, j, best = 0;
         int count=0x3FFFFFF/buf_size;
         unsigned int config_flags = seye_mm_accel();
       
         if ( buf_size>1024000 )
             buf_size=102400;
         if (buf_size>128000 || count>0xFFFFFF)
            count=1000;
         if (!(buf1 = (char *)malloc( buf_size*8 )))
              return;

         if (!(buf2 = (char *)malloc( buf_size*8 ))) {
              free( buf1 );
              return;
         }

         /* make sure buffers are present on physical memory */
         memcpy( buf1, buf2, buf_size*8 );
         memcpy( buf2, buf1, buf_size*8 );

         for (i=1; memcpy_method[i].name; i++) {
              if (memcpy_method[i].cpu_require & ~config_flags)
                   continue;

              t = rdtsc();

              for (j=0; j<count; j++)
                   memcpy_method[i].function( buf1 + (j%8)*buf_size, buf2 + (j%8)*buf_size, buf_size );

              t = rdtsc() - t;
              memcpy_method[i].time = t;

              //fprintf(stderr, "\t%-10s  %20lld\n", memcpy_method[i].name, t );

              if (best == 0 || t < memcpy_method[best].time)
                   best = i;
         }

         if (best) {
              seye_memcpy = memcpy_method[best].function;
         

              fprintf(stderr,"Memcpy: buf_size=%d count=%d Using %s\n", buf_size,count, memcpy_method[best].desc );
              memcpy_method[best].best_count++;
         }

         free( buf1 );
         free( buf2 );
         return;
    }

    void m_seye_find_best_memcpy( int try_count )
    {
        int i, best_index;
        int best_count=0;
        if (try_count<5)
            try_count=5;
        for (i=1; i<=try_count; i++)
        {
            seye_find_best_memcpy( i*10240 );
        }
        for (i=1; memcpy_method[i].name; i++)
        {
            fprintf(stderr,"Memcpy: %s best=%d\n", memcpy_method[i].desc,memcpy_method[i].best_count );
            if (best_count<memcpy_method[i].best_count)
            {
                best_index = i;
                best_count=memcpy_method[i].best_count;
            }
        }
        seye_memcpy = memcpy_method[best_index].function;
        fprintf(stderr,"Memcpy: try %d times, best_count=%d Using %s\n",
                    try_count, best_count, memcpy_method[best_index].desc );
        return;
    }


    void seye_print_memcpy_routines()
    {
         int   i;
         unsigned int unsupported;
         unsigned int config_flags = seye_mm_accel();

         fprintf( stderr, "\nPossible values for memcpy option are:\n\n" );

         for (i=1; memcpy_method[i].name; i++) {
              unsupported = (memcpy_method[i].cpu_require & ~config_flags);

              fprintf( stderr, "  %-10s  %-27s  %s\n", memcpy_method[i].name,
                       memcpy_method[i].desc, unsupported ? "" : "supported" );
         }

         fprintf( stderr, "\n" );
    }

