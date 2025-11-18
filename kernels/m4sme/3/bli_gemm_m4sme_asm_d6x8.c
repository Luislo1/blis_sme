/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas at Austin
   Copyright (C) 2021, The University of Tokyo

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions are
   met:
    - Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    - Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    - Neither the name(s) of the copyright holder(s) nor the names of its
      contributors may be used to endorse or promote products derived
      from this software without specific prior written permission.

   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
   "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
   A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
   HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
   SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
   LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
   DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
   THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
   (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
   OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

*/

#include "blis.h"
#include <arm_sme.h>
#include "m4sme_asm_utils.h"

#define SVPRFOP_READ   0
#define SVPRFOP_WRITE  1
#define SVPRFOP_STRM   2
#define SVPRFOP_NOP    3

// #define DISPLAY_DEBUG_INFO

// Added prefetch fix for non-cacheline aligned C columns
// (with the prefetches interleaved with other instructions)
// to both sgemm and dgemm versions.  Previously, only the
// first cacheline in each column was prefetched.

// Added sgemm prefetch fix for non-cacheline aligned C columns
// (with the prefetches interleaved with other instructions)

/*
   o 4x4 Single precision micro-kernel fully functional.
   o Runnable on ARMv8, compiled with aarch64 GCC.
   o Use it together with the armv8 BLIS configuration.
   o Tested on Juno board. Around 7.3 GFLOPS @ 1.1 GHz.

   December 2014.

 * UPDATE NOVEMBER 2015
 * Micro-kernel changed to 8x12
 * Tested on Juno Board. Around  8.1 GFLOPS, 1 x A57 core  @ 1.1 GHz.
 * Tested on Juno Board. Around 15.9 GFLOPS, 2 x A57 cores @ 1.1 GHz.
 * Tested on Juno board. Around  3.1 GFLOPS, 1 x A53 core  @ 850 MHz.
 * Tested on Juno board. Around 12   GFLOPS, 4 x A53 cores @ 850 MHz.

 * UPDATE JULY 2021 - Leick Robinson
 * Both Microkernels changed to fix two prefetching performance bugs
 * Tested on 2s Altra.   Around 6,900 GFLOPS, 160 x N2 cores @ 3.0 GHz
 * Tested on 1s Altra Max. Arnd 5,800 GFLOPS. 128 x N2 cores @ 3.0 GHz
*/

#if 0
void bli_sgemm_m4sme_asm_8x12_impl
     (
             dim_t      m,
             dim_t      n,
             dim_t      k,
       const void*      alpha,
       const void*      a,
       const void*      b,
       const void*      beta,
             void*      c, inc_t rs_c0, inc_t cs_c0,
       const auxinfo_t* data,
       const cntx_t*    cntx
     ) __arm_streaming __arm_inout("za")
{
	float * a_ = (float*)a;
	float * b_ = (float*)b;
	float * c_ = (float*)c;
 	uint64_t SVL = svcntsw();
//printf("SVL: %d. m=%lld. n=%lld. k=%lld\n", SVL, m, n, k);
//printf("alpha: %f. beta=%f\n", *(float*)alpha, *(float*)beta);
//printf("rs_c0: %d. cs_c0=%d\n", rs_c0, cs_c0);
	svbool_t pMDim = svwhilelt_b32(m, m);
	svbool_t pNDim = svwhilelt_b32(n, n);

	svzero_za();

	for (uint64_t k_ = 0; k_ < k; k_++) {
                svfloat32_t zL = svld1(svptrue_b32(), (float32_t*)(&a_[k_ * SVL]));
                svfloat32_t zR = svld1(svptrue_b32(), (float32_t*)(&b_[k_ * n  ]));
                svmopa_za32_m(0, svptrue_b32(), svptrue_b32(), zL, zR);
	}

	// Store ZA to matResult.
        const uint64_t result_tile_UL_corner = 0;

        for (uint64_t tcol = 0; tcol < SVL; tcol += 4) {
                svbool_t p0 = svpsel_lane_b32(pNDim, pMDim, tcol + 0);
                svbool_t p1 = svpsel_lane_b32(pNDim, pMDim, tcol + 1);
                svbool_t p2 = svpsel_lane_b32(pNDim, pMDim, tcol + 2);
                svbool_t p3 = svpsel_lane_b32(pNDim, pMDim, tcol + 3);

		//printf("tcol: %d\n", tcol);
                svst1_ver_za32(
                    /* tile: */ 0, /* slice: */ tcol + 0, svptrue_b32(),
                    &c_[result_tile_UL_corner + (tcol + 0) * cs_c0]);
                svst1_ver_za32(
                    /* tile: */ 0, /* slice: */ tcol + 1, svptrue_b32(),
                    &c_[result_tile_UL_corner + (tcol + 1) * cs_c0]);
                svst1_ver_za32(
                    /* tile: */ 0, /* slice: */ tcol + 2, svptrue_b32(),
                    &c_[result_tile_UL_corner + (tcol + 2) * cs_c0]);
                svst1_ver_za32(
                    /* tile: */ 0, /* slice: */ tcol + 3, svptrue_b32(),
                    &c_[result_tile_UL_corner + (tcol + 3) * cs_c0]);
        }
	return;
}
#endif

__arm_new("za") __arm_locally_streaming void bli_sgemm_m4sme_asm_8x12
     (
             dim_t      m,
             dim_t      n,
             dim_t      k,
       const void*      alpha,
       const void*      a,
       const void*      b,
       const void*      beta,
             void*      c, inc_t rs_c, inc_t cs_c,
       const auxinfo_t* data,
       const cntx_t*    cntx
     ) 
{
	GEMM_UKR_SETUP_CT_AMBI( s, 16, 64, false );
	float * a_ = (float*)a;
	float * b_ = (float*)b;
 	uint64_t SVL = svcntsw();
//printf("SVL: %d. m=%lld. n=%lld. k=%lld\n", SVL, m, n, k);
//printf("alpha: %f. beta=%f\n", *(float*)alpha, *(float*)beta);
//printf("rs_c: %d. cs_c=%d\n", rs_c, cs_c);
	//svbool_t pMDim = svwhilelt_b32(m, m);
	//svbool_t pNDim = svwhilelt_b32(n, n);

	const void* a_next = bli_auxinfo_next_a( data );
	const void* b_next = bli_auxinfo_next_b( data );

	float * c_ = (float*)c;
	//int vnum = cs_c / SVL;

	svzero_za();


	uint64_t k_;
	uint64_t k_iter = k/4;
	uint64_t k_left = k%4;

	for (k_ = 0; k_ < k_iter; k_++) {
// Aplica prefetch explícito para lectura
	//svprfb(svptrue_b32(), &a_[(k_+1) * (2*SVL)      ], 0);
	//svprfb(svptrue_b32(), &a_[(k_+2) * (2*SVL)      ], 0);

	//svprfb(svptrue_b32(), &b_[(k_+1) * (2*SVL)      ], 0);
	//svprfb(svptrue_b32(), &b_[(k_+2) * (2*SVL)      ], 0);

// Loads.
		svfloat32x4_t zL00 = svld1_f32_x4(svptrue_c32(), (float32_t*)(&a_[0      ]));

		svfloat32x4_t zR00 = svld1_f32_x4(svptrue_c32(), (float32_t*)(&b_[0     ]));

	// svprfb(svptrue_b32(), &a_[(k_+3) * (SVL)      ], 0);
// Prefetch (dudo si funciona).
		svmopa_za32_m(0, svptrue_b32(), svptrue_b32(), svget4(zL00, 0), svget4(zR00, 0));
// Prefetch (dudo si funciona).
	// svprfb(svptrue_b32(), &b_[(k_+3) * (4*SVL)      ], 0);
		svmopa_za32_m(1, svptrue_b32(), svptrue_b32(), svget4(zL00, 0), svget4(zR00, 1));

		svfloat32x4_t zR01 = svld1_f32_x4(svptrue_c32(), (float32_t*)(&b_[(4*SVL)      ]));

		svmopa_za32_m(2, svptrue_b32(), svptrue_b32(), svget4(zL00, 0), svget4(zR00, 2));
		svmopa_za32_m(3, svptrue_b32(), svptrue_b32(), svget4(zL00, 0), svget4(zR00, 3));

		svmopa_za32_m(0, svptrue_b32(), svptrue_b32(), svget4(zL00, 1), svget4(zR01, 0));
		svmopa_za32_m(1, svptrue_b32(), svptrue_b32(), svget4(zL00, 1), svget4(zR01, 1));

		svfloat32x4_t zR02 = svld1_f32_x4(svptrue_c32(), (float32_t*)(&b_[2 * (4*SVL)      ]));

		svmopa_za32_m(2, svptrue_b32(), svptrue_b32(), svget4(zL00, 1), svget4(zR01, 2));
		svmopa_za32_m(3, svptrue_b32(), svptrue_b32(), svget4(zL00, 1), svget4(zR01, 3));


		svmopa_za32_m(0, svptrue_b32(), svptrue_b32(), svget4(zL00, 2), svget4(zR02,0));
		svmopa_za32_m(1, svptrue_b32(), svptrue_b32(), svget4(zL00, 2), svget4(zR02,1));

		svfloat32x4_t zR03 = svld1_f32_x4(svptrue_c32(), (float32_t*)(&b_[3 * (4*SVL)      ]));

		svmopa_za32_m(2, svptrue_b32(), svptrue_b32(), svget4(zL00, 2), svget4(zR02,2));
		svmopa_za32_m(3, svptrue_b32(), svptrue_b32(), svget4(zL00, 2), svget4(zR02,3));


		svmopa_za32_m(0, svptrue_b32(), svptrue_b32(), svget4(zL00, 3), svget4(zR03,0));
// Prefetch (dudo si funciona).
		svprfb(svptrue_b32(), (float*)&a_next, 0);
		svmopa_za32_m(1, svptrue_b32(), svptrue_b32(), svget4(zL00, 3), svget4(zR03,1));
// Prefetch (dudo si funciona).
		svprfb(svptrue_b32(), (float*)&b_next, 0);
		svmopa_za32_m(2, svptrue_b32(), svptrue_b32(), svget4(zL00, 3), svget4(zR03,2));
		svmopa_za32_m(3, svptrue_b32(), svptrue_b32(), svget4(zL00, 3), svget4(zR03,3));
		a_ += (4*SVL);
		b_ += (4*4*SVL);
	}

	for (k_=0; k_ < k_left; k_+=1) {
		svfloat32_t zL00 = svld1_f32(svptrue_b32(), (float32_t*)(&a_[0]));
		svfloat32x4_t zR00 = svld1_f32_x4(svptrue_c32(), (float32_t*)(&b_[0]));

		svmopa_za32_m(0, svptrue_b32(), svptrue_b32(), zL00, svget4(zR00,0));
		svmopa_za32_m(1, svptrue_b32(), svptrue_b32(), zL00, svget4(zR00,1));

		svmopa_za32_m(2, svptrue_b32(), svptrue_b32(), zL00, svget4(zR00,2));
		svmopa_za32_m(3, svptrue_b32(), svptrue_b32(), zL00, svget4(zR00,3));

		a_ += (SVL);
		b_ += (4*SVL);
	}
		// Store ZA to matResult.


		float beta_ = *(float*) beta;
		float alpha_ = *(float*) alpha;
        const uint64_t result_tile_TL_corner = 0;
        const uint64_t result_tile_BL_corner = SVL * cs_c;
        const uint64_t result_tile_TR_corner = SVL * 2 * cs_c;
        const uint64_t result_tile_BR_corner = SVL * 3 * cs_c;
		svfloat32_t zbeta = svdup_f32(beta_);
		svfloat32_t zalpha = svdup_f32(alpha_);
		if (rs_c == 1) {
			if (beta_ == 0) {
				for (uint64_t tcol = 0; tcol < SVL; tcol += 4) {
					//Read ZA slices into Z regs
					svfloat32_t z0 = svread_ver_za32_m(
						z0, svptrue_b32(),
						/* tile: */ 0, /* slice: */ tcol + 0);
					svfloat32_t z1 = svread_ver_za32_m(
						z1, svptrue_b32(),
						/* tile: */ 1, /* slice: */ tcol + 0);
					svfloat32_t z2 = svread_ver_za32_m(
						z2, svptrue_b32(),
						/* tile: */ 2, /* slice: */ tcol + 0);
					svfloat32_t z3 = svread_ver_za32_m(
						z3, svptrue_b32(),
						/* tile: */ 3, /* slice: */ tcol + 0);

					//Scale Z regs by broadcast alpha
					z0 = svmul_f32_z(svptrue_b32(), z0, zalpha);
					z1 = svmul_f32_z(svptrue_b32(), z1, zalpha);
					z2 = svmul_f32_z(svptrue_b32(), z2, zalpha);
					z3 = svmul_f32_z(svptrue_b32(), z3, zalpha);

					//Store full result into C

					svst1(svptrue_b32(), &c_[result_tile_TL_corner + (tcol + 0) * cs_c], z0);
					svst1(svptrue_b32(), &c_[result_tile_BL_corner + (tcol + 0) * cs_c], z1);
					svst1(svptrue_b32(), &c_[result_tile_TR_corner + (tcol + 0) * cs_c], z2);
					svst1(svptrue_b32(), &c_[result_tile_BR_corner + (tcol + 0) * cs_c], z3);
					
					z0 = svread_ver_za32_m(
						z0, svptrue_b32(),
						/* tile: */ 0, /* slice: */ tcol + 1);
					z1 = svread_ver_za32_m(
						z1, svptrue_b32(),
						/* tile: */ 1, /* slice: */ tcol + 1);
					z2 = svread_ver_za32_m(
						z2, svptrue_b32(),
						/* tile: */ 2, /* slice: */ tcol + 1);
					z3 = svread_ver_za32_m(
						z3, svptrue_b32(),
						/* tile: */ 3, /* slice: */ tcol + 1);


					z0 = svmul_f32_z(svptrue_b32(), z0, zalpha);
					z1 = svmul_f32_z(svptrue_b32(), z1, zalpha);
					z2 = svmul_f32_z(svptrue_b32(), z2, zalpha);
					z3 = svmul_f32_z(svptrue_b32(), z3, zalpha);
					
					svst1(svptrue_b32(), &c_[result_tile_TL_corner + (tcol + 1) * cs_c], z0);
					svst1(svptrue_b32(), &c_[result_tile_BL_corner + (tcol + 1) * cs_c], z1);
					svst1(svptrue_b32(), &c_[result_tile_TR_corner + (tcol + 1) * cs_c], z2);
					svst1(svptrue_b32(), &c_[result_tile_BR_corner + (tcol + 1) * cs_c], z3);


					z0 = svread_ver_za32_m(
						z0, svptrue_b32(),
						/* tile: */ 0, /* slice: */ tcol + 2);
					z1 = svread_ver_za32_m(
						z1, svptrue_b32(),
						/* tile: */ 1, /* slice: */ tcol + 2);
					z2 = svread_ver_za32_m(
						z2, svptrue_b32(),
						/* tile: */ 2, /* slice: */ tcol + 2);
					z3 = svread_ver_za32_m(
						z3, svptrue_b32(),
						/* tile: */ 3, /* slice: */ tcol + 2);


					z0 = svmul_f32_z(svptrue_b32(), z0, zalpha);
					z1 = svmul_f32_z(svptrue_b32(), z1, zalpha);
					z2 = svmul_f32_z(svptrue_b32(), z2, zalpha);
					z3 = svmul_f32_z(svptrue_b32(), z3, zalpha);	
					
					svst1(svptrue_b32(), &c_[result_tile_TL_corner + (tcol + 2) * cs_c], z0);
					svst1(svptrue_b32(), &c_[result_tile_BL_corner + (tcol + 2) * cs_c], z1);
					svst1(svptrue_b32(), &c_[result_tile_TR_corner + (tcol + 2) * cs_c], z2);
					svst1(svptrue_b32(), &c_[result_tile_BR_corner + (tcol + 2) * cs_c], z3);

					z0 = svread_ver_za32_m(
						z0, svptrue_b32(),
						/* tile: */ 0, /* slice: */ tcol + 3);
					z1 = svread_ver_za32_m(
						z1, svptrue_b32(),
						/* tile: */ 1, /* slice: */ tcol + 3);
					z2 = svread_ver_za32_m(
						z2, svptrue_b32(),
						/* tile: */ 2, /* slice: */ tcol + 3);
					z3 = svread_ver_za32_m(
						z3, svptrue_b32(),
						/* tile: */ 3, /* slice: */ tcol + 3);


					z0 = svmul_f32_z(svptrue_b32(), z0, zalpha);
					z1 = svmul_f32_z(svptrue_b32(), z1, zalpha);
					z2 = svmul_f32_z(svptrue_b32(), z2, zalpha);
					z3 = svmul_f32_z(svptrue_b32(), z3, zalpha);
					
					svst1(svptrue_b32(), &c_[result_tile_TL_corner + (tcol + 3) * cs_c], z0);
					svst1(svptrue_b32(), &c_[result_tile_BL_corner + (tcol + 3) * cs_c], z1);
					svst1(svptrue_b32(), &c_[result_tile_TR_corner + (tcol + 3) * cs_c], z2);
					svst1(svptrue_b32(), &c_[result_tile_BR_corner + (tcol + 3) * cs_c], z3);
			}
		}
		else {
				for (uint64_t tcol = 0; tcol < SVL; tcol += 4) {
					//Read ZA slices into Z regs
					svfloat32_t z0 = svread_ver_za32_m(
						z0, svptrue_b32(),
						/* tile: */ 0, /* slice: */ tcol + 0);
					svfloat32_t z1 = svread_ver_za32_m(
						z1, svptrue_b32(),
						/* tile: */ 1, /* slice: */ tcol + 0);
					svfloat32_t z2 = svread_ver_za32_m(
						z2, svptrue_b32(),
						/* tile: */ 2, /* slice: */ tcol + 0);
					svfloat32_t z3 = svread_ver_za32_m(
						z3, svptrue_b32(),
						/* tile: */ 3, /* slice: */ tcol + 0);

					//Scale Z regs by broadcast alpha
					z0 = svmul_f32_z(svptrue_b32(), z0, zalpha);
					z1 = svmul_f32_z(svptrue_b32(), z1, zalpha);
					z2 = svmul_f32_z(svptrue_b32(), z2, zalpha);
					z3 = svmul_f32_z(svptrue_b32(), z3, zalpha);
						
					//Load C into Z regs
					svfloat32_t z4 = svld1_f32(svptrue_b32(), &c_[result_tile_TL_corner + (tcol + 0) * cs_c]);
					svfloat32_t z5 = svld1_f32(svptrue_b32(), &c_[result_tile_BL_corner + (tcol + 0) * cs_c]);
					svfloat32_t z6 = svld1_f32(svptrue_b32(), &c_[result_tile_TR_corner + (tcol + 0) * cs_c]);
					svfloat32_t z7 = svld1_f32(svptrue_b32(), &c_[result_tile_BR_corner + (tcol + 0) * cs_c]);

					//Scale Z regs by broadcast beta
					z4 = svmla_m(svptrue_b32(), z0, z4, zbeta);
					z5 = svmla_m(svptrue_b32(), z1, z5, zbeta);
					z6 = svmla_m(svptrue_b32(), z2, z6, zbeta);
					z7 = svmla_m(svptrue_b32(), z3, z7, zbeta);

					//Store full result into C

					svst1(svptrue_b32(), &c_[result_tile_TL_corner + (tcol + 0) * cs_c], z4);
					svst1(svptrue_b32(), &c_[result_tile_BL_corner + (tcol + 0) * cs_c], z5);
					svst1(svptrue_b32(), &c_[result_tile_TR_corner + (tcol + 0) * cs_c], z6);
					svst1(svptrue_b32(), &c_[result_tile_BR_corner + (tcol + 0) * cs_c], z7);
					
					z0 = svread_ver_za32_m(
						z0, svptrue_b32(),
						/* tile: */ 0, /* slice: */ tcol + 1);
					z1 = svread_ver_za32_m(
						z1, svptrue_b32(),
						/* tile: */ 1, /* slice: */ tcol + 1);
					z2 = svread_ver_za32_m(
						z2, svptrue_b32(),
						/* tile: */ 2, /* slice: */ tcol + 1);
					z3 = svread_ver_za32_m(
						z3, svptrue_b32(),
						/* tile: */ 3, /* slice: */ tcol + 1);


					z0 = svmul_f32_z(svptrue_b32(), z0, zalpha);
					z1 = svmul_f32_z(svptrue_b32(), z1, zalpha);
					z2 = svmul_f32_z(svptrue_b32(), z2, zalpha);
					z3 = svmul_f32_z(svptrue_b32(), z3, zalpha);




					z4 = svld1_f32(svptrue_b32(), &c_[result_tile_TL_corner + (tcol + 1) * cs_c]);
					z5 = svld1_f32(svptrue_b32(), &c_[result_tile_BL_corner + (tcol + 1) * cs_c]);
					z6 = svld1_f32(svptrue_b32(), &c_[result_tile_TR_corner + (tcol + 1) * cs_c]);
					z7 = svld1_f32(svptrue_b32(), &c_[result_tile_BR_corner + (tcol + 1) * cs_c]);

					z4 = svmla_m(svptrue_b32(), z0, z4, zbeta);
					z5 = svmla_m(svptrue_b32(), z1, z5, zbeta);
					z6 = svmla_m(svptrue_b32(), z2, z6, zbeta);
					z7 = svmla_m(svptrue_b32(), z3, z7, zbeta);
					
					svst1(svptrue_b32(), &c_[result_tile_TL_corner + (tcol + 1) * cs_c], z4);
					svst1(svptrue_b32(), &c_[result_tile_BL_corner + (tcol + 1) * cs_c], z5);
					svst1(svptrue_b32(), &c_[result_tile_TR_corner + (tcol + 1) * cs_c], z6);
					svst1(svptrue_b32(), &c_[result_tile_BR_corner + (tcol + 1) * cs_c], z7);


					z0 = svread_ver_za32_m(
						z0, svptrue_b32(),
						/* tile: */ 0, /* slice: */ tcol + 2);
					z1 = svread_ver_za32_m(
						z1, svptrue_b32(),
						/* tile: */ 1, /* slice: */ tcol + 2);
					z2 = svread_ver_za32_m(
						z2, svptrue_b32(),
						/* tile: */ 2, /* slice: */ tcol + 2);
					z3 = svread_ver_za32_m(
						z3, svptrue_b32(),
						/* tile: */ 3, /* slice: */ tcol + 2);


					z0 = svmul_f32_z(svptrue_b32(), z0, zalpha);
					z1 = svmul_f32_z(svptrue_b32(), z1, zalpha);
					z2 = svmul_f32_z(svptrue_b32(), z2, zalpha);
					z3 = svmul_f32_z(svptrue_b32(), z3, zalpha);


					z4 = svld1_f32(svptrue_b32(), &c_[result_tile_TL_corner + (tcol + 2) * cs_c]);
					z5 = svld1_f32(svptrue_b32(), &c_[result_tile_BL_corner + (tcol + 2) * cs_c]);
					z6 = svld1_f32(svptrue_b32(), &c_[result_tile_TR_corner + (tcol + 2) * cs_c]);
					z7 = svld1_f32(svptrue_b32(), &c_[result_tile_BR_corner + (tcol + 2) * cs_c]);

					z4 = svmla_m(svptrue_b32(), z0, z4, zbeta);
					z5 = svmla_m(svptrue_b32(), z1, z5, zbeta);
					z6 = svmla_m(svptrue_b32(), z2, z6, zbeta);
					z7 = svmla_m(svptrue_b32(), z3, z7, zbeta);			
					
					svst1(svptrue_b32(), &c_[result_tile_TL_corner + (tcol + 2) * cs_c], z4);
					svst1(svptrue_b32(), &c_[result_tile_BL_corner + (tcol + 2) * cs_c], z5);
					svst1(svptrue_b32(), &c_[result_tile_TR_corner + (tcol + 2) * cs_c], z6);
					svst1(svptrue_b32(), &c_[result_tile_BR_corner + (tcol + 2) * cs_c], z7);

					z0 = svread_ver_za32_m(
						z0, svptrue_b32(),
						/* tile: */ 0, /* slice: */ tcol + 3);
					z1 = svread_ver_za32_m(
						z1, svptrue_b32(),
						/* tile: */ 1, /* slice: */ tcol + 3);
					z2 = svread_ver_za32_m(
						z2, svptrue_b32(),
						/* tile: */ 2, /* slice: */ tcol + 3);
					z3 = svread_ver_za32_m(
						z3, svptrue_b32(),
						/* tile: */ 3, /* slice: */ tcol + 3);


					z0 = svmul_f32_z(svptrue_b32(), z0, zalpha);
					z1 = svmul_f32_z(svptrue_b32(), z1, zalpha);
					z2 = svmul_f32_z(svptrue_b32(), z2, zalpha);
					z3 = svmul_f32_z(svptrue_b32(), z3, zalpha);


					z4 = svld1_f32(svptrue_b32(), &c_[result_tile_TL_corner + (tcol + 3) * cs_c]);
					z5 = svld1_f32(svptrue_b32(), &c_[result_tile_BL_corner + (tcol + 3) * cs_c]);
					z6 = svld1_f32(svptrue_b32(), &c_[result_tile_TR_corner + (tcol + 3) * cs_c]);
					z7 = svld1_f32(svptrue_b32(), &c_[result_tile_BR_corner + (tcol + 3) * cs_c]);

					z4 = svmla_m(svptrue_b32(), z0, z4, zbeta);
					z5 = svmla_m(svptrue_b32(), z1, z5, zbeta);
					z6 = svmla_m(svptrue_b32(), z2, z6, zbeta);
					z7 = svmla_m(svptrue_b32(), z3, z7, zbeta);
					
					svst1(svptrue_b32(), &c_[result_tile_TL_corner + (tcol + 3) * cs_c], z4);
					svst1(svptrue_b32(), &c_[result_tile_BL_corner + (tcol + 3) * cs_c], z5);
					svst1(svptrue_b32(), &c_[result_tile_TR_corner + (tcol + 3) * cs_c], z6);
					svst1(svptrue_b32(), &c_[result_tile_BR_corner + (tcol + 3) * cs_c], z7);
			}
		}
	}
	else {

		if (beta_ == 0) {
			for (uint64_t tcol = 0; tcol < SVL; tcol += 4) {
				//Read ZA slices into Z regs
				svfloat32_t z0 = svread_hor_za32_m(
					z0, svptrue_b32(),
					/* tile: */ 0, /* slice: */ tcol + 0);
				svfloat32_t z1 = svread_hor_za32_m(
					z1, svptrue_b32(),
					/* tile: */ 1, /* slice: */ tcol + 0);
				svfloat32_t z2 = svread_hor_za32_m(
					z2, svptrue_b32(),
					/* tile: */ 2, /* slice: */ tcol + 0);
				svfloat32_t z3 = svread_hor_za32_m(
					z3, svptrue_b32(),
					/* tile: */ 3, /* slice: */ tcol + 0);

				//Scale Z regs by broadcast alpha
				z0 = svmul_f32_z(svptrue_b32(), z0, zalpha);
				z1 = svmul_f32_z(svptrue_b32(), z1, zalpha);
				z2 = svmul_f32_z(svptrue_b32(), z2, zalpha);
				z3 = svmul_f32_z(svptrue_b32(), z3, zalpha);

				//Store full result into C
				svfloat32x4_t z4w = svcreate4(z0, z1, z2, z3);
				svst1_f32_x4(svptrue_c32(), &c_[result_tile_TL_corner + (tcol + 0) * rs_c], z4w);
				
				z0 = svread_hor_za32_m(
					z0, svptrue_b32(),
					/* tile: */ 0, /* slice: */ tcol + 1);
				z1 = svread_hor_za32_m(
					z1, svptrue_b32(),
					/* tile: */ 1, /* slice: */ tcol + 1);
				z2 = svread_hor_za32_m(
					z2, svptrue_b32(),
					/* tile: */ 2, /* slice: */ tcol + 1);
				z3 = svread_hor_za32_m(
					z3, svptrue_b32(),
					/* tile: */ 3, /* slice: */ tcol + 1);


				z0 = svmul_f32_z(svptrue_b32(), z0, zalpha);
				z1 = svmul_f32_z(svptrue_b32(), z1, zalpha);
				z2 = svmul_f32_z(svptrue_b32(), z2, zalpha);
				z3 = svmul_f32_z(svptrue_b32(), z3, zalpha);
				
				svfloat32x4_t z5w = svcreate4(z0, z1, z2, z3);
				svst1_f32_x4(svptrue_c32(), &c_[result_tile_TL_corner + (tcol + 1) * rs_c], z5w);

				z0 = svread_hor_za32_m(
					z0, svptrue_b32(),
					/* tile: */ 0, /* slice: */ tcol + 2);
				z1 = svread_hor_za32_m(
					z1, svptrue_b32(),
					/* tile: */ 1, /* slice: */ tcol + 2);
				z2 = svread_hor_za32_m(
					z2, svptrue_b32(),
					/* tile: */ 2, /* slice: */ tcol + 2);
				z3 = svread_hor_za32_m(
					z3, svptrue_b32(),
					/* tile: */ 3, /* slice: */ tcol + 2);


				z0 = svmul_f32_z(svptrue_b32(), z0, zalpha);
				z1 = svmul_f32_z(svptrue_b32(), z1, zalpha);
				z2 = svmul_f32_z(svptrue_b32(), z2, zalpha);
				z3 = svmul_f32_z(svptrue_b32(), z3, zalpha);      
				
				svfloat32x4_t z6w = svcreate4(z0, z1, z2, z3);
				svst1_f32_x4(svptrue_c32(), &c_[result_tile_TL_corner + (tcol + 2) * rs_c], z6w);

				z0 = svread_hor_za32_m(
					z0, svptrue_b32(),
					/* tile: */ 0, /* slice: */ tcol + 3);
				z1 = svread_hor_za32_m(
					z1, svptrue_b32(),
					/* tile: */ 1, /* slice: */ tcol + 3);
				z2 = svread_hor_za32_m(
					z2, svptrue_b32(),
					/* tile: */ 2, /* slice: */ tcol + 3);
				z3 = svread_hor_za32_m(
					z3, svptrue_b32(),
					/* tile: */ 3, /* slice: */ tcol + 3);


				z0 = svmul_f32_z(svptrue_b32(), z0, zalpha);
				z1 = svmul_f32_z(svptrue_b32(), z1, zalpha);
				z2 = svmul_f32_z(svptrue_b32(), z2, zalpha);
				z3 = svmul_f32_z(svptrue_b32(), z3, zalpha);

				svfloat32x4_t z7w = svcreate4(z0, z1, z2, z3);
				svst1_f32_x4(svptrue_c32(), &c_[result_tile_TL_corner + (tcol + 3) * rs_c], z7w);
			}
		}
		else {
			for (uint64_t tcol = 0; tcol < SVL; tcol += 4) {
				//Read ZA slices into Z regs
				svfloat32_t z0 = svread_hor_za32_m(
					z0, svptrue_b32(),
					/* tile: */ 0, /* slice: */ tcol + 0);
				svfloat32_t z1 = svread_hor_za32_m(
					z1, svptrue_b32(),
					/* tile: */ 1, /* slice: */ tcol + 0);
				svfloat32_t z2 = svread_hor_za32_m(
					z2, svptrue_b32(),
					/* tile: */ 2, /* slice: */ tcol + 0);
				svfloat32_t z3 = svread_hor_za32_m(
					z3, svptrue_b32(),
					/* tile: */ 3, /* slice: */ tcol + 0);

				//Scale Z regs by broadcast alpha
				z0 = svmul_f32_z(svptrue_b32(), z0, zalpha);
				z1 = svmul_f32_z(svptrue_b32(), z1, zalpha);
				z2 = svmul_f32_z(svptrue_b32(), z2, zalpha);
				z3 = svmul_f32_z(svptrue_b32(), z3, zalpha);
					
				//Load C into Z regs
				svfloat32x4_t z4q = svld1_f32_x4(svptrue_c32(), &c_[result_tile_TL_corner + (tcol + 0) * rs_c]);

				//Scale Z regs by broadcast beta
				svfloat32_t z4 = svmla_m(svptrue_b32(), z0, svget4(z4q,0), zbeta);
				svfloat32_t z5 = svmla_m(svptrue_b32(), z1, svget4(z4q,1), zbeta);
				svfloat32_t z6 = svmla_m(svptrue_b32(), z2, svget4(z4q,2), zbeta);
				svfloat32_t z7 = svmla_m(svptrue_b32(), z3, svget4(z4q,3), zbeta);

				//Store full result into C
				svfloat32x4_t z4w = svcreate4(z4, z5, z6, z7);
				svst1_f32_x4(svptrue_c32(), &c_[result_tile_TL_corner + (tcol + 0) * rs_c], z4w);
				
				z0 = svread_hor_za32_m(
					z0, svptrue_b32(),
					/* tile: */ 0, /* slice: */ tcol + 1);
				z1 = svread_hor_za32_m(
					z1, svptrue_b32(),
					/* tile: */ 1, /* slice: */ tcol + 1);
				z2 = svread_hor_za32_m(
					z2, svptrue_b32(),
					/* tile: */ 2, /* slice: */ tcol + 1);
				z3 = svread_hor_za32_m(
					z3, svptrue_b32(),
					/* tile: */ 3, /* slice: */ tcol + 1);


				z0 = svmul_f32_z(svptrue_b32(), z0, zalpha);
				z1 = svmul_f32_z(svptrue_b32(), z1, zalpha);
				z2 = svmul_f32_z(svptrue_b32(), z2, zalpha);
				z3 = svmul_f32_z(svptrue_b32(), z3, zalpha);

				svfloat32x4_t z5q = svld1_f32_x4(svptrue_c32(), &c_[result_tile_TL_corner + (tcol + 1) * rs_c]);

				z4 = svmla_m(svptrue_b32(), z0, svget4(z5q,0), zbeta);
				z5 = svmla_m(svptrue_b32(), z1, svget4(z5q,1), zbeta);
				z6 = svmla_m(svptrue_b32(), z2, svget4(z5q,2), zbeta);
				z7 = svmla_m(svptrue_b32(), z3, svget4(z5q,3), zbeta);
				
				svfloat32x4_t z5w = svcreate4(z4, z5, z6, z7);
				svst1_f32_x4(svptrue_c32(), &c_[result_tile_TL_corner + (tcol + 1) * rs_c], z5w);

				z0 = svread_hor_za32_m(
					z0, svptrue_b32(),
					/* tile: */ 0, /* slice: */ tcol + 2);
				z1 = svread_hor_za32_m(
					z1, svptrue_b32(),
					/* tile: */ 1, /* slice: */ tcol + 2);
				z2 = svread_hor_za32_m(
					z2, svptrue_b32(),
					/* tile: */ 2, /* slice: */ tcol + 2);
				z3 = svread_hor_za32_m(
					z3, svptrue_b32(),
					/* tile: */ 3, /* slice: */ tcol + 2);


				z0 = svmul_f32_z(svptrue_b32(), z0, zalpha);
				z1 = svmul_f32_z(svptrue_b32(), z1, zalpha);
				z2 = svmul_f32_z(svptrue_b32(), z2, zalpha);
				z3 = svmul_f32_z(svptrue_b32(), z3, zalpha);

				svfloat32x4_t z6q = svld1_f32_x4(svptrue_c32(), &c_[result_tile_TL_corner + (tcol + 2) * rs_c]);

				z4 = svmla_m(svptrue_b32(), z0, svget4(z6q,0), zbeta);
				z5 = svmla_m(svptrue_b32(), z1, svget4(z6q,1), zbeta);
				z6 = svmla_m(svptrue_b32(), z2, svget4(z6q,2), zbeta);
				z7 = svmla_m(svptrue_b32(), z3, svget4(z6q,3), zbeta);         
				
				svfloat32x4_t z6w = svcreate4(z4, z5, z6, z7);
				svst1_f32_x4(svptrue_c32(), &c_[result_tile_TL_corner + (tcol + 2) * rs_c], z6w);

				z0 = svread_hor_za32_m(
					z0, svptrue_b32(),
					/* tile: */ 0, /* slice: */ tcol + 3);
				z1 = svread_hor_za32_m(
					z1, svptrue_b32(),
					/* tile: */ 1, /* slice: */ tcol + 3);
				z2 = svread_hor_za32_m(
					z2, svptrue_b32(),
					/* tile: */ 2, /* slice: */ tcol + 3);
				z3 = svread_hor_za32_m(
					z3, svptrue_b32(),
					/* tile: */ 3, /* slice: */ tcol + 3);


				z0 = svmul_f32_z(svptrue_b32(), z0, zalpha);
				z1 = svmul_f32_z(svptrue_b32(), z1, zalpha);
				z2 = svmul_f32_z(svptrue_b32(), z2, zalpha);
				z3 = svmul_f32_z(svptrue_b32(), z3, zalpha);

				svfloat32x4_t z7q = svld1_f32_x4(svptrue_c32(), &c_[result_tile_TL_corner + (tcol + 3) * rs_c]);

				z4 = svmla_m(svptrue_b32(), z0, svget4(z7q,0), zbeta);
				z5 = svmla_m(svptrue_b32(), z1, svget4(z7q,1), zbeta);
				z6 = svmla_m(svptrue_b32(), z2, svget4(z7q,2), zbeta);
				z7 = svmla_m(svptrue_b32(), z3, svget4(z7q,3), zbeta);

				svfloat32x4_t z7w = svcreate4(z4, z5, z6, z7);
				svst1_f32_x4(svptrue_c32(), &c_[result_tile_TL_corner + (tcol + 3) * rs_c], z7w);
			}
		}
	}

	GEMM_UKR_FLUSH_CT( s );

	return;

     //bli_sgemm_m4sme_asm_8x12_impl(m,n,k,alpha, a, b, beta, c, rs_c0, cs_c0, data, cntx);
//return;
}
/*
   o 4x4 Double precision micro-kernel NOT fully functional yet.
   o Runnable on ARMv8, compiled with aarch64 GCC.
   o Use it together with the armv8 BLIS configuration.
   o Tested on Juno board. Around 3 GFLOPS @ 1.1 GHz.

   December 2014.

 * UPDATE OCTOBER 2015: Now is fully functional.
 * Tested on Juno board. Around 5.6 GFLOPS, 2 A57 cores @ 1.1 GHz.
 * Tested on Juno board. Around 4 GFLOPS, 4 A53 cores @ 850 MHz.

 * UPDATE NOVEMBER 2015
 * Micro-kernel changed to 6x8
 * Tested on Juno Board. Around 4   GFLOPS, 1 x A57 core  @ 1.1 GHz.
 * Tested on Juno Board. Around 7.6 GFLOPS, 2 x A57 cores @ 1.1 GHz.
 * Tested on Juno board. Around 1.5 GFLOPS, 1 x A53 core  @ 850 MHz.
 * Tested on Juno board. Around 5.5 GFLOPS, 4 x A53 cores @ 850 MHz.

 * UPDATE JULY 2021 - Leick Robinson
 * Both Microkernels changed to fix two prefetching performance bugs
 * Tested on 2s Altra. Around 3,200 GFLOPS, 160 x N2 cores @ 3.0 GHz
 * Tested on 1s Altra, Around 1,700 GFLOPS,  80 x N2 cores @ 3.0 GHz
 * Tested on 1s Altra Max,  ~ 2,600 GFLOPS, 128 x N2 cores @ 3.0 GHz
*/
void bli_dgemm_m4sme_asm_6x8
     (
             dim_t      m,
             dim_t      n,
             dim_t      k,
       const void*      alpha,
       const void*      a,
       const void*      b,
       const void*      beta,
             void*      c, inc_t rs_c0, inc_t cs_c0,
       const auxinfo_t* data,
       const cntx_t*    cntx
     )
{
#ifdef DISPLAY_DEBUG_INFO

	static bool bFirstTime = true;

	if ( bFirstTime )
	{
		printf( "In bli_dgemm_m4sme_asm_6x8: rs_c0=%d, cs_c0=%d \n",
		        (int) rs_c0, (int) cs_c0 );
		fflush( stdout );
		bFirstTime = false;
	}

#endif

	const void* a_next = bli_auxinfo_next_a( data );
	const void* b_next = bli_auxinfo_next_b( data );

	// Typecast local copies of integers in case dim_t and inc_t are a
	// different size than is expected by load instructions.
	uint64_t k_iter = k / 4;
	uint64_t k_left = k % 4;
	uint64_t rs_c   = rs_c0;
	uint64_t cs_c   = cs_c0;

	GEMM_UKR_SETUP_CT_ANY( d, 6, 8, false );


	__asm__ volatile
	(
	" smstart                                    \n\t"
	" smstop                                     \n\t"
	" ldr x0,%[aaddr]                            \n\t" // Load address of A
	" ldr x1,%[baddr]                            \n\t" // Load address of B
	" ldr x2,%[caddr]                            \n\t" // Load address of C
	"                                            \n\t"
	" ldr x10,%[cs_c]                            \n\t" // Load cs_c
	" lsl x10,x10,#3                             \n\t" // cs_c * sizeof(double)
	"                                            \n\t"
	" ldr x5,%[k_iter]                           \n\t" // Init guard (k_iter)
	" ldr x6,%[k_left]                           \n\t" // Init guard (k_iter)
	" add x20,x2,x10                             \n\t" //Load address Column 1 of C
	"                                            \n\t"
	" ldr x14,%[rs_c]                            \n\t" // Load rs_c.
	" lsl x14,x14,#3                             \n\t" // rs_c * sizeof(double).
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	" dup  v8.2d, xzr                            \n\t" // Vector for accummulating column 0
	" dup  v9.2d, xzr                            \n\t" // Vector for accummulating column 0
	" dup  v10.2d, xzr                           \n\t" // Vector for accummulating column 0
	" dup  v11.2d, xzr                           \n\t" // Vector for accummulating column 1
	" dup  v12.2d, xzr                           \n\t" // Vector for accummulating column 1
	" add x21,x20,x10                            \n\t" //Load address Column 2 of C
	" dup  v13.2d, xzr                           \n\t" // Vector for accummulating column 1
	"                                            \n\t"
	" dup  v14.2d, xzr                           \n\t" // Vector for accummulating column 2
	" add x22,x21,x10                            \n\t" //Load address Column 3 of C
	" dup  v15.2d, xzr                           \n\t" // Vector for accummulating column 2
	" dup  v16.2d, xzr                           \n\t" // Vector for accummulating column 2
	"                                            \n\t" // Since the columns can cross a cache line boundary,
	                                                   // we also need to prefetch the "ends"
	" add x23,x22,x10                            \n\t" //Load address Column 4 of C
	" dup  v17.2d, xzr                           \n\t" // Vector for accummulating column 3
	" dup  v18.2d, xzr                           \n\t" // Vector for accummulating column 3
	" add x24,x23,x10                            \n\t" //Load address Column 5 of C
	" dup  v19.2d, xzr                           \n\t" // Vector for accummulating column 3
	"                                            \n\t"
	" dup  v20.2d, xzr                           \n\t" // Vector for accummulating column 4
	" add x25,x24,x10                            \n\t" //Load address Column 6 of C
	" dup  v21.2d, xzr                           \n\t" // Vector for accummulating column 4
	" dup  v22.2d, xzr                           \n\t" // Vector for accummulating column 4
	" add x26,x25,x10                            \n\t" //Load address Column 7 of C
	" dup  v23.2d, xzr                           \n\t" // Vector for accummulating column 5
	" dup  v24.2d, xzr                           \n\t" // Vector for accummulating column 5
	" dup  v25.2d, xzr                           \n\t" // Vector for accummulating column 5
	"                                            \n\t"
	" dup  v26.2d, xzr                           \n\t" // Vector for accummulating column 6
	" dup  v27.2d, xzr                           \n\t" // Vector for accummulating column 6
	" dup  v28.2d, xzr                           \n\t" // Vector for accummulating column 6
	" dup  v29.2d, xzr                           \n\t" // Vector for accummulating column 7
	" dup  v30.2d, xzr                           \n\t" // Vector for accummulating column 7
	" dup  v31.2d, xzr                           \n\t" // Vector for accummulating column 7
	"                                            \n\t"
	"                                            \n\t"
	" cmp x5,#0                                  \n\t" // If k_iter == 0, jump to k_left.
	BEQ (DCONSIDERKLEFT)
	"                                            \n\t"
	" ldr q0, [x0]                               \n\t" // Load a
	" ldr q1, [x0, #16]                          \n\t"
	" ldr q2, [x0, #32]                          \n\t"
	"                                            \n\t"
	" ldr q3, [x1]                               \n\t" // Load b
	" ldr q4, [x1, #16]                          \n\t"
	" ldr q5, [x1, #32]                          \n\t"
	" ldr q6, [x1, #48]                          \n\t"
	"                                            \n\t"
	" add x0, x0, #48                            \n\t" //update address of A
	" add x1, x1, #64                            \n\t" //update address of B
	"                                            \n\t"
	" cmp x5,1                                   \n\t" // If there is just one k_iter, jump to that one.
	BEQ (DLASTITER)                                    // (as loop is do-while-like).
	"                                            \n\t"
	LABEL (DLOOP)                                      // Body
	"                                            \n\t"
	" fmla v8.2d ,v0.2d,v3.d[0]                  \n\t" // Accummulate
	" fmla v9.2d ,v1.2d,v3.d[0]                  \n\t" // Accummulate
	" fmla v10.2d,v2.2d,v3.d[0]                  \n\t" // Accummulate
	"                                            \n\t"
	" fmla v11.2d,v0.2d,v3.d[1]                  \n\t" // Accummulate
	" fmla v12.2d,v1.2d,v3.d[1]                  \n\t" // Accummulate
	" fmla v13.2d,v2.2d,v3.d[1]                  \n\t" // Accummulate
	"                                            \n\t"
	" fmla v14.2d,v0.2d,v4.d[0]                  \n\t" // Accummulate
	" fmla v15.2d,v1.2d,v4.d[0]                  \n\t" // Accummulate
	" fmla v16.2d,v2.2d,v4.d[0]                  \n\t" // Accummulate
	" ldr q3, [x1]                               \n\t"
	"                                            \n\t"
	" fmla v17.2d,v0.2d,v4.d[1]                  \n\t" // Accummulate
	" fmla v18.2d,v1.2d,v4.d[1]                  \n\t" // Accummulate
	" fmla v19.2d,v2.2d,v4.d[1]                  \n\t" // Accummulate
	" ldr q7, [x0, #32]                          \n\t"
	"                                            \n\t"
	" fmla v20.2d,v0.2d,v5.d[0]                  \n\t" // Accummulate
	" fmla v21.2d,v1.2d,v5.d[0]                  \n\t" // Accummulate
	" fmla v22.2d,v2.2d,v5.d[0]                  \n\t" // Accummulate
	" ldr q4, [x1, #16]                          \n\t"
	"                                            \n\t"
	" fmla v23.2d,v0.2d,v5.d[1]                  \n\t" // Accummulate
	" fmla v24.2d,v1.2d,v5.d[1]                  \n\t" // Accummulate
	" fmla v25.2d,v2.2d,v5.d[1]                  \n\t" // Accummulate
	" ldr q5, [x1, #32]                          \n\t"
	"                                            \n\t"
	" fmla v26.2d,v0.2d,v6.d[0]                  \n\t" // Accummulate
	" fmla v29.2d,v0.2d,v6.d[1]                  \n\t" // Accummulate
	" ldr q0, [x0]                               \n\t"
	"                                            \n\t"
	" fmla v27.2d,v1.2d,v6.d[0]                  \n\t" // Accummulate
	" fmla v30.2d,v1.2d,v6.d[1]                  \n\t" // Accummulate
	" ldr q1, [x0, #16]                          \n\t"
	"                                            \n\t"
	" fmla v28.2d,v2.2d,v6.d[0]                  \n\t" // Accummulate
	" fmla v31.2d,v2.2d,v6.d[1]                  \n\t" // Accummulate
	" ldr q6, [x1, #48]                          \n\t"
	"                                            \n\t" // End it 1
	" fmla v8.2d ,v0.2d,v3.d[0]                  \n\t" // Accummulate
	" fmla v9.2d ,v1.2d,v3.d[0]                  \n\t" // Accummulate
	" fmla v10.2d,v7.2d,v3.d[0]                  \n\t" // Accummulate
	"                                            \n\t"
	" fmla v11.2d,v0.2d,v3.d[1]                  \n\t" // Accummulate
	" fmla v12.2d,v1.2d,v3.d[1]                  \n\t" // Accummulate
	" fmla v13.2d,v7.2d,v3.d[1]                  \n\t" // Accummulate
	"                                            \n\t"
	" fmla v14.2d,v0.2d,v4.d[0]                  \n\t" // Accummulate
	" fmla v15.2d,v1.2d,v4.d[0]                  \n\t" // Accummulate
	" fmla v16.2d,v7.2d,v4.d[0]                  \n\t" // Accummulate
	" ldr q3, [x1, #64]                          \n\t"
	"                                            \n\t"
	" fmla v17.2d,v0.2d,v4.d[1]                  \n\t" // Accummulate
	" fmla v18.2d,v1.2d,v4.d[1]                  \n\t" // Accummulate
	" fmla v19.2d,v7.2d,v4.d[1]                  \n\t" // Accummulate
	" ldr q2, [x0, #80]                          \n\t"
	"                                            \n\t"
	" fmla v20.2d,v0.2d,v5.d[0]                  \n\t" // Accummulate
	" fmla v21.2d,v1.2d,v5.d[0]                  \n\t" // Accummulate
	" fmla v22.2d,v7.2d,v5.d[0]                  \n\t" // Accummulate
	" ldr q4, [x1, #80]                          \n\t"
	"                                            \n\t"
	" fmla v23.2d,v0.2d,v5.d[1]                  \n\t" // Accummulate
	" fmla v24.2d,v1.2d,v5.d[1]                  \n\t" // Accummulate
	" fmla v25.2d,v7.2d,v5.d[1]                  \n\t" // Accummulate
	" ldr q5, [x1, #96]                          \n\t"
	"                                            \n\t"
	" fmla v26.2d,v0.2d,v6.d[0]                  \n\t" // Accummulate
	" fmla v29.2d,v0.2d,v6.d[1]                  \n\t" // Accummulate
	" ldr q0, [x0, #48]                          \n\t"
	"                                            \n\t"
	" fmla v27.2d,v1.2d,v6.d[0]                  \n\t" // Accummulate
	" fmla v30.2d,v1.2d,v6.d[1]                  \n\t" // Accummulate
	" ldr q1, [x0, #64]                          \n\t"
	"                                            \n\t"
	" fmla v28.2d,v7.2d,v6.d[0]                  \n\t" // Accummulate
	" fmla v31.2d,v7.2d,v6.d[1]                  \n\t" // Accummulate
	" ldr q6, [x1, #112]                         \n\t"
	"                                            \n\t" //End it 2
	" fmla v8.2d ,v0.2d,v3.d[0]                  \n\t" // Accummulate
	" fmla v9.2d ,v1.2d,v3.d[0]                  \n\t" // Accummulate
	" fmla v10.2d,v2.2d,v3.d[0]                  \n\t" // Accummulate
	"                                            \n\t"
	" fmla v11.2d,v0.2d,v3.d[1]                  \n\t" // Accummulate
	" fmla v12.2d,v1.2d,v3.d[1]                  \n\t" // Accummulate
	" fmla v13.2d,v2.2d,v3.d[1]                  \n\t" // Accummulate
	"                                            \n\t"
	" fmla v14.2d,v0.2d,v4.d[0]                  \n\t" // Accummulate
	" fmla v15.2d,v1.2d,v4.d[0]                  \n\t" // Accummulate
	" fmla v16.2d,v2.2d,v4.d[0]                  \n\t" // Accummulate
	" ldr q3, [x1, #128]                         \n\t"
	"                                            \n\t"
	" fmla v17.2d,v0.2d,v4.d[1]                  \n\t" // Accummulate
	" fmla v18.2d,v1.2d,v4.d[1]                  \n\t" // Accummulate
	" fmla v19.2d,v2.2d,v4.d[1]                  \n\t" // Accummulate
	" ldr q7, [x0, #128]                         \n\t"
	"                                            \n\t"
	" fmla v20.2d,v0.2d,v5.d[0]                  \n\t" // Accummulate
	" fmla v21.2d,v1.2d,v5.d[0]                  \n\t" // Accummulate
	" fmla v22.2d,v2.2d,v5.d[0]                  \n\t" // Accummulate
	" ldr q4, [x1, #144]                         \n\t"
	"                                            \n\t"
	" fmla v23.2d,v0.2d,v5.d[1]                  \n\t" // Accummulate
	" fmla v24.2d,v1.2d,v5.d[1]                  \n\t" // Accummulate
	" fmla v25.2d,v2.2d,v5.d[1]                  \n\t" // Accummulate
	" ldr q5, [x1, #160]                         \n\t"
	"                                            \n\t"
	" fmla v26.2d,v0.2d,v6.d[0]                  \n\t" // Accummulate
	" fmla v29.2d,v0.2d,v6.d[1]                  \n\t" // Accummulate
	" ldr q0, [x0, #96]                          \n\t"
	"                                            \n\t"
	" fmla v27.2d,v1.2d,v6.d[0]                  \n\t" // Accummulate
	" fmla v30.2d,v1.2d,v6.d[1]                  \n\t" // Accummulate
	" ldr q1, [x0, #112]                         \n\t"
	"                                            \n\t"
	" fmla v28.2d,v2.2d,v6.d[0]                  \n\t" // Accummulate
	" fmla v31.2d,v2.2d,v6.d[1]                  \n\t" // Accummulate
	" ldr q6, [x1, #176]                         \n\t"
	"                                            \n\t" // End it 3
	" fmla v8.2d ,v0.2d,v3.d[0]                  \n\t" // Accummulate
	" fmla v9.2d ,v1.2d,v3.d[0]                  \n\t" // Accummulate
	" fmla v10.2d,v7.2d,v3.d[0]                  \n\t" // Accummulate
	"                                            \n\t"
	" fmla v11.2d,v0.2d,v3.d[1]                  \n\t" // Accummulate
	" fmla v12.2d,v1.2d,v3.d[1]                  \n\t" // Accummulate
	" fmla v13.2d,v7.2d,v3.d[1]                  \n\t" // Accummulate
	" ldr q3, [x1, #192]                         \n\t"
	"                                            \n\t"
	" fmla v14.2d,v0.2d,v4.d[0]                  \n\t" // Accummulate
	" fmla v15.2d,v1.2d,v4.d[0]                  \n\t" // Accummulate
	" fmla v16.2d,v7.2d,v4.d[0]                  \n\t" // Accummulate
	" ldr q2, [x0, #176]                         \n\t"
	"                                            \n\t"
	" fmla v17.2d,v0.2d,v4.d[1]                  \n\t" // Accummulate
	" fmla v18.2d,v1.2d,v4.d[1]                  \n\t" // Accummulate
	" fmla v19.2d,v7.2d,v4.d[1]                  \n\t" // Accummulate
	" ldr q4, [x1, #208]                         \n\t"
	"                                            \n\t"
	" fmla v20.2d,v0.2d,v5.d[0]                  \n\t" // Accummulate
	" fmla v21.2d,v1.2d,v5.d[0]                  \n\t" // Accummulate
	" fmla v22.2d,v7.2d,v5.d[0]                  \n\t" // Accummulate
	"                                            \n\t"
	" fmla v23.2d,v0.2d,v5.d[1]                  \n\t" // Accummulate
	" fmla v24.2d,v1.2d,v5.d[1]                  \n\t" // Accummulate
	" fmla v25.2d,v7.2d,v5.d[1]                  \n\t" // Accummulate
	" ldr q5, [x1, #224]                         \n\t"
	"                                            \n\t"
	" fmla v26.2d,v0.2d,v6.d[0]                  \n\t" // Accummulate
	" fmla v29.2d,v0.2d,v6.d[1]                  \n\t" // Accummulate
	" ldr q0, [x0, #144]                         \n\t"
	"                                            \n\t"
	" fmla v27.2d,v1.2d,v6.d[0]                  \n\t" // Accummulate
	" fmla v30.2d,v1.2d,v6.d[1]                  \n\t" // Accummulate
	" ldr q1, [x0, #160]                         \n\t"
	"                                            \n\t"
	" fmla v28.2d,v7.2d,v6.d[0]                  \n\t" // Accummulate
	" fmla v31.2d,v7.2d,v6.d[1]                  \n\t" // Accummulate
	" ldr q6, [x1, #240]                         \n\t"
	"                                            \n\t" //End it 4
	" add x0, x0, #192                           \n\t"
	" add x1, x1, #256                           \n\t"
	"                                            \n\t"
	" sub x5,x5,1                                \n\t" // i-=1
	" cmp x5,1                                   \n\t" // Iterate again if we are not in k_iter == 1.
	BNE (DLOOP)
	"                                            \n\t"
	LABEL (DLASTITER)
	"                                            \n\t"
	" fmla v8.2d ,v0.2d,v3.d[0]                  \n\t" // Accummulate
	" fmla v9.2d ,v1.2d,v3.d[0]                  \n\t" // Accummulate
	" fmla v10.2d,v2.2d,v3.d[0]                  \n\t" // Accummulate
	"                                            \n\t"
	" fmla v11.2d,v0.2d,v3.d[1]                  \n\t" // Accummulate
	" fmla v12.2d,v1.2d,v3.d[1]                  \n\t" // Accummulate
	" fmla v13.2d,v2.2d,v3.d[1]                  \n\t" // Accummulate
	" ldr q3, [x1]                               \n\t"
	"                                            \n\t"
	" fmla v14.2d,v0.2d,v4.d[0]                  \n\t" // Accummulate
	" fmla v15.2d,v1.2d,v4.d[0]                  \n\t" // Accummulate
	" fmla v16.2d,v2.2d,v4.d[0]                  \n\t" // Accummulate
	" ldr q7, [x0, #32]                          \n\t"
	"                                            \n\t"
	" fmla v17.2d,v0.2d,v4.d[1]                  \n\t" // Accummulate
	" fmla v18.2d,v1.2d,v4.d[1]                  \n\t" // Accummulate
	" fmla v19.2d,v2.2d,v4.d[1]                  \n\t" // Accummulate
	" ldr q4, [x1, #16]                          \n\t"
	"                                            \n\t"
	" fmla v20.2d,v0.2d,v5.d[0]                  \n\t" // Accummulate
	" fmla v21.2d,v1.2d,v5.d[0]                  \n\t" // Accummulate
	" fmla v22.2d,v2.2d,v5.d[0]                  \n\t" // Accummulate
	"                                            \n\t"
	" fmla v23.2d,v0.2d,v5.d[1]                  \n\t" // Accummulate
	" fmla v24.2d,v1.2d,v5.d[1]                  \n\t" // Accummulate
	" fmla v25.2d,v2.2d,v5.d[1]                  \n\t" // Accummulate
	" ldr q5, [x1, #32]                          \n\t"
	"                                            \n\t"
	" fmla v26.2d,v0.2d,v6.d[0]                  \n\t" // Accummulate
	" fmla v29.2d,v0.2d,v6.d[1]                  \n\t" // Accummulate
	" ldr q0, [x0]                               \n\t"
	"                                            \n\t"
	" fmla v27.2d,v1.2d,v6.d[0]                  \n\t" // Accummulate
	" fmla v30.2d,v1.2d,v6.d[1]                  \n\t" // Accummulate
	" ldr q1, [x0, #16]                          \n\t"
	"                                            \n\t"
	" fmla v28.2d,v2.2d,v6.d[0]                  \n\t" // Accummulate
	" fmla v31.2d,v2.2d,v6.d[1]                  \n\t" // Accummulate
	" ldr q6, [x1, #48]                          \n\t"
	"                                            \n\t" // End it 1
	" fmla v8.2d ,v0.2d,v3.d[0]                  \n\t" // Accummulate
	" fmla v9.2d ,v1.2d,v3.d[0]                  \n\t" // Accummulate
	" fmla v10.2d,v7.2d,v3.d[0]                  \n\t" // Accummulate
	"                                            \n\t"
	" fmla v11.2d,v0.2d,v3.d[1]                  \n\t" // Accummulate
	" fmla v12.2d,v1.2d,v3.d[1]                  \n\t" // Accummulate
	" fmla v13.2d,v7.2d,v3.d[1]                  \n\t" // Accummulate
	" ldr q3, [x1, #64]                          \n\t"
	"                                            \n\t"
	" fmla v14.2d,v0.2d,v4.d[0]                  \n\t" // Accummulate
	" fmla v15.2d,v1.2d,v4.d[0]                  \n\t" // Accummulate
	" fmla v16.2d,v7.2d,v4.d[0]                  \n\t" // Accummulate
	" ldr q2, [x0, #80]                          \n\t"
	"                                            \n\t"
	" fmla v17.2d,v0.2d,v4.d[1]                  \n\t" // Accummulate
	" fmla v18.2d,v1.2d,v4.d[1]                  \n\t" // Accummulate
	" fmla v19.2d,v7.2d,v4.d[1]                  \n\t" // Accummulate
	" ldr q4, [x1, #80]                          \n\t"
	"                                            \n\t"
	" fmla v20.2d,v0.2d,v5.d[0]                  \n\t" // Accummulate
	" fmla v21.2d,v1.2d,v5.d[0]                  \n\t" // Accummulate
	" fmla v22.2d,v7.2d,v5.d[0]                  \n\t" // Accummulate
	"                                            \n\t"
	" fmla v23.2d,v0.2d,v5.d[1]                  \n\t" // Accummulate
	" fmla v24.2d,v1.2d,v5.d[1]                  \n\t" // Accummulate
	" fmla v25.2d,v7.2d,v5.d[1]                  \n\t" // Accummulate
	" ldr q5, [x1, #96]                          \n\t"
	"                                            \n\t"
	" fmla v26.2d,v0.2d,v6.d[0]                  \n\t" // Accummulate
	" fmla v29.2d,v0.2d,v6.d[1]                  \n\t" // Accummulate
	" ldr q0, [x0, #48]                          \n\t"
	"                                            \n\t"
	" fmla v27.2d,v1.2d,v6.d[0]                  \n\t" // Accummulate
	" fmla v30.2d,v1.2d,v6.d[1]                  \n\t" // Accummulate
	" ldr q1, [x0, #64]                          \n\t"
	"                                            \n\t"
	" fmla v28.2d,v7.2d,v6.d[0]                  \n\t" // Accummulate
	" fmla v31.2d,v7.2d,v6.d[1]                  \n\t" // Accummulate
	" ldr q6, [x1, #112]                         \n\t"
	"                                            \n\t" //End it 2
	" fmla v8.2d ,v0.2d,v3.d[0]                  \n\t" // Accummulate
	" fmla v9.2d ,v1.2d,v3.d[0]                  \n\t" // Accummulate
	" fmla v10.2d,v2.2d,v3.d[0]                  \n\t" // Accummulate
	"                                            \n\t"
	" fmla v11.2d,v0.2d,v3.d[1]                  \n\t" // Accummulate
	" fmla v12.2d,v1.2d,v3.d[1]                  \n\t" // Accummulate
	" fmla v13.2d,v2.2d,v3.d[1]                  \n\t" // Accummulate
	" ldr q3, [x1, #128]                         \n\t"
	"                                            \n\t"
	" fmla v14.2d,v0.2d,v4.d[0]                  \n\t" // Accummulate
	" fmla v15.2d,v1.2d,v4.d[0]                  \n\t" // Accummulate
	" fmla v16.2d,v2.2d,v4.d[0]                  \n\t" // Accummulate
	" ldr q7, [x0, #128]                         \n\t"
	"                                            \n\t"
	" fmla v17.2d,v0.2d,v4.d[1]                  \n\t" // Accummulate
	" fmla v18.2d,v1.2d,v4.d[1]                  \n\t" // Accummulate
	" fmla v19.2d,v2.2d,v4.d[1]                  \n\t" // Accummulate
	" ldr q4, [x1, #144]                         \n\t"
	"                                            \n\t"
	" fmla v20.2d,v0.2d,v5.d[0]                  \n\t" // Accummulate
	" fmla v21.2d,v1.2d,v5.d[0]                  \n\t" // Accummulate
	" fmla v22.2d,v2.2d,v5.d[0]                  \n\t" // Accummulate
	"                                            \n\t"
	" fmla v23.2d,v0.2d,v5.d[1]                  \n\t" // Accummulate
	" fmla v24.2d,v1.2d,v5.d[1]                  \n\t" // Accummulate
	" fmla v25.2d,v2.2d,v5.d[1]                  \n\t" // Accummulate
	" ldr q5, [x1, #160]                         \n\t"
	"                                            \n\t"
	" fmla v26.2d,v0.2d,v6.d[0]                  \n\t" // Accummulate
	" fmla v29.2d,v0.2d,v6.d[1]                  \n\t" // Accummulate
	" ldr q0, [x0, #96]                          \n\t"
	"                                            \n\t"
	" fmla v27.2d,v1.2d,v6.d[0]                  \n\t" // Accummulate
	" fmla v30.2d,v1.2d,v6.d[1]                  \n\t" // Accummulate
	" ldr q1, [x0, #112]                         \n\t"
	"                                            \n\t"
	" fmla v28.2d,v2.2d,v6.d[0]                  \n\t" // Accummulate
	" fmla v31.2d,v2.2d,v6.d[1]                  \n\t" // Accummulate
	" ldr q6, [x1, #176]                         \n\t"
	"                                            \n\t" // End it 3
	" fmla v8.2d ,v0.2d,v3.d[0]                  \n\t" // Accummulate
	" fmla v9.2d ,v1.2d,v3.d[0]                  \n\t" // Accummulate
	" fmla v10.2d,v7.2d,v3.d[0]                  \n\t" // Accummulate
	"                                            \n\t"
	" fmla v11.2d,v0.2d,v3.d[1]                  \n\t" // Accummulate
	" fmla v12.2d,v1.2d,v3.d[1]                  \n\t" // Accummulate
	" fmla v13.2d,v7.2d,v3.d[1]                  \n\t" // Accummulate
	"                                            \n\t"
	" fmla v14.2d,v0.2d,v4.d[0]                  \n\t" // Accummulate
	" fmla v15.2d,v1.2d,v4.d[0]                  \n\t" // Accummulate
	" fmla v16.2d,v7.2d,v4.d[0]                  \n\t" // Accummulate
	"                                            \n\t"
	" fmla v17.2d,v0.2d,v4.d[1]                  \n\t" // Accummulate
	" fmla v18.2d,v1.2d,v4.d[1]                  \n\t" // Accummulate
	" fmla v19.2d,v7.2d,v4.d[1]                  \n\t" // Accummulate
	"                                            \n\t"
	" fmla v20.2d,v0.2d,v5.d[0]                  \n\t" // Accummulate
	" fmla v21.2d,v1.2d,v5.d[0]                  \n\t" // Accummulate
	" fmla v22.2d,v7.2d,v5.d[0]                  \n\t" // Accummulate
	"                                            \n\t"
	" fmla v23.2d,v0.2d,v5.d[1]                  \n\t" // Accummulate
	" fmla v24.2d,v1.2d,v5.d[1]                  \n\t" // Accummulate
	" fmla v25.2d,v7.2d,v5.d[1]                  \n\t" // Accummulate
	"                                            \n\t"
	" fmla v26.2d,v0.2d,v6.d[0]                  \n\t" // Accummulate
	" add x1, x1, #192                           \n\t"
	" fmla v29.2d,v0.2d,v6.d[1]                  \n\t" // Accummulate
	"                                            \n\t"
	" fmla v27.2d,v1.2d,v6.d[0]                  \n\t" // Accummulate
	" fmla v30.2d,v1.2d,v6.d[1]                  \n\t" // Accummulate
	"                                            \n\t"
	" fmla v28.2d,v7.2d,v6.d[0]                  \n\t" // Accummulate
	" fmla v31.2d,v7.2d,v6.d[1]                  \n\t" // Accummulate
	"                                            \n\t" //End it 4
	" add x0, x0, #144                           \n\t"
	"                                            \n\t"
	LABEL (DCONSIDERKLEFT)
	" cmp x6,0                                   \n\t" // If k_left == 0, we are done.
	BEQ (DPOSTACCUM)                                   // else, we enter the k_left loop.
	"                                            \n\t"
	LABEL (DLOOPKLEFT)
	"                                            \n\t"
	" ldr q0, [x0],#16                           \n\t"
	" ldr q1, [x0],#16                           \n\t" // Load a
	" ldr q2, [x0],#16                           \n\t"
	"                                            \n\t"
	" ldr q3, [x1],#16                           \n\t" // Load b
	" ldr q4, [x1],#16                           \n\t"
	" ldr q5, [x1],#16                           \n\t"
	" ldr q6, [x1],#16                           \n\t"
	"                                            \n\t"
	" sub x6,x6,1                                \n\t"
	"                                            \n\t"
	" fmla v8.2d ,v0.2d,v3.d[0]                  \n\t" // Accummulate
	" fmla v9.2d ,v1.2d,v3.d[0]                  \n\t" // Accummulate
	" fmla v10.2d,v2.2d,v3.d[0]                  \n\t" // Accummulate
	"                                            \n\t"
	" fmla v11.2d,v0.2d,v3.d[1]                  \n\t" // Accummulate
	" fmla v12.2d,v1.2d,v3.d[1]                  \n\t" // Accummulate
	" fmla v13.2d,v2.2d,v3.d[1]                  \n\t" // Accummulate
	"                                            \n\t"
	" fmla v14.2d,v0.2d,v4.d[0]                  \n\t" // Accummulate
	" fmla v15.2d,v1.2d,v4.d[0]                  \n\t" // Accummulate
	" fmla v16.2d,v2.2d,v4.d[0]                  \n\t" // Accummulate
	"                                            \n\t"
	" fmla v17.2d,v0.2d,v4.d[1]                  \n\t" // Accummulate
	" fmla v18.2d,v1.2d,v4.d[1]                  \n\t" // Accummulate
	" fmla v19.2d,v2.2d,v4.d[1]                  \n\t" // Accummulate
	"                                            \n\t"
	" fmla v20.2d,v0.2d,v5.d[0]                  \n\t" // Accummulate
	" fmla v21.2d,v1.2d,v5.d[0]                  \n\t" // Accummulate
	" fmla v22.2d,v2.2d,v5.d[0]                  \n\t" // Accummulate
	"                                            \n\t"
	" fmla v23.2d,v0.2d,v5.d[1]                  \n\t" // Accummulate
	" fmla v24.2d,v1.2d,v5.d[1]                  \n\t" // Accummulate
	" fmla v25.2d,v2.2d,v5.d[1]                  \n\t" // Accummulate
	"                                            \n\t"
	" fmla v26.2d,v0.2d,v6.d[0]                  \n\t" // Accummulate
	" fmla v29.2d,v0.2d,v6.d[1]                  \n\t" // Accummulate
	"                                            \n\t"
	" fmla v27.2d,v1.2d,v6.d[0]                  \n\t" // Accummulate
	" fmla v30.2d,v1.2d,v6.d[1]                  \n\t" // Accummulate
	"                                            \n\t"
	" fmla v28.2d,v2.2d,v6.d[0]                  \n\t" // Accummulate
	" fmla v31.2d,v2.2d,v6.d[1]                  \n\t" // Accummulate
	"                                            \n\t"
	" cmp x6,0                                   \n\t" // Iterate again.
	BNE (DLOOPKLEFT)                                   // if i!=0.
	"                                            \n\t"
	LABEL (DPOSTACCUM)
	"                                            \n\t"
	" ldr x0,%[alpha]                            \n\t" // Alpha address
	" ldr x1,%[beta]                             \n\t" // Beta address
	"                                            \n\t"
	" ld1r {v6.2d},[x0]                          \n\t" // Load alpha.
	" ld1r {v7.2d},[x1]                          \n\t" // Load beta
	"                                            \n\t"
	" ldr x0,%[a_next]                           \n\t" // Next A address for later use.
	" ldr x1,%[b_next]                           \n\t" // Next B address for later use.
	"                                            \n\t"
	" cmp x14,#8                                 \n\t" // If rs_c != 1 (column-major)
	BNE (DGENSTORED)
	"                                            \n\t"
	LABEL (DCOLSTORED)                                 // C is column-major.
	"                                            \n\t"
	" dup  v0.2d, xzr                            \n\t"
	" dup  v1.2d, xzr                            \n\t"
	" dup  v2.2d, xzr                            \n\t"
	" dup  v3.2d, xzr                            \n\t"
	" dup  v4.2d, xzr                            \n\t"
	" dup  v5.2d, xzr                            \n\t"
	"                                            \n\t"
	" fcmp d7,#0.0                               \n\t"
	BEQ (DBETAZEROCOLSTOREDS1)                         // Taking care of the beta==0 case.
	"                                            \n\t"
	" ldr q0, [x2]                               \n\t" //Load column 0 of C
	" ldr q1, [x2, #16]                          \n\t"
	" ldr q2, [x2, #32]                          \n\t"
	"                                            \n\t"
	" ldr q3, [x20]                              \n\t" //Load column 1 of C
	" ldr q4, [x20, #16]                         \n\t"
	" ldr q5, [x20, #32]                         \n\t"
	"                                            \n\t"
	" fmul v0.2d,v0.2d,v7.d[0]                   \n\t" // Scale by beta
	" fmul v1.2d,v1.2d,v7.d[0]                   \n\t" // Scale by beta
	" fmul v2.2d,v2.2d,v7.d[0]                   \n\t" // Scale by beta
	" fmul v3.2d,v3.2d,v7.d[0]                   \n\t" // Scale by beta
	" fmul v4.2d,v4.2d,v7.d[0]                   \n\t" // Scale by beta
	" fmul v5.2d,v5.2d,v7.d[0]                   \n\t" // Scale by beta
	"                                            \n\t"
	LABEL (DBETAZEROCOLSTOREDS1)
	"                                            \n\t"
	" fmla v0.2d,v8.2d,v6.d[0]                   \n\t" // Scale by alpha
	" fmla v1.2d,v9.2d,v6.d[0]                   \n\t" // Scale by alpha
	" fmla v2.2d,v10.2d,v6.d[0]                  \n\t" // Scale by alpha
	" fmla v3.2d,v11.2d,v6.d[0]                  \n\t" // Scale by alpha
	" fmla v4.2d,v12.2d,v6.d[0]                  \n\t" // Scale by alpha
	" fmla v5.2d,v13.2d,v6.d[0]                  \n\t" // Scale by alpha
	"                                            \n\t"
	" str q0, [x2]                               \n\t" //Store column 0 of C
	" str q1, [x2, #16]                          \n\t"
	" str q2, [x2, #32]                          \n\t"
	"                                            \n\t"
	" str q3, [x20]                              \n\t" //Store column 1 of C
	" str q4, [x20, #16]                         \n\t"
	" str q5, [x20, #32]                         \n\t"
	"                                            \n\t"
	" dup  v8.2d, xzr                            \n\t"
	" dup  v9.2d, xzr                            \n\t"
	" dup  v10.2d, xzr                           \n\t"
	" dup  v11.2d, xzr                           \n\t"
	" dup  v12.2d, xzr                           \n\t"
	" dup  v13.2d, xzr                           \n\t"
	"                                            \n\t"
	" fcmp d7,#0.0                               \n\t"
	BEQ (DBETAZEROCOLSTOREDS2)                         // Taking care of the beta==0 case.
	"                                            \n\t"
	" ldr q8, [x21]                              \n\t" //Load column 2 of C
	" ldr q9, [x21, #16]                         \n\t"
	" ldr q10, [x21, #32]                        \n\t"
	"                                            \n\t"
	" ldr q11, [x22]                             \n\t" //Load column 3 of C
	" ldr q12, [x22, #16]                        \n\t"
	" ldr q13, [x22, #32]                        \n\t"
	"                                            \n\t"
	" fmul v8.2d, v8.2d, v7.d[0]                 \n\t" // Scale by beta
	" fmul v9.2d, v9.2d, v7.d[0]                 \n\t" // Scale by beta
	" fmul v10.2d,v10.2d,v7.d[0]                 \n\t" // Scale by beta
	" fmul v11.2d,v11.2d,v7.d[0]                 \n\t" // Scale by beta
	" fmul v12.2d,v12.2d,v7.d[0]                 \n\t" // Scale by beta
	" fmul v13.2d,v13.2d,v7.d[0]                 \n\t" // Scale by beta
	"                                            \n\t"
	LABEL (DBETAZEROCOLSTOREDS2)
	"                                            \n\t"
	" fmla v8.2d, v14.2d,v6.d[0]                 \n\t" // Scale by alpha
	" fmla v9.2d, v15.2d,v6.d[0]                 \n\t" // Scale by alpha
	" fmla v10.2d,v16.2d,v6.d[0]                 \n\t" // Scale by alpha
	" fmla v11.2d,v17.2d,v6.d[0]                 \n\t" // Scale by alpha
	" fmla v12.2d,v18.2d,v6.d[0]                 \n\t" // Scale by alpha
	" fmla v13.2d,v19.2d,v6.d[0]                 \n\t" // Scale by alpha
	"                                            \n\t"
	" str q8, [x21]                              \n\t" //Store column 2 of C
	" str q9, [x21, #16]                         \n\t"
	" str q10, [x21, #32]                        \n\t"
	"                                            \n\t"
	" str q11, [x22]                             \n\t" //Store column 3 of C
	" str q12, [x22, #16]                        \n\t"
	" str q13, [x22, #32]                        \n\t"
	"                                            \n\t"
	" dup  v0.2d, xzr                            \n\t"
	" dup  v1.2d, xzr                            \n\t"
	" dup  v2.2d, xzr                            \n\t"
	" dup  v3.2d, xzr                            \n\t"
	" dup  v4.2d, xzr                            \n\t"
	" dup  v5.2d, xzr                            \n\t"
	"                                            \n\t"
	" fcmp d7,#0.0                               \n\t"
	BEQ (DBETAZEROCOLSTOREDS3)                         // Taking care of the beta==0 case.
	"                                            \n\t"
	" ldr q0, [x23]                              \n\t" //Load column 4 of C
	" ldr q1, [x23, #16]                         \n\t"
	" ldr q2, [x23, #32]                         \n\t"
	"                                            \n\t"
	" ldr q3, [x24]                              \n\t" //Load column 5 of C
	" ldr q4, [x24, #16]                         \n\t"
	" ldr q5, [x24, #32]                         \n\t"
	"                                            \n\t"
	" fmul v0.2d,v0.2d,v7.d[0]                   \n\t" // Scale by beta
	" fmul v1.2d,v1.2d,v7.d[0]                   \n\t" // Scale by beta
	" fmul v2.2d,v2.2d,v7.d[0]                   \n\t" // Scale by beta
	" fmul v3.2d,v3.2d,v7.d[0]                   \n\t" // Scale by beta
	" fmul v4.2d,v4.2d,v7.d[0]                   \n\t" // Scale by beta
	" fmul v5.2d,v5.2d,v7.d[0]                   \n\t" // Scale by beta
	"                                            \n\t"
	LABEL (DBETAZEROCOLSTOREDS3)
	"                                            \n\t"
	" fmla v0.2d,v20.2d,v6.d[0]                  \n\t" // Scale by alpha
	" fmla v1.2d,v21.2d,v6.d[0]                  \n\t" // Scale by alpha
	" fmla v2.2d,v22.2d,v6.d[0]                  \n\t" // Scale by alpha
	" fmla v3.2d,v23.2d,v6.d[0]                  \n\t" // Scale by alpha
	" fmla v4.2d,v24.2d,v6.d[0]                  \n\t" // Scale by alpha
	" fmla v5.2d,v25.2d,v6.d[0]                  \n\t" // Scale by alpha
	"                                            \n\t"
	" str q0, [x23]                              \n\t" //Store column 4 of C
	" str q1, [x23, #16]                         \n\t"
	" str q2, [x23, #32]                         \n\t"
	"                                            \n\t"
	" str q3, [x24]                              \n\t" //Store column 5 of C
	" str q4, [x24, #16]                         \n\t"
	" str q5, [x24, #32]                         \n\t"
	"                                            \n\t"
	" dup  v8.2d, xzr                            \n\t"
	" dup  v9.2d, xzr                            \n\t"
	" dup  v10.2d, xzr                           \n\t"
	" dup  v11.2d, xzr                           \n\t"
	" dup  v12.2d, xzr                           \n\t"
	" dup  v13.2d, xzr                           \n\t"
	"                                            \n\t"
	" fcmp d7,#0.0                               \n\t"
	BEQ (DBETAZEROCOLSTOREDS4)                         // Taking care of the beta==0 case.
	"                                            \n\t"
	" ldr q8, [x25]                              \n\t" //Load column 6 of C
	" ldr q9, [x25, #16]                         \n\t"
	" ldr q10, [x25, #32]                        \n\t"
	"                                            \n\t"
	" ldr q11, [x26]                             \n\t" //Load column 7 of C
	" ldr q12, [x26, #16]                        \n\t"
	" ldr q13, [x26, #32]                        \n\t"
	"                                            \n\t"
	" fmul v8.2d, v8.2d, v7.d[0]                 \n\t" // Scale by beta
	" fmul v9.2d, v9.2d, v7.d[0]                 \n\t" // Scale by beta
	" fmul v10.2d,v10.2d,v7.d[0]                 \n\t" // Scale by beta
	" fmul v11.2d,v11.2d,v7.d[0]                 \n\t" // Scale by beta
	" fmul v12.2d,v12.2d,v7.d[0]                 \n\t" // Scale by beta
	" fmul v13.2d,v13.2d,v7.d[0]                 \n\t" // Scale by beta
	"                                            \n\t"
	LABEL (DBETAZEROCOLSTOREDS4)
	"                                            \n\t"
	"                                            \n\t"
	" fmla v8.2d, v26.2d,v6.d[0]                 \n\t" // Scale by alpha
	" fmla v9.2d, v27.2d,v6.d[0]                 \n\t" // Scale by alpha
	" fmla v10.2d,v28.2d,v6.d[0]                 \n\t" // Scale by alpha
	" fmla v11.2d,v29.2d,v6.d[0]                 \n\t" // Scale by alpha
	" fmla v12.2d,v30.2d,v6.d[0]                 \n\t" // Scale by alpha
	" fmla v13.2d,v31.2d,v6.d[0]                 \n\t" // Scale by alpha
	"                                            \n\t"
	" str q8, [x25]                              \n\t" //Store column 6 of C
	" str q9, [x25, #16]                         \n\t"
	" str q10, [x25, #32]                        \n\t"
	"                                            \n\t"
	" str q11, [x26]                             \n\t" //Store column 7 of C
	" str q12, [x26, #16]                        \n\t"
	" str q13, [x26, #32]                        \n\t"
	"                                            \n\t"
	BRANCH (DEND)
	"                                            \n\t"
	LABEL (DGENSTORED)                                 // C is general-stride stored.
	"                                            \n\t"
	" dup  v0.2d, xzr                            \n\t"
	" dup  v1.2d, xzr                            \n\t"
	" dup  v2.2d, xzr                            \n\t"
	" dup  v3.2d, xzr                            \n\t"
	" dup  v4.2d, xzr                            \n\t"
	" dup  v5.2d, xzr                            \n\t"
	"                                            \n\t"
	" fcmp d7,#0.0                               \n\t"
	BEQ (DBETAZEROGENSTOREDS1)                         // Taking care of the beta==0 case.
	"                                            \n\t"
	" mov x27, x2                                \n\t"
	"                                            \n\t" // Load address of C.
	" ld1 {v0.d}[0],[x27],x14                    \n\t" // Load c00  into quad and increment by rs_c.
	" ld1 {v0.d}[1],[x27],x14                    \n\t" // Load c01  into quad and increment by rs_c.
	" ld1 {v1.d}[0],[x27],x14                    \n\t" // Load c02  into quad and increment by rs_c.
	" ld1 {v1.d}[1],[x27],x14                    \n\t" // Load c03  into quad and increment by rs_c.
	" ld1 {v2.d}[0],[x27],x14                    \n\t" // Load c04  into quad and increment by rs_c.
	" ld1 {v2.d}[1],[x27],x14                    \n\t" // Load c05  into quad and increment by rs_c.
	"                                            \n\t"
	" mov x27, x20                               \n\t" // Load address of C.
	"                                            \n\t"
	" ld1 {v3.d}[0],[x27],x14                    \n\t" // Load c10  into quad and increment by rs_c.
	" ld1 {v3.d}[1],[x27],x14                    \n\t" // Load c11  into quad and increment by rs_c.
	" ld1 {v4.d}[0],[x27],x14                    \n\t" // Load c12  into quad and increment by rs_c.
	" ld1 {v4.d}[1],[x27],x14                    \n\t" // Load c13  into quad and increment by rs_c.
	" ld1 {v5.d}[0],[x27],x14                    \n\t" // Load c14  into quad and increment by rs_c.
	" ld1 {v5.d}[1],[x27],x14                    \n\t" // Load c15  into quad and increment by rs_c.
	"                                            \n\t"
	" fmul v0.2d,v0.2d,v7.d[0]                   \n\t" // Scale by beta
	" fmul v1.2d,v1.2d,v7.d[0]                   \n\t" // Scale by beta
	" fmul v2.2d,v2.2d,v7.d[0]                   \n\t" // Scale by beta
	" fmul v3.2d,v3.2d,v7.d[0]                   \n\t" // Scale by beta
	" fmul v4.2d,v4.2d,v7.d[0]                   \n\t" // Scale by beta
	" fmul v5.2d,v5.2d,v7.d[0]                   \n\t" // Scale by beta
	"                                            \n\t"
	LABEL (DBETAZEROGENSTOREDS1)
	"                                            \n\t"
	" fmla v0.2d,v8.2d,v6.d[0]                   \n\t" // Scale by alpha
	" fmla v1.2d,v9.2d,v6.d[0]                   \n\t" // Scale by alpha
	" fmla v2.2d,v10.2d,v6.d[0]                  \n\t" // Scale by alpha
	" fmla v3.2d,v11.2d,v6.d[0]                  \n\t" // Scale by alpha
	" fmla v4.2d,v12.2d,v6.d[0]                  \n\t" // Scale by alpha
	" fmla v5.2d,v13.2d,v6.d[0]                  \n\t" // Scale by alpha
	"                                            \n\t"
	" mov x27, x2                                \n\t" // Load address of C.
	"                                            \n\t"
	" st1 {v0.d}[0],[x27],x14                    \n\t" // Store c00  into quad and increment by rs_c.
	" st1 {v0.d}[1],[x27],x14                    \n\t" // Store c01  into quad and increment by rs_c.
	" st1 {v1.d}[0],[x27],x14                    \n\t" // Store c02  into quad and increment by rs_c.
	" st1 {v1.d}[1],[x27],x14                    \n\t" // Store c03  into quad and increment by rs_c.
	" st1 {v2.d}[0],[x27],x14                    \n\t" // Store c04  into quad and increment by rs_c.
	" st1 {v2.d}[1],[x27],x14                    \n\t" // Store c05  into quad and increment by rs_c.
	"                                            \n\t"
	" mov x27, x20                               \n\t" // Load address of C.
	"                                            \n\t"
	" st1 {v3.d}[0],[x27],x14                    \n\t" // Store c10  into quad and increment by rs_c.
	" st1 {v3.d}[1],[x27],x14                    \n\t" // Store c11  into quad and increment by rs_c.
	" st1 {v4.d}[0],[x27],x14                    \n\t" // Store c12  into quad and increment by rs_c.
	" st1 {v4.d}[1],[x27],x14                    \n\t" // Store c13  into quad and increment by rs_c.
	" st1 {v5.d}[0],[x27],x14                    \n\t" // Store c14  into quad and increment by rs_c.
	" st1 {v5.d}[1],[x27],x14                    \n\t" // Store c15  into quad and increment by rs_c.
	"                                            \n\t"
	" dup  v8.2d, xzr                            \n\t"
	" dup  v9.2d, xzr                            \n\t"
	" dup  v10.2d, xzr                           \n\t"
	" dup  v11.2d, xzr                           \n\t"
	" dup  v12.2d, xzr                           \n\t"
	" dup  v13.2d, xzr                           \n\t"
	"                                            \n\t"
	" fcmp d7,#0.0                               \n\t"
	BEQ (DBETAZEROGENSTOREDS2)                         // Taking care of the beta==0 case.
	"                                            \n\t"
	" mov x27, x21                               \n\t" // Load address of C.
	"                                            \n\t"
	" ld1 {v8.d}[0], [x27],x14                   \n\t" // Load c20  into quad and increment by rs_c.
	" ld1 {v8.d}[1], [x27],x14                   \n\t" // Load c21  into quad and increment by rs_c.
	" ld1 {v9.d}[0], [x27],x14                   \n\t" // Load c22  into quad and increment by rs_c.
	" ld1 {v9.d}[1], [x27],x14                   \n\t" // Load c23  into quad and increment by rs_c.
	" ld1 {v10.d}[0],[x27],x14                   \n\t" // Load c24  into quad and increment by rs_c.
	" ld1 {v10.d}[1],[x27],x14                   \n\t" // Load c25  into quad and increment by rs_c.
	"                                            \n\t"
	" mov x27, x22                               \n\t" // Load address of C.
	"                                            \n\t"
	" ld1 {v11.d}[0],[x27],x14                   \n\t" // Load c30  into quad and increment by rs_c.
	" ld1 {v11.d}[1],[x27],x14                   \n\t" // Load c31  into quad and increment by rs_c.
	" ld1 {v12.d}[0],[x27],x14                   \n\t" // Load c32  into quad and increment by rs_c.
	" ld1 {v12.d}[1],[x27],x14                   \n\t" // Load c33  into quad and increment by rs_c.
	" ld1 {v13.d}[0],[x27],x14                   \n\t" // Load c34  into quad and increment by rs_c.
	" ld1 {v13.d}[1],[x27],x14                   \n\t" // Load c35  into quad and increment by rs_c.
	"                                            \n\t"
	" fmul v8.2d, v8.2d, v7.d[0]                 \n\t" // Scale by beta
	" fmul v9.2d, v9.2d, v7.d[0]                 \n\t" // Scale by beta
	" fmul v10.2d,v10.2d,v7.d[0]                 \n\t" // Scale by beta
	" fmul v11.2d,v11.2d,v7.d[0]                 \n\t" // Scale by beta
	" fmul v12.2d,v12.2d,v7.d[0]                 \n\t" // Scale by beta
	" fmul v13.2d,v13.2d,v7.d[0]                 \n\t" // Scale by beta
	"                                            \n\t"
	LABEL (DBETAZEROGENSTOREDS2)
	"                                            \n\t"
	" fmla v8.2d, v14.2d,v6.d[0]                 \n\t" // Scale by alpha
	" fmla v9.2d, v15.2d,v6.d[0]                 \n\t" // Scale by alpha
	" fmla v10.2d,v16.2d,v6.d[0]                 \n\t" // Scale by alpha
	" fmla v11.2d,v17.2d,v6.d[0]                 \n\t" // Scale by alpha
	" fmla v12.2d,v18.2d,v6.d[0]                 \n\t" // Scale by alpha
	" fmla v13.2d,v19.2d,v6.d[0]                 \n\t" // Scale by alpha
	"                                            \n\t"
	" mov x27, x21                               \n\t" // Load address of C.
	"                                            \n\t"
	" st1 {v8.d}[0], [x27],x14                   \n\t" // Store c20  into quad and increment by rs_c.
	" st1 {v8.d}[1], [x27],x14                   \n\t" // Store c21  into quad and increment by rs_c.
	" st1 {v9.d}[0], [x27],x14                   \n\t" // Store c22  into quad and increment by rs_c.
	" st1 {v9.d}[1], [x27],x14                   \n\t" // Store c23  into quad and increment by rs_c.
	" st1 {v10.d}[0],[x27],x14                   \n\t" // Store c24  into quad and increment by rs_c.
	" st1 {v10.d}[1],[x27],x14                   \n\t" // Store c25  into quad and increment by rs_c.
	"                                            \n\t"
	" mov x27, x22                               \n\t" // Load address of C.
	"                                            \n\t"
	" st1 {v11.d}[0],[x27],x14                   \n\t" // Store c30  into quad and increment by rs_c.
	" st1 {v11.d}[1],[x27],x14                   \n\t" // Store c31  into quad and increment by rs_c.
	" st1 {v12.d}[0],[x27],x14                   \n\t" // Store c32  into quad and increment by rs_c.
	" st1 {v12.d}[1],[x27],x14                   \n\t" // Store c33  into quad and increment by rs_c.
	" st1 {v13.d}[0],[x27],x14                   \n\t" // Store c34  into quad and increment by rs_c.
	" st1 {v13.d}[1],[x27],x14                   \n\t" // Store c35  into quad and increment by rs_c.
	"                                            \n\t"
	" dup  v0.2d, xzr                            \n\t"
	" dup  v1.2d, xzr                            \n\t"
	" dup  v2.2d, xzr                            \n\t"
	" dup  v3.2d, xzr                            \n\t"
	" dup  v4.2d, xzr                            \n\t"
	" dup  v5.2d, xzr                            \n\t"
	"                                            \n\t"
	" fcmp d7,#0.0                               \n\t"
	BEQ (DBETAZEROGENSTOREDS3)                         // Taking care of the beta==0 case.
	"                                            \n\t"
	" mov x27, x23                               \n\t" // Load address of C.
	"                                            \n\t"
	" ld1 {v0.d}[0],[x27],x14                    \n\t" // Load c40  into quad and increment by rs_c.
	" ld1 {v0.d}[1],[x27],x14                    \n\t" // Load c41  into quad and increment by rs_c.
	" ld1 {v1.d}[0],[x27],x14                    \n\t" // Load c42  into quad and increment by rs_c.
	" ld1 {v1.d}[1],[x27],x14                    \n\t" // Load c43  into quad and increment by rs_c.
	" ld1 {v2.d}[0],[x27],x14                    \n\t" // Load c44  into quad and increment by rs_c.
	" ld1 {v2.d}[1],[x27],x14                    \n\t" // Load c45  into quad and increment by rs_c.
	"                                            \n\t"
	" mov x27, x24                               \n\t" // Load address of C.
	"                                            \n\t"
	" ld1 {v3.d}[0],[x27],x14                    \n\t" // Load c50  into quad and increment by rs_c.
	" ld1 {v3.d}[1],[x27],x14                    \n\t" // Load c51  into quad and increment by rs_c.
	" ld1 {v4.d}[0],[x27],x14                    \n\t" // Load c52  into quad and increment by rs_c.
	" ld1 {v4.d}[1],[x27],x14                    \n\t" // Load c53  into quad and increment by rs_c.
	" ld1 {v5.d}[0],[x27],x14                    \n\t" // Load c54  into quad and increment by rs_c.
	" ld1 {v5.d}[1],[x27],x14                    \n\t" // Load c55  into quad and increment by rs_c.
	"                                            \n\t"
	" fmul v0.2d,v0.2d,v7.d[0]                   \n\t" // Scale by beta
	" fmul v1.2d,v1.2d,v7.d[0]                   \n\t" // Scale by beta
	" fmul v2.2d,v2.2d,v7.d[0]                   \n\t" // Scale by beta
	" fmul v3.2d,v3.2d,v7.d[0]                   \n\t" // Scale by beta
	" fmul v4.2d,v4.2d,v7.d[0]                   \n\t" // Scale by beta
	" fmul v5.2d,v5.2d,v7.d[0]                   \n\t" // Scale by beta
	"                                            \n\t"
	LABEL (DBETAZEROGENSTOREDS3)
	"                                            \n\t"
	" fmla v0.2d,v20.2d,v6.d[0]                  \n\t" // Scale by alpha
	" fmla v1.2d,v21.2d,v6.d[0]                  \n\t" // Scale by alpha
	" fmla v2.2d,v22.2d,v6.d[0]                  \n\t" // Scale by alpha
	" fmla v3.2d,v23.2d,v6.d[0]                  \n\t" // Scale by alpha
	" fmla v4.2d,v24.2d,v6.d[0]                  \n\t" // Scale by alpha
	" fmla v5.2d,v25.2d,v6.d[0]                  \n\t" // Scale by alpha
	"                                            \n\t"
	" mov x27, x23                               \n\t" // Load address of C.
	"                                            \n\t"
	" st1 {v0.d}[0],[x27],x14                    \n\t" // Store c40  into quad and increment by rs_c.
	" st1 {v0.d}[1],[x27],x14                    \n\t" // Store c41  into quad and increment by rs_c.
	" st1 {v1.d}[0],[x27],x14                    \n\t" // Store c42  into quad and increment by rs_c.
	" st1 {v1.d}[1],[x27],x14                    \n\t" // Store c43  into quad and increment by rs_c.
	" st1 {v2.d}[0],[x27],x14                    \n\t" // Store c44  into quad and increment by rs_c.
	" st1 {v2.d}[1],[x27],x14                    \n\t" // Store c45  into quad and increment by rs_c.
	"                                            \n\t"
	" mov x27, x24                               \n\t" // Load address of C.
	"                                            \n\t"
	" st1 {v3.d}[0],[x27],x14                    \n\t" // Store c50  into quad and increment by rs_c.
	" st1 {v3.d}[1],[x27],x14                    \n\t" // Store c51  into quad and increment by rs_c.
	" st1 {v4.d}[0],[x27],x14                    \n\t" // Store c52  into quad and increment by rs_c.
	" st1 {v4.d}[1],[x27],x14                    \n\t" // Store c53  into quad and increment by rs_c.
	" st1 {v5.d}[0],[x27],x14                    \n\t" // Store c54  into quad and increment by rs_c.
	" st1 {v5.d}[1],[x27],x14                    \n\t" // Store c55  into quad and increment by rs_c.
	"                                            \n\t"
	" dup  v8.2d, xzr                            \n\t"
	" dup  v9.2d, xzr                            \n\t"
	" dup  v10.2d, xzr                           \n\t"
	" dup  v11.2d, xzr                           \n\t"
	" dup  v12.2d, xzr                           \n\t"
	" dup  v13.2d, xzr                           \n\t"
	"                                            \n\t"
	" fcmp d7,#0.0                               \n\t"
	BEQ (DBETAZEROGENSTOREDS4)                         // Taking care of the beta==0 case.
	"                                            \n\t"
	" mov x27, x25                               \n\t"
	"                                            \n\t"
	" ld1 {v8.d}[0], [x27],x14                   \n\t" // Load c60  into quad and increment by rs_c.
	" ld1 {v8.d}[1], [x27],x14                   \n\t" // Load c61  into quad and increment by rs_c.
	" ld1 {v9.d}[0], [x27],x14                   \n\t" // Load c62  into quad and increment by rs_c.
	" ld1 {v9.d}[1], [x27],x14                   \n\t" // Load c63  into quad and increment by rs_c.
	" ld1 {v10.d}[0],[x27],x14                   \n\t" // Load c64  into quad and increment by rs_c.
	" ld1 {v10.d}[1],[x27],x14                   \n\t" // Load c65  into quad and increment by rs_c.
	"                                            \n\t"
	" mov x27, x26                               \n\t" // Load address of C.
	"                                            \n\t"
	" ld1 {v11.d}[0],[x27],x14                   \n\t" // Load c70  into quad and increment by rs_c.
	" ld1 {v11.d}[1],[x27],x14                   \n\t" // Load c71  into quad and increment by rs_c.
	" ld1 {v12.d}[0],[x27],x14                   \n\t" // Load c72  into quad and increment by rs_c.
	" ld1 {v12.d}[1],[x27],x14                   \n\t" // Load c73  into quad and increment by rs_c.
	" ld1 {v13.d}[0],[x27],x14                   \n\t" // Load c74  into quad and increment by rs_c.
	" ld1 {v13.d}[1],[x27],x14                   \n\t" // Load c75  into quad and increment by rs_c.
	"                                            \n\t"
	" fmul v8.2d, v8.2d, v7.d[0]                 \n\t" // Scale by beta
	" fmul v9.2d, v9.2d, v7.d[0]                 \n\t" // Scale by beta
	" fmul v10.2d,v10.2d,v7.d[0]                 \n\t" // Scale by beta
	" fmul v11.2d,v11.2d,v7.d[0]                 \n\t" // Scale by beta
	" fmul v12.2d,v12.2d,v7.d[0]                 \n\t" // Scale by beta
	" fmul v13.2d,v13.2d,v7.d[0]                 \n\t" // Scale by beta
	"                                            \n\t"
	LABEL (DBETAZEROGENSTOREDS4)
	"                                            \n\t"
	"                                            \n\t"
	" fmla v8.2d, v26.2d,v6.d[0]                 \n\t" // Scale by alpha
	" fmla v9.2d, v27.2d,v6.d[0]                 \n\t" // Scale by alpha
	" fmla v10.2d,v28.2d,v6.d[0]                 \n\t" // Scale by alpha
	" fmla v11.2d,v29.2d,v6.d[0]                 \n\t" // Scale by alpha
	" fmla v12.2d,v30.2d,v6.d[0]                 \n\t" // Scale by alpha
	" fmla v13.2d,v31.2d,v6.d[0]                 \n\t" // Scale by alpha
	"                                            \n\t"
	" mov x27, x25                               \n\t" // Load address of C.
	"                                            \n\t"
	" st1 {v8.d}[0], [x27],x14                   \n\t" // Store c60  into quad and increment by rs_c.
	" st1 {v8.d}[1], [x27],x14                   \n\t" // Store c61  into quad and increment by rs_c.
	" st1 {v9.d}[0], [x27],x14                   \n\t" // Store c62  into quad and increment by rs_c.
	" st1 {v9.d}[1], [x27],x14                   \n\t" // Store c63  into quad and increment by rs_c.
	" st1 {v10.d}[0],[x27],x14                   \n\t" // Store c64  into quad and increment by rs_c.
	" st1 {v10.d}[1],[x27],x14                   \n\t" // Store c65  into quad and increment by rs_c.
	"                                            \n\t"
	" mov x27, x26                               \n\t" // Load address of C.
	"                                            \n\t"
	" st1 {v11.d}[0],[x27],x14                   \n\t" // Store c70  into quad and increment by rs_c.
	" st1 {v11.d}[1],[x27],x14                   \n\t" // Store c71  into quad and increment by rs_c.
	" st1 {v12.d}[0],[x27],x14                   \n\t" // Store c72  into quad and increment by rs_c.
	" st1 {v12.d}[1],[x27],x14                   \n\t" // Store c73  into quad and increment by rs_c.
	" st1 {v13.d}[0],[x27],x14                   \n\t" // Store c74  into quad and increment by rs_c.
	" st1 {v13.d}[1],[x27],x14                   \n\t" // Store c75  into quad and increment by rs_c.
	"                                            \n\t"
	LABEL (DEND)                                       // Done!
	"                                            \n\t"
	:// output operands (none)
	:// input operands
	 [aaddr] "m" (a),       // 0
	 [baddr] "m" (b),       // 1
	 [caddr] "m" (c),       // 2
	 [k_iter] "m" (k_iter), // 3
	 [k_left] "m" (k_left), // 4
	 [alpha] "m" (alpha),   // 5
	 [beta] "m" (beta),     // 6
	 [rs_c] "m" (rs_c),     // 7
	 [cs_c] "m" (cs_c),     // 8
	 [a_next] "m" (a_next), // 9
	 [b_next] "m" (b_next)  // 10
	 :// Register clobber list
	 "x0", "x1", "x2",
	 "x5", "x6", "x10", "x14",
	 "x16", "x17", "x19", "x20",
	 "x21", "x22", "x23", "x24",
	 "x25", "x26", "x27",
	 "v0", "v1", "v2", "v3",
	 "v4", "v5", "v6", "v7",
	 "v8", "v9", "v10", "v11",
	 "v12", "v13", "v14", "v15",
	 "v16", "v17", "v18", "v19",
	 "v20", "v21", "v22", "v23",
	 "v24", "v25", "v26", "v27",
	 "v28", "v29", "v30", "v31"
	);

	GEMM_UKR_FLUSH_CT( d );
}






