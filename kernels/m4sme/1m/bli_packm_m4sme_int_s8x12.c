/*
 *
 * BLIS An object-based framework for developing high-performance BLAS-like
 * libraries.
 *
 * Copyright (C) 2014, The University of Texas at Austin Copyright (C) 2020,
 * Linaro Limited
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are
 * met: - Redistributions of source code must retain the above copyright
 * notice, this list of conditions and the following disclaimer. -
 * Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution. - Neither the
 * name(s) of the copyright holder(s) nor the names of its contributors may
 * be used to endorse or promote products derived from this software without
 * specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
 * IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
 * THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
 * LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 */

#include "blis.h"
#include <arm_sme.h>
#include <arm_sve.h>

#if defined(__clang__)
#define PRAGMA_NOUNROLL _Pragma("nounroll")
#define PRAGMA_UNROLL_2 _Pragma("unroll 2")
#define PRAGMA_UNROLL_4 _Pragma("unroll 4")
#elif defined(__GNUC__)
#define PRAGMA_NOUNROLL _Pragma("GCC unroll 1")
#define PRAGMA_UNROLL_2 _Pragma("GCC unroll 2")
#define PRAGMA_UNROLL_4 _Pragma("GCC unroll 4")
#else
#define PRAGMA_NOUNROLL
#define PRAGMA_UNROLL_2
#define PRAGMA_UNROLL_4
#endif


__arm_new("za") __arm_locally_streaming
void		bli_spackm_m4sme_int_8x12
		(
		 conj_t conja,
		 pack_t schema,
		 dim_t cdim_,
		 dim_t cdim_max,
		 dim_t cdim_bcast,
		 dim_t n_,
		 dim_t n_max_,
		 const void *kappa,
		 const void *a, inc_t inca_, inc_t lda_,
		 void *p, inc_t ldp_,
		 const void *params,
		 const cntx_t * cntx
){
	const		int64_t cdim = cdim_;
	const		int64_t mr = 64;
	const		int64_t n = n_;
	const		int64_t inca = inca_;
	const		int64_t lda = lda_;
	const		int64_t ldp = ldp_;

	float	       *restrict a_ = (float *)a;
	float	       *restrict p_ = (float *)p;

	uint64_t	SVL = svcntsw();
	svfloat32x4_t	tmp;
	// svfloat32_t	tmp;

	const float    *restrict alpha1 = a;
	float	       *restrict pi1 = p;
		// printf("%d %d %d %d %d %d\n", cdim, mr, cdim_bcast, ldp, lda, n);
		// printf("Inca: %d\n", inca);
	
	if (true) {
		if (bli_seq1(*((float *)kappa))) {
			if (inca == 1)
				//continous memory.packA style
			{
				// printf("Pack A\n");
				for (dim_t k = n; k != 0; --k) {
					// tmp = svld1_f32(svptrue_b32(), alpha1);
					// svst1_f32(svptrue_b32(), pi1, tmp);
					tmp = svld1_f32_x4(svptrue_c32(), alpha1);
					svst1_f32_x4(svptrue_c32(), pi1, tmp);


					alpha1 += lda;
					pi1 += ldp;
				}

			} else {
				// printf("Pack B\n");
				{
					for (uint64_t col = 0; col < n; col += SVL) {
						for (uint64_t trow = 0; trow < SVL; trow += 4) {
							svbool_t	p0 = svptrue_b32();
							svbool_t	p1 = svptrue_b32();
							svbool_t	p2 = svptrue_b32();
							svbool_t	p3 = svptrue_b32();

							//const uint64_t tile_UL_corner = 0;
							//Load 4 rows of A as double vectors(from the upper part).
							//		zp0[0] - SVL | zp0[1] - SVL
							//		zp1[0] - SVL | zp1[1] - SVL
							//		zp2[0] - SVL | zp2[1] - SVL
							//		zp3[0] - SVL | zp3[1] - SVL
							//		Load 4 rows of A as double vectors(from the bottom part).
							//		zp4[0] - SVL | zp4[1] - SVL
							//		zp5[0] - SVL | zp1[1] - SVL
							//		zp6[0] - SVL | zp6[1] - SVL
							//		zp7[0] - SVL | zp7[1] - SVL

							const		uint64_t tile_UL_corner = ( /* row + */ trow) * inca /* n */ + col;
							svfloat32_t	zp0 = svld1_f32(p0, &a_[tile_UL_corner + 0 * inca]);
							svfloat32_t	zp1 = svld1_f32(p1, &a_[tile_UL_corner + 1 * inca]);
							svfloat32_t	zp2 = svld1_f32(p2, &a_[tile_UL_corner + 2 * inca]);
							svfloat32_t	zp3 = svld1_f32(p3, &a_[tile_UL_corner + 3 * inca]);
							int cuartSVL = SVL / 4;
							const		uint64_t tile_BL_corner = tile_UL_corner + inca * SVL;
							//+col;
							//+col;
							// printf("%d %d\n", tile_UL_corner, tile_BL_corner);
							svfloat32_t	zp4 = svld1_f32(p0, &a_[tile_BL_corner + 0 * inca]);
							svfloat32_t	zp5 = svld1_f32(p1, &a_[tile_BL_corner + 1 * inca]);
							svfloat32_t	zp6 = svld1_f32(p2, &a_[tile_BL_corner + 2 * inca]);
							svfloat32_t	zp7 = svld1_f32(p3, &a_[tile_BL_corner + 3 * inca]);

							const		uint64_t tile_BBL_corner = tile_UL_corner +  2 * inca * SVL;
							svfloat32_t	zp8 = svld1_f32(p0, &a_[tile_BBL_corner + 0 * inca]);
							svfloat32_t	zp9 = svld1_f32(p1, &a_[tile_BBL_corner + 1 * inca]);
							svfloat32_t	zp10 = svld1_f32(p2, &a_[tile_BBL_corner + 2 * inca]);
							svfloat32_t	zp11 = svld1_f32(p3, &a_[tile_BBL_corner + 3 * inca]);

							const		uint64_t tile_BBBL_corner = tile_UL_corner + 3 * inca * SVL;
							//+col;
							//+col;
							// printf("%d %d\n", tile_UL_corner, tile_BL_corner);
							svfloat32_t	zp12 = svld1_f32(p0, &a_[tile_BBBL_corner + 0 * inca]);
							svfloat32_t	zp13 = svld1_f32(p1, &a_[tile_BBBL_corner + 1 * inca]);
							svfloat32_t	zp14 = svld1_f32(p2, &a_[tile_BBBL_corner + 2 * inca]);
							svfloat32_t	zp15 = svld1_f32(p3, &a_[tile_BBBL_corner + 3 * inca]);

							//zq0 < -zp0[0] | zp1[0] | zp2[0] | zp3[0]
								// zq1 < -zp0[1] | zp1[1] | zp2[1] | zp3[1]
								//
								//zq2 < -zp4[0] | zp5[0] | zp6[0] | zp7[0]
								// zq3 < -zp4[1] | zp5[1] | zp6[1] | zp7[1]
							svfloat32x4_t zq0 = svcreate4(zp0, zp1,
											zp2, zp3);
							svfloat32x4_t zq1 = svcreate4(zp4, zp5,
											zp6, zp7);				
							svfloat32x4_t zq2 = svcreate4(zp8, zp9,
											zp10, zp11);
							svfloat32x4_t zq3 = svcreate4(zp12, zp13,
											zp14, zp15);	
					//ZA contents:
					//Tile 0:SVL rows(top) x SVL columns(left).Tile 1:SVL rows(top) x SVL columns(right)
					// Tile 2:SVL rows(bot) x SVL columns(left).Tile 3:SVL rows(bot) x SVL columns(right)

							svwrite_hor_za32_f32_vg4(
											  /* tile: */ 0, /* slice: */ trow, zq0);
							svwrite_hor_za32_f32_vg4(
										  /* tile: */ 1, /* slice: */ trow, zq1);
							svwrite_hor_za32_f32_vg4(
										  /* tile: */ 2, /* slice: */ trow, zq2);
							svwrite_hor_za32_f32_vg4(
										  /* tile: */ 3, /* slice: */ trow, zq3);
						}

						//Read - as - columns and store
							for (uint64_t tcol = 0; tcol < SVL; tcol += 4) {
							svcount_t	p0 = svptrue_c32();

							//Each svread_ver reads 4 columns of the tile(SVL).
								svfloat32x4_t zq0 = svread_ver_za32_f32_vg4( /* tile: */ 0, /* slice: */ tcol);
							svfloat32x4_t	zq2 = svread_ver_za32_f32_vg4( /* tile: */ 2, /* slice: */ tcol);

							svfloat32x4_t	zq1 = svread_ver_za32_f32_vg4( /* tile: */ 1, /* slice: */ tcol);
							svfloat32x4_t	zq3 = svread_ver_za32_f32_vg4( /* tile: */ 3, /* slice: */ tcol);

							svfloat32x4_t	zq0_ = svcreate4(svget4(zq0, 0), svget4(zq1, 0),
											 svget4(zq2, 0), svget4(zq3, 0));

							svfloat32x4_t	zq1_ = svcreate4(svget4(zq0, 1), svget4(zq1, 1),
											 svget4(zq2, 1), svget4(zq3, 1));

							svfloat32x4_t	zq2_ = svcreate4(svget4(zq0, 2), svget4(zq1, 2),
											 svget4(zq2, 2), svget4(zq3, 2));

							svfloat32x4_t	zq3_ = svcreate4(svget4(zq0, 3), svget4(zq1, 3),
											 svget4(zq2, 3), svget4(zq3, 3));

							svst1(p0, &p_[0], zq0_);
							svst1(p0, &p_[4 * SVL], zq1_);
							svst1(p0, &p_[8 * SVL], zq2_);
							svst1(p0, &p_[12 * SVL], zq3_);

							p_ += (16 * SVL);

						}
						//p_ += (2 * SVL * SVL);
					}
				}

				p_ = (float *)p;

			}
		} else {
			bli_sscal2bbs_mxn
				(
				 conja,
				 cdim_,
				 n_,
				 kappa,
				 a, inca, lda,
				 p_, cdim_bcast, ldp
				);

		}
	} else {
		bli_sscal2bbs_mxn
			(
			 conja,
			 cdim_,
			 n_,
			 kappa,
			 a, inca, lda,
			 p_, cdim_bcast, ldp
			);


	}

	bli_sset0s_edge
		(
		 cdim_ * cdim_bcast, cdim_max * cdim_bcast,
		 n_, n_max_,
		 p_, ldp
		);
}

