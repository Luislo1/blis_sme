/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas at Austin
   Copyright (C) 2020, Linaro Limited

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

#include <arm_sme.h>
#include <arm_sve.h>

#include "blis.h"

__arm_new( "za" ) __arm_locally_streaming void bli_dpackm_m4sme_int_SVLx8SVL
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
	)
{
	const int64_t cdim = cdim_;
	const int64_t n = n_;
	const int64_t inca = inca_;
	const int64_t lda = lda_;
	const int64_t ldp = ldp_;

	double* restrict a_ = (double*)a;
	double* restrict p_ = (double*)p;

	uint64_t SVL = svcntsd();

	svfloat64x4_t tmp, tmp2;
	svfloat64_t tmp3;

	const double* restrict alpha1 = a;
	double* restrict pi1 = p;

	const bool gs = ( inca != 1 && lda != 1 );

	if ( !gs && ( cdim == ( 8 * SVL ) || cdim == SVL ) && cdim_bcast )
	{
		if ( bli_deq1( *( (double*)kappa ) ) )
		{
			if ( inca == 1 && ldp == 8 * SVL )
			// continous memory.packA style
			{
				for ( dim_t k = n; k != 0; --k )
				{
					tmp = svld1_f64_x4( svptrue_c32(), alpha1 );
					tmp2 = svld1_f64_x4( svptrue_c32(), alpha1 + 4 * SVL );
					svst1_f64_x4( svptrue_c32(), pi1, tmp );
					svst1_f64_x4( svptrue_c32(), pi1 + 4 * SVL, tmp2 );

					alpha1 += lda;
					pi1 += ldp;
				}
			}
			if ( inca == 1 && ldp == SVL )
			// continous memory.packA style
			{
				for ( dim_t k = n; k != 0; --k )
				{
					tmp3 = svld1_f64( svptrue_b32(), alpha1 );
					svst1_f64( svptrue_b32(), pi1, tmp3 );

					alpha1 += lda;
					pi1 += ldp;
				}
			}
			else if ( inca != 1 && ldp == SVL )
			{
				{
					for ( uint64_t col = 0; col < n; col += 8 * SVL )
					{
						for ( uint64_t trow = 0; trow < SVL; trow += 4 )
						{
							svcount_t p0 = svptrue_c32();

							//	Load 4 rows of A as double vectors 
							//	(from the upper part).
							//	zp0[0] - SVL | zp0[1] - SVL
							//	zp1[0] - SVL | zp1[1] - SVL
							//	zp2[0] - SVL | zp2[1] - SVL
							//	zp3[0] - SVL | zp3[1] - SVL
							//	Load 4 rows of A as double vectors
							//	(from the bottom part).
							//	zp4[0] - SVL | zp4[1] - SVL
							//	zp5[0] - SVL | zp1[1] - SVL
							//	zp6[0] - SVL | zp6[1] - SVL
							//	zp7[0] - SVL | zp7[1] - SVL

							const uint64_t tile_UL_corner =
								( /* row + */ trow ) * inca /* n */ + col;

							svfloat64x4_t zp0 = svld1_f64_x4( p0,
								&a_[tile_UL_corner + 0 * inca] );
							svfloat64x4_t zp1 = svld1_f64_x4( p0,
								&a_[tile_UL_corner + 1 * inca] );
							svfloat64x4_t zp2 = svld1_f64_x4( p0,
								&a_[tile_UL_corner + 2 * inca] );
							svfloat64x4_t zp3 = svld1_f64_x4( p0,
								&a_[tile_UL_corner + 3 * inca] );

							// zq0 < -zp0[0] | zp1[0] | zp2[0] | zp3[0]
							//  zq1 < -zp0[1] | zp1[1] | zp2[1] | zp3[1]
							//
							// zq2 < -zp4[0] | zp5[0] | zp6[0] | zp7[0]
							//  zq3 < -zp4[1] | zp5[1] | zp6[1] | zp7[1]

							svfloat64x4_t zq0 = svcreate4( svget4( zp0, 0 ),
								svget4( zp1, 0 ), svget4( zp2, 0 ),
								svget4( zp3, 0 ) );

							svfloat64x4_t zq1 = svcreate4( svget4( zp0, 1 ),
								svget4( zp1, 1 ), svget4( zp2, 1 ),
								svget4( zp3, 1 ) );

							svfloat64x4_t zq2 = svcreate4( svget4( zp0, 2 ),
								svget4( zp1, 2 ), svget4( zp2, 2 ),
								svget4( zp3, 2 ) );

							svfloat64x4_t zq3 = svcreate4( svget4( zp0, 3 ),
								svget4( zp1, 3 ), svget4( zp2, 3 ),
								svget4( zp3, 3 ) );

							// ZA contents:
							// Tile 0:SVL rows(top) x SVL columns(left)
							// Tile 1:SVL rows(top) x SVL columns(right)
							// Tile 2:SVL rows(bot) x SVL columns(left)
							// Tile 3:SVL rows(bot) x SVL columns(right)

							svwrite_hor_za64_f64_vg4(
								/* tile: */ 0, /* slice: */ trow, zq0 );
							svwrite_hor_za64_f64_vg4(
								/* tile: */ 1, /* slice: */ trow, zq1 );
							svwrite_hor_za64_f64_vg4(
								/* tile: */ 2, /* slice: */ trow, zq2 );
							svwrite_hor_za64_f64_vg4(
								/* tile: */ 3, /* slice: */ trow, zq3 );

							svfloat64x4_t zp4 = svld1_f64_x4( p0,
								&a_[tile_UL_corner + 0 * inca + 4 * SVL] );
							svfloat64x4_t zp5 = svld1_f64_x4( p0,
								&a_[tile_UL_corner + 1 * inca + 4 * SVL] );
							svfloat64x4_t zp6 = svld1_f64_x4( p0,
								&a_[tile_UL_corner + 2 * inca + 4 * SVL] );
							svfloat64x4_t zp7 = svld1_f64_x4( p0,
								&a_[tile_UL_corner + 3 * inca + 4 * SVL] );

							// zq0 < -zp0[0] | zp1[0] | zp2[0] | zp3[0]
							//  zq1 < -zp0[1] | zp1[1] | zp2[1] | zp3[1]
							//
							// zq2 < -zp4[0] | zp5[0] | zp6[0] | zp7[0]
							//  zq3 < -zp4[1] | zp5[1] | zp6[1] | zp7[1]

							svfloat64x4_t zq4 = svcreate4( svget4( zp4, 0 ),
								svget4( zp5, 0 ), svget4( zp6, 0 ),
								svget4( zp7, 0 ) );

							svfloat64x4_t zq5 = svcreate4( svget4( zp4, 1 ),
								svget4( zp5, 1 ), svget4( zp6, 1 ),
								svget4( zp7, 1 ) );

							svfloat64x4_t zq6 = svcreate4( svget4( zp4, 2 ),
								svget4( zp5, 2 ), svget4( zp6, 2 ),
								svget4( zp7, 2 ) );

							svfloat64x4_t zq7 = svcreate4( svget4( zp4, 3 ),
								svget4( zp5, 3 ), svget4( zp6, 3 ),
								svget4( zp7, 3 ) );

							// ZA contents:
							// Tile 0:SVL rows(top) x SVL columns(left).Tile
							// 1:SVL rows(top) x SVL columns(right)
							//  Tile 2:SVL rows(bot) x SVL columns(left).Tile
							//  3:SVL rows(bot) x SVL columns(right)

							svwrite_hor_za64_f64_vg4(
								/* tile: */ 4, /* slice: */ trow, zq4 );
							svwrite_hor_za64_f64_vg4(
								/* tile: */ 5, /* slice: */ trow, zq5 );
							svwrite_hor_za64_f64_vg4(
								/* tile: */ 6, /* slice: */ trow, zq6 );
							svwrite_hor_za64_f64_vg4(
								/* tile: */ 7, /* slice: */ trow, zq7 );
						}

						// Read - as - columns and store
						for ( uint64_t tcol = 0; tcol < SVL; tcol += 4 )
						{
							svcount_t p0 = svptrue_c32();

							// Each svread_ver reads 4 columns of the tile(SVL).
							svfloat64x4_t zq0 = svread_ver_za64_f64_vg4(
								/* tile: */ 0, /* slice: */ tcol );
							svfloat64x4_t zq2 = svread_ver_za64_f64_vg4(
								/* tile: */ 2, /* slice: */ tcol );

							svfloat64x4_t zq1 = svread_ver_za64_f64_vg4(
								/* tile: */ 1, /* slice: */ tcol );
							svfloat64x4_t zq3 = svread_ver_za64_f64_vg4(
								/* tile: */ 3, /* slice: */ tcol );

							svst1( p0, &p_[0], zq0 );
							svst1( p0, &p_[SVL * SVL], zq1 );
							svst1( p0, &p_[2 * SVL * SVL], zq2 );
							svst1( p0, &p_[3 * SVL * SVL], zq3 );

							// Each svread_ver reads 4 columns of the tile(SVL).
							svfloat64x4_t zq4 = svread_ver_za64_f64_vg4(
								/* tile: */ 4, /* slice: */ tcol );
							svfloat64x4_t zq5 = svread_ver_za64_f64_vg4(
								/* tile: */ 5, /* slice: */ tcol );

							svfloat64x4_t zq6 = svread_ver_za64_f64_vg4(
								/* tile: */ 6, /* slice: */ tcol );
							svfloat64x4_t zq7 = svread_ver_za64_f64_vg4(
								/* tile: */ 7, /* slice: */ tcol );

							svst1( p0, &p_[4 * SVL * SVL], zq4 );
							svst1( p0, &p_[5 * SVL * SVL], zq5 );
							svst1( p0, &p_[6 * SVL * SVL], zq6 );
							svst1( p0, &p_[7 * SVL * SVL], zq7 );

							p_ += ( 4 * SVL );
						}
						p_ += ( 7 * SVL * SVL );
					}
				}

				p_ = (double*)p;
			}
			else if ( inca != 1 && ldp == 8 * SVL )
			{
				{
					for ( uint64_t col = 0; col < n; col += SVL )
					{
						for ( uint64_t trow = 0; trow < SVL; trow += 4 )
						{
							svbool_t p0 = svptrue_b32();

							//	Load 4 rows of A as double vectors 
							//	(from the upper part).
							//	zp0[0] - SVL | zp0[1] - SVL
							//	zp1[0] - SVL | zp1[1] - SVL
							//	zp2[0] - SVL | zp2[1] - SVL
							//	zp3[0] - SVL | zp3[1] - SVL
							//	Load 4 rows of A as double vectors
							//	(from the bottom part).
							//	zp4[0] - SVL | zp4[1] - SVL
							//	zp5[0] - SVL | zp1[1] - SVL
							//	zp6[0] - SVL | zp6[1] - SVL
							//	zp7[0] - SVL | zp7[1] - SVL

							const uint64_t tile_0 =
								( /* row + */ trow ) * inca /* n */ + col;
							const uint64_t tile_1 = tile_0 + inca * SVL;
							const uint64_t tile_2 = tile_0 + inca * 2 * SVL;
							const uint64_t tile_3 = tile_0 + inca * 3 * SVL;
							const uint64_t tile_4 = tile_0 + inca * 4 * SVL;
							const uint64_t tile_5 = tile_0 + inca * 5 * SVL;
							const uint64_t tile_6 = tile_0 + inca * 6 * SVL;
							const uint64_t tile_7 = tile_0 + inca * 7 * SVL;

							svfloat64_t zp0 = svld1_f64( p0,
								&a_[tile_0 + 0 * inca] );
							svfloat64_t zp1 = svld1_f64( p0,
								&a_[tile_0 + 1 * inca] );
							svfloat64_t zp2 = svld1_f64( p0,
								&a_[tile_0 + 2 * inca] );
							svfloat64_t zp3 = svld1_f64( p0,
								&a_[tile_0 + 3 * inca] );

							svfloat64_t zp4 = svld1_f64( p0,
								&a_[tile_1 + 0 * inca] );
							svfloat64_t zp5 = svld1_f64( p0,
								&a_[tile_1 + 1 * inca] );
							svfloat64_t zp6 = svld1_f64( p0,
								&a_[tile_1 + 2 * inca] );
							svfloat64_t zp7 = svld1_f64( p0,
								&a_[tile_1 + 3 * inca] );

							svfloat64_t zp8 = svld1_f64( p0,
								&a_[tile_2 + 0 * inca] );
							svfloat64_t zp9 = svld1_f64( p0,
								&a_[tile_2 + 1 * inca] );
							svfloat64_t zp10 = svld1_f64( p0,
								&a_[tile_2 + 2 * inca] );
							svfloat64_t zp11 = svld1_f64( p0,
								&a_[tile_2 + 3 * inca] );

							svfloat64_t zp12 = svld1_f64( p0,
								&a_[tile_3 + 0 * inca] );
							svfloat64_t zp13 = svld1_f64( p0,
								&a_[tile_3 + 1 * inca] );
							svfloat64_t zp14 = svld1_f64( p0,
								&a_[tile_3 + 2 * inca] );
							svfloat64_t zp15 = svld1_f64( p0,
								&a_[tile_3 + 3 * inca] );

							// zq0 < -zp0[0] | zp1[0] | zp2[0] | zp3[0]
							//  zq1 < -zp0[1] | zp1[1] | zp2[1] | zp3[1]
							//
							// zq2 < -zp4[0] | zp5[0] | zp6[0] | zp7[0]
							//  zq3 < -zp4[1] | zp5[1] | zp6[1] | zp7[1]

							svfloat64x4_t zq0 = svcreate4( zp0, zp1, zp2, zp3 );

							svfloat64x4_t zq1 = svcreate4( zp4, zp5, zp6, zp7 );

							svfloat64x4_t zq2 = svcreate4( zp8, zp9, zp10,
								zp11 );

							svfloat64x4_t zq3 = svcreate4( zp12, zp13, zp14,
								zp15 );

							// ZA contents:
							// Tile 0:SVL rows(top) x SVL columns(left).Tile
							// 1:SVL rows(top) x SVL columns(right)
							//  Tile 2:SVL rows(bot) x SVL columns(left).Tile
							//  3:SVL rows(bot) x SVL columns(right)

							svwrite_hor_za64_f64_vg4(
								/* tile: */ 0, /* slice: */ trow, zq0 );
							svwrite_hor_za64_f64_vg4(
								/* tile: */ 1, /* slice: */ trow, zq1 );
							svwrite_hor_za64_f64_vg4(
								/* tile: */ 2, /* slice: */ trow, zq2 );
							svwrite_hor_za64_f64_vg4(
								/* tile: */ 3, /* slice: */ trow, zq3 );

							svfloat64_t zp16 = svld1_f64( p0,
								&a_[tile_4 + 0 * inca] );
							svfloat64_t zp17 = svld1_f64( p0,
								&a_[tile_4 + 1 * inca] );
							svfloat64_t zp18 = svld1_f64( p0,
								&a_[tile_4 + 2 * inca] );
							svfloat64_t zp19 = svld1_f64( p0,
								&a_[tile_4 + 3 * inca] );

							svfloat64_t zp20 = svld1_f64( p0,
								&a_[tile_5 + 0 * inca] );
							svfloat64_t zp21 = svld1_f64( p0,
								&a_[tile_5 + 1 * inca] );
							svfloat64_t zp22 = svld1_f64( p0,
								&a_[tile_5 + 2 * inca] );
							svfloat64_t zp23 = svld1_f64( p0,
								&a_[tile_5 + 3 * inca] );

							svfloat64_t zp24 = svld1_f64( p0,
								&a_[tile_6 + 0 * inca] );
							svfloat64_t zp25 = svld1_f64( p0,
								&a_[tile_6 + 1 * inca] );
							svfloat64_t zp26 = svld1_f64( p0,
								&a_[tile_6 + 2 * inca] );
							svfloat64_t zp27 = svld1_f64( p0,
								&a_[tile_6 + 3 * inca] );

							svfloat64_t zp28 = svld1_f64( p0,
								&a_[tile_7 + 0 * inca] );
							svfloat64_t zp29 = svld1_f64( p0,
								&a_[tile_7 + 1 * inca] );
							svfloat64_t zp30 = svld1_f64( p0,
								&a_[tile_7 + 2 * inca] );
							svfloat64_t zp31 = svld1_f64( p0,
								&a_[tile_7 + 3 * inca] );

							// zq0 < -zp0[0] | zp1[0] | zp2[0] | zp3[0]
							//  zq1 < -zp0[1] | zp1[1] | zp2[1] | zp3[1]
							//
							// zq2 < -zp4[0] | zp5[0] | zp6[0] | zp7[0]
							//  zq3 < -zp4[1] | zp5[1] | zp6[1] | zp7[1]

							svfloat64x4_t zq4 = svcreate4( zp16, zp17, zp18,
								zp19 );

							svfloat64x4_t zq5 = svcreate4( zp20, zp21, zp22,
								zp23 );

							svfloat64x4_t zq6 = svcreate4( zp24, zp25, zp26,
								zp27 );

							svfloat64x4_t zq7 = svcreate4( zp28, zp29, zp30,
								zp31 );

							// ZA contents:
							// Tile 0:SVL rows(top) x SVL columns(left).Tile
							// 1:SVL rows(top) x SVL columns(right)
							//  Tile 2:SVL rows(bot) x SVL columns(left).Tile
							//  3:SVL rows(bot) x SVL columns(right)

							svwrite_hor_za64_f64_vg4(
								/* tile: */ 4, /* slice: */ trow, zq4 );
							svwrite_hor_za64_f64_vg4(
								/* tile: */ 5, /* slice: */ trow, zq5 );
							svwrite_hor_za64_f64_vg4(
								/* tile: */ 6, /* slice: */ trow, zq6 );
							svwrite_hor_za64_f64_vg4(
								/* tile: */ 7, /* slice: */ trow, zq7 );
						}

						// Read - as - columns and store
						for ( uint64_t tcol = 0; tcol < SVL; tcol += 4 )
						{
							svcount_t p0 = svptrue_c32();

							// Each svread_ver reads 4 columns of the tile(SVL).
							svfloat64x4_t zq0 = svread_ver_za64_f64_vg4(
								/* tile: */ 0, /* slice: */ tcol );
							svfloat64x4_t zq2 = svread_ver_za64_f64_vg4(
								/* tile: */ 2, /* slice: */ tcol );

							svfloat64x4_t zq1 = svread_ver_za64_f64_vg4(
								/* tile: */ 1, /* slice: */ tcol );
							svfloat64x4_t zq3 = svread_ver_za64_f64_vg4(
								/* tile: */ 3, /* slice: */ tcol );

							svfloat64x4_t zq0_ = svcreate4( svget4( zq0, 0 ),
								svget4( zq1, 0 ), svget4( zq2, 0 ),
								svget4( zq3, 0 ) );

							svfloat64x4_t zq1_ = svcreate4( svget4( zq0, 1 ),
								svget4( zq1, 1 ), svget4( zq2, 1 ),
								svget4( zq3, 1 ) );

							svfloat64x4_t zq2_ = svcreate4( svget4( zq0, 2 ),
								svget4( zq1, 2 ), svget4( zq2, 2 ),
								svget4( zq3, 2 ) );

							svfloat64x4_t zq3_ = svcreate4( svget4( zq0, 3 ),
								svget4( zq1, 3 ), svget4( zq2, 3 ),
								svget4( zq3, 3 ) );

							svst1( p0, &p_[0], zq0_ );
							svst1( p0, &p_[SVL * SVL], zq1_ );
							svst1( p0, &p_[2 * SVL * SVL], zq2_ );
							svst1( p0, &p_[3 * SVL * SVL], zq3_ );

							// Each svread_ver reads 4 columns of the tile(SVL).
							svfloat64x4_t zq4 = svread_ver_za64_f64_vg4(
								/* tile: */ 4, /* slice: */ tcol );
							svfloat64x4_t zq5 = svread_ver_za64_f64_vg4(
								/* tile: */ 5, /* slice: */ tcol );

							svfloat64x4_t zq6 = svread_ver_za64_f64_vg4(
								/* tile: */ 6, /* slice: */ tcol );
							svfloat64x4_t zq7 = svread_ver_za64_f64_vg4(
								/* tile: */ 7, /* slice: */ tcol );

							svfloat64x4_t zq4_ = svcreate4( svget4( zq4, 0 ),
								svget4( zq5, 0 ), svget4( zq6, 0 ),
								svget4( zq7, 0 ) );

							svfloat64x4_t zq5_ = svcreate4( svget4( zq4, 1 ),
								svget4( zq5, 1 ), svget4( zq6, 1 ),
								svget4( zq7, 1 ) );

							svfloat64x4_t zq6_ = svcreate4( svget4( zq4, 2 ),
								svget4( zq5, 2 ), svget4( zq6, 2 ),
								svget4( zq7, 2 ) );

							svfloat64x4_t zq7_ = svcreate4( svget4( zq4, 3 ),
								svget4( zq5, 3 ), svget4( zq6, 3 ),
								svget4( zq7, 3 ) );

							svst1( p0, &p_[4 * SVL], zq4_ );
							svst1( p0, &p_[SVL * SVL + 4 * SVL], zq5_ );
							svst1( p0, &p_[2 * SVL * SVL + 4 * SVL], zq6_ );
							svst1( p0, &p_[3 * SVL * SVL + 4 * SVL], zq7_ );

							p_ += ( 4 * SVL * SVL );
						}
					}
				}

				p_ = (double*)p;
			}
		}
		else 
		{
			bli_dscal2bbs_mxn
				(
				 conja,
				 cdim_,
				 n_,
				 kappa,
				 a, inca, lda,
				 p_, cdim_bcast, ldp
				);
		}
	} 
	else 
	{
		bli_dscal2bbs_mxn
			(
			 conja,
			 cdim_,
			 n_,
			 kappa,
			 a, inca, lda,
			 p_, cdim_bcast, ldp
			);
	}

	bli_dset0s_edge
		(
		 cdim_ * cdim_bcast, cdim_max * cdim_bcast,
		 n_, n_max_,
		 p_, ldp
		);
}
