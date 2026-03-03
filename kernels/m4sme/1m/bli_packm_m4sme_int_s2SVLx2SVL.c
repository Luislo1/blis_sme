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

#include <arm_sme.h>
#include <arm_sve.h>

#include "blis.h"

// MACROS FOR FALLTHROUGH LOGIC

// 1. Execution Blocks combined with storing

// [FULL] Stores 8 Vectors
#define OP_VG4_1( tcol, p_ )                    \
	{                                           \
		svbool_t p0 = svptrue_b32();            \
		svfloat32_t z0_0 = svread_ver_za32_m( svundef_f32(), p0, 0, tcol + 0 ); \
		svfloat32_t z0_1 = svread_ver_za32_m( svundef_f32(), p0, 2, tcol + 0 ); \
		svfloat32_t z0_2 = svread_ver_za32_m( svundef_f32(), p0, 0, tcol + 1 ); \
		svfloat32_t z0_3 = svread_ver_za32_m( svundef_f32(), p0, 2, tcol + 1 ); \
		svfloat32_t z1_0 = svread_ver_za32_m( svundef_f32(), p0, 0, tcol + 2 ); \
		svfloat32_t z1_1 = svread_ver_za32_m( svundef_f32(), p0, 2, tcol + 2 ); \
		svfloat32_t z1_2 = svread_ver_za32_m( svundef_f32(), p0, 0, tcol + 3 ); \
		svfloat32_t z1_3 = svread_ver_za32_m( svundef_f32(), p0, 2, tcol + 3 ); \
		svst1_f32( p0, &p_[0], z0_0 );          \
		svst1_f32( p0, &p_[1 * SVL], z0_1 );    \
		svst1_f32( p0, &p_[2 * SVL], z0_2 );    \
		svst1_f32( p0, &p_[3 * SVL], z0_3 );    \
		svst1_f32( p0, &p_[4 * SVL], z1_0 );    \
		svst1_f32( p0, &p_[5 * SVL], z1_1 );    \
		svst1_f32( p0, &p_[6 * SVL], z1_2 );    \
		svst1_f32( p0, &p_[7 * SVL], z1_3 );    \
		p_ += ( 8 * SVL );                      \
	}

#define OP_VG4_2( tcol, p_ )                    \
	{                                           \
		svbool_t p0 = svptrue_b32();            \
		svfloat32_t z1_0 = svread_ver_za32_m( svundef_f32(), p0, 1, tcol + 0 ); \
		svfloat32_t z1_1 = svread_ver_za32_m( svundef_f32(), p0, 3, tcol + 0 ); \
		svfloat32_t z1_2 = svread_ver_za32_m( svundef_f32(), p0, 1, tcol + 1 ); \
		svfloat32_t z1_3 = svread_ver_za32_m( svundef_f32(), p0, 3, tcol + 1 ); \
		svfloat32_t z3_0 = svread_ver_za32_m( svundef_f32(), p0, 1, tcol + 2 ); \
		svfloat32_t z3_1 = svread_ver_za32_m( svundef_f32(), p0, 3, tcol + 2 ); \
		svfloat32_t z3_2 = svread_ver_za32_m( svundef_f32(), p0, 1, tcol + 3 ); \
		svfloat32_t z3_3 = svread_ver_za32_m( svundef_f32(), p0, 3, tcol + 3 ); \
		svst1_f32( p0, &p_[0], z1_0 );          \
		svst1_f32( p0, &p_[1 * SVL], z1_1 );    \
		svst1_f32( p0, &p_[2 * SVL], z1_2 );    \
		svst1_f32( p0, &p_[3 * SVL], z1_3 );    \
		svst1_f32( p0, &p_[4 * SVL], z3_0 );    \
		svst1_f32( p0, &p_[5 * SVL], z3_1 );    \
		svst1_f32( p0, &p_[6 * SVL], z3_2 );    \
		svst1_f32( p0, &p_[7 * SVL], z3_3 );    \
		p_ += ( 8 * SVL );                      \
	}

//[TAIL VG2] Stores 4 Vectors
#define OP_TAIL_VG2_1( tcol, p_ )           \
	{                                       \
		svbool_t p0 = svptrue_b32();        \
		svfloat32_t z0_0 = svread_ver_za32_m( svundef_f32(), p0, 0, tcol + 0 ); \
		svfloat32_t z0_1 = svread_ver_za32_m( svundef_f32(), p0, 2, tcol + 0 ); \
		svfloat32_t z0_2 = svread_ver_za32_m( svundef_f32(), p0, 0, tcol + 1 ); \
		svfloat32_t z0_3 = svread_ver_za32_m( svundef_f32(), p0, 2, tcol + 1 ); \
		svst1_f32( p0, &p_[0], z0_0 );      \
		svst1_f32( p0, &p_[1 * SVL], z0_1 );\
		svst1_f32( p0, &p_[2 * SVL], z0_2 );\
		svst1_f32( p0, &p_[3 * SVL], z0_3 );\
		p_ += ( 4 * SVL );                  \
	}

#define OP_TAIL_VG2_2( tcol, p_ )           \
	{                                       \
		svbool_t p0 = svptrue_b32();        \
		svfloat32_t z1_0 = svread_ver_za32_m( svundef_f32(), p0, 1, tcol + 0 ); \
		svfloat32_t z1_1 = svread_ver_za32_m( svundef_f32(), p0, 3, tcol + 0 ); \
		svfloat32_t z1_2 = svread_ver_za32_m( svundef_f32(), p0, 1, tcol + 1 ); \
		svfloat32_t z1_3 = svread_ver_za32_m( svundef_f32(), p0, 3, tcol + 1 ); \
		svst1_f32( p0, &p_[0], z1_0 );      \
		svst1_f32( p0, &p_[1 * SVL], z1_1 );\
		svst1_f32( p0, &p_[2 * SVL], z1_2 );\
		svst1_f32( p0, &p_[3 * SVL], z1_3 );\
		p_ += ( 4 * SVL );                  \
	}

// [TAIL] Stores 2 Vectors
#define OP_TAIL_1( tcol, p_ )           \
	{                                   \
		svbool_t p0 = svptrue_b32();    \
		svfloat32_t z0 = svread_ver_za32_m( svundef_f32(), p0, 0, tcol ); \
		svfloat32_t z2 = svread_ver_za32_m( svundef_f32(), p0, 2, tcol ); \
		svst1_f32( p0, &p_[0], z0 );    \
		svst1_f32( p0, &p_[SVL], z2 );  \
		p_ += ( 2 * SVL );              \
	}

#define OP_TAIL_2( tcol, p_ )           \
	{                                   \
		svbool_t p0 = svptrue_b32();    \
		svfloat32_t z1 = svread_ver_za32_m( svundef_f32(), p0, 1, tcol ); \
		svfloat32_t z3 = svread_ver_za32_m( svundef_f32(), p0, 3, tcol ); \
		svst1_f32( p0, &p_[0], z1 );    \
		svst1_f32( p0, &p_[SVL], z3 );  \
		p_ += ( 2 * SVL );              \
	}

__arm_new( "za" ) __arm_locally_streaming void bli_spackm_m4sme_int_2SVLx2SVL
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

	float* restrict a_ = (float*)a;
	float* restrict p_ = (float*)p;

	uint64_t SVL = svcntsw();

	const float* restrict alpha1 = a;
	float* restrict pi1 = p;

	const bool gs = ( inca != 1 && lda != 1 );

	if ( cdim_bcast == 1 && !gs )
	{
		if ( bli_seq1( *( (float*)kappa ) ) )
		{
			if ( inca == 1 )
			// continous memory.packA style
			{
				svbool_t p0 = svwhilelt_b32( (int64_t)0, cdim );
				svbool_t p1 = svwhilelt_b32( (int64_t)SVL, cdim );

				for ( dim_t k = n; k != 0; --k )
				{
					svfloat32_t z0 = svld1_f32( p0, alpha1 + 0 * SVL );
					svfloat32_t z1 = svld1_f32( p1, alpha1 + 1 * SVL );

					svst1_f32( svptrue_b32(), pi1, z0 );
					svst1_f32( svptrue_b32(), pi1 + SVL, z1 );

					alpha1 += lda;
					pi1 += ldp;
				}
			}
			else
			{
				{
					for ( uint64_t col = 0; col < n; col += 2 * SVL )
					{
						int64_t valid_cols = n - col;

						// Determine total valid rows for this vertical block
						// (max 2 * SVL)
						int64_t valid_rows = ( cdim % ( 2 * SVL ) == 0 ) ?
							( 2 * SVL ) :
							( cdim % ( 2 * SVL ) );

						// Generate the 2 standard SVE column predicates for the
						// pairs of left and right tiles
						svbool_t pc0 = svwhilelt_b32( (int64_t)( 0 * SVL ),
							valid_cols );
						svbool_t pc1 = svwhilelt_b32( (int64_t)( 1 * SVL ),
							valid_cols );

						svbool_t p_all = svptrue_b32();

						if ( valid_cols >= 2 * SVL && valid_rows >= 2 * SVL )
						{
							// FAST PATH: Perfect 2*SVL x 2*SVL block
							for ( uint64_t trow = 0; trow < SVL; trow += 4 )
							{
								const uint64_t tile_UL_corner = (trow)*inca +
									col;

								// Group 1 (Tiles 0 and 1)

								svfloat32_t zp0 = svld1_f32(p_all, &a_[tile_UL_corner + 0 * inca]);
								svfloat32_t zp01 = svld1_f32(p_all, &a_[tile_UL_corner + 0 * inca + SVL]);
								svfloat32_t zp1 = svld1_f32(p_all, &a_[tile_UL_corner + 1 * inca]);
								svfloat32_t zp11 = svld1_f32(p_all, &a_[tile_UL_corner + 1 * inca + SVL]);
								svfloat32_t zp2 = svld1_f32(p_all, &a_[tile_UL_corner + 2 * inca]);
								svfloat32_t zp21 = svld1_f32(p_all, &a_[tile_UL_corner + 2 * inca + SVL]);
								svfloat32_t zp3 = svld1_f32(p_all, &a_[tile_UL_corner + 3 * inca]);
								svfloat32_t zp31 = svld1_f32(p_all, &a_[tile_UL_corner + 3 * inca +  SVL]);

								const uint64_t tile_BL_corner = tile_UL_corner +
									inca * SVL;

								// Group 1 (Tiles 2 and 3)

								svfloat32_t zp4 = svld1_f32(p_all, &a_[tile_BL_corner + 0 * inca]);
								svfloat32_t zp41 = svld1_f32(p_all, &a_[tile_BL_corner + 0 * inca + SVL]);
								svfloat32_t zp5 = svld1_f32(p_all, &a_[tile_BL_corner + 1 * inca]);
								svfloat32_t zp51 = svld1_f32(p_all, &a_[tile_BL_corner + 1 * inca + SVL]);
								svfloat32_t zp6 = svld1_f32(p_all, &a_[tile_BL_corner + 2 * inca]);
								svfloat32_t zp61 = svld1_f32(p_all, &a_[tile_BL_corner + 2 * inca + SVL]);
								svfloat32_t zp7 = svld1_f32(p_all, &a_[tile_BL_corner + 3 * inca]);
								svfloat32_t zp71 = svld1_f32(p_all, &a_[tile_BL_corner + 3 * inca +  SVL]);

								// ZA writes
								svwrite_hor_za32_f32_m(0, trow, p_all, zp0);
								svwrite_hor_za32_f32_m(0, trow + 1, p_all, zp1);
								svwrite_hor_za32_f32_m(0, trow + 2, p_all, zp2);
								svwrite_hor_za32_f32_m(0, trow + 3, p_all, zp3);
								svwrite_hor_za32_f32_m(1, trow, p_all, zp01);
								svwrite_hor_za32_f32_m(1, trow + 1, p_all, zp11);
								svwrite_hor_za32_f32_m(1, trow + 2, p_all, zp21);
								svwrite_hor_za32_f32_m(1, trow + 3, p_all, zp31);
								svwrite_hor_za32_f32_m(2, trow, p_all, zp4);
								svwrite_hor_za32_f32_m(2, trow + 1, p_all, zp5);
								svwrite_hor_za32_f32_m(2, trow + 2, p_all, zp6);
								svwrite_hor_za32_f32_m(2, trow + 3, p_all, zp7);
								svwrite_hor_za32_f32_m(3, trow, p_all, zp41);
								svwrite_hor_za32_f32_m(3, trow + 1, p_all, zp51);
								svwrite_hor_za32_f32_m(3, trow + 2, p_all, zp61);
								svwrite_hor_za32_f32_m(3, trow + 3, p_all, zp71);
							}
						}
						else
						{
							// SAFE PATH: Matrix edge
							for ( uint64_t trow = 0; trow < SVL; trow += 4 )
							{
								const uint64_t tile_UL_corner = (trow)*inca +
									col;
								const uint64_t tile_BL_corner = tile_UL_corner +
									inca * SVL;

								// 1. Create undefined default vectors
								svfloat32_t undef_v = svundef_f32();

								// 2. Default all load arrays to empty
								svfloat32_t v0_0 = undef_v, v0_1 = undef_v;
								svfloat32_t v1_0 = undef_v, v1_1 = undef_v;
								svfloat32_t v2_0 = undef_v, v2_1 = undef_v;
								svfloat32_t v3_0 = undef_v, v3_1 = undef_v;
								svfloat32_t v4_0 = undef_v, v4_1 = undef_v;
								svfloat32_t v5_0 = undef_v, v5_1 = undef_v;
								svfloat32_t v6_0 = undef_v, v6_1 = undef_v;
								svfloat32_t v7_0 = undef_v, v7_1 = undef_v;

								// 3. Calculate rows left independently for the
								// top and bottom block
								int64_t rows_left_top = valid_rows - trow;
								int64_t rows_left_bot = valid_rows -
									( SVL + trow );

								// 4. Load top rows (writes to tiles 0,1)
								if ( rows_left_top > 0 )
								{
									v0_0 = svld1_f32( pc0,
										&a_[tile_UL_corner + 0 * inca] );
									v0_1 = svld1_f32( pc1,
										&a_[tile_UL_corner + 0 * inca +
											SVL] );
								}
								if ( rows_left_top > 1 )
								{
									v1_0 = svld1_f32( pc0,
										&a_[tile_UL_corner + 1 * inca] );
									v1_1 = svld1_f32( pc1,
										&a_[tile_UL_corner + 1 * inca +
											SVL] );
								}
								if ( rows_left_top > 2 )
								{
									v2_0 = svld1_f32( pc0,
										&a_[tile_UL_corner + 2 * inca] );
									v2_1 = svld1_f32( pc1,
										&a_[tile_UL_corner + 2 * inca +
											SVL] );
								}
								if ( rows_left_top > 3 )
								{
									v3_0 = svld1_f32( pc0,
										&a_[tile_UL_corner + 3 * inca] );
									v3_1 = svld1_f32( pc1,
										&a_[tile_UL_corner + 3 * inca +
											SVL] );
								}

								// 5. Load bottom rows (writes to tiles 2, 3)
								if ( rows_left_bot > 0 )
								{
									v4_0 = svld1_f32( pc0,
										&a_[tile_BL_corner + 0 * inca] );
									v4_1 = svld1_f32( pc1,
										&a_[tile_BL_corner + 0 * inca +
											SVL] );
								}
								if ( rows_left_bot > 1 )
								{
									v5_0 = svld1_f32( pc0,
										&a_[tile_BL_corner + 1 * inca] );
									v5_1 = svld1_f32( pc1,
										&a_[tile_BL_corner + 1 * inca +
											SVL] );
								}
								if ( rows_left_bot > 2 )
								{
									v6_0 = svld1_f32( pc0,
										&a_[tile_BL_corner + 2 * inca] );
									v6_1 = svld1_f32( pc1,
										&a_[tile_BL_corner + 2 * inca +
											SVL] );
								}
								if ( rows_left_bot > 3 )
								{
									v7_0 = svld1_f32( pc0,
										&a_[tile_BL_corner + 3 * inca] );
									v7_1 = svld1_f32( pc1,
										&a_[tile_BL_corner + 3 * inca +
											SVL] );
								}

								// 6. Write into ZA
								svwrite_hor_za32_f32_m( 0, trow + 0, p_all, v0_0 );
								svwrite_hor_za32_f32_m( 0, trow + 1, p_all, v1_0 );
								svwrite_hor_za32_f32_m( 0, trow + 2, p_all, v2_0 );
								svwrite_hor_za32_f32_m( 0, trow + 3, p_all, v3_0 );

								svwrite_hor_za32_f32_m( 1, trow + 0, p_all, v0_1 );
								svwrite_hor_za32_f32_m( 1, trow + 1, p_all, v1_1 );
								svwrite_hor_za32_f32_m( 1, trow + 2, p_all, v2_1 );
								svwrite_hor_za32_f32_m( 1, trow + 3, p_all, v3_1 );

								svwrite_hor_za32_f32_m( 2, trow + 0, p_all, v4_0 );
								svwrite_hor_za32_f32_m( 2, trow + 1, p_all, v5_0 );
								svwrite_hor_za32_f32_m( 2, trow + 2, p_all, v6_0 );
								svwrite_hor_za32_f32_m( 2, trow + 3, p_all, v7_0 );

								svwrite_hor_za32_f32_m( 3, trow + 0, p_all, v4_1 );
								svwrite_hor_za32_f32_m( 3, trow + 1, p_all, v5_1 );
								svwrite_hor_za32_f32_m( 3, trow + 2, p_all, v6_1 );
								svwrite_hor_za32_f32_m( 3, trow + 3, p_all, v7_1 );
							}
						}
						// Check if we are at the edge and fewer than
						// 2 * SVL columns remain
						if ( col + ( 2 * SVL ) > n )
						{
							// Total columns left to process in this tail.
							// Range: [1, 2*SVL - 1]
							int total_rem = n - col;

							// Split total_rem into columns for Tile Pair 0/2
							// (rem1) and 1/3 (rem2) Each vertical tile pair has
							// a width of SVL.
							int rem1 = ( total_rem > (int)SVL ) ? (int)SVL :
																  total_rem;
							int rem2 = ( total_rem > (int)SVL ) ?
								( total_rem - (int)SVL ) :
								0;

							// PART 1: Process Tiles 0 & 2
							if ( rem1 > 0 )
							{
								int tcol = 0;
								int n4 = rem1 >> 2;

								if ( n4 > 0 )
								{
									int i = ( n4 + 3 ) >> 2;
									// Duff's Device unrolling VG4 operations
									switch ( n4 & 3 )
									{
									case 0:
										do
										{
											OP_VG4_1( tcol, p_ );
											tcol += 4;
										case 3:
											OP_VG4_1( tcol, p_ );
											tcol += 4;
										case 2:
											OP_VG4_1( tcol, p_ );
											tcol += 4;
										case 1:
											OP_VG4_1( tcol, p_ );
											tcol += 4;
										} while ( --i > 0 );
									}
								}

								// Handle remaining 1, 2, or 3 columns
								switch ( rem1 & 3 )
								{
								case 3:
									OP_TAIL_VG2_1( tcol, p_ );
									tcol += 2;
									OP_TAIL_1( tcol, p_ );
									break;
								case 2:
									OP_TAIL_VG2_1( tcol, p_ );
									break;
								case 1:
									OP_TAIL_1( tcol, p_ );
									break;
								default:
									break;
								}
							}

							// PART 2: Process Tiles 1 & 3
							if ( rem2 > 0 )
							{
								int tcol = 0;
								int n4 = rem2 >> 2;

								if ( n4 > 0 )
								{
									int i = ( n4 + 3 ) >> 2;
									// Duff's Device unrolling VG4 operations
									switch ( n4 & 3 )
									{
									case 0:
										do
										{
											OP_VG4_2( tcol, p_ );
											tcol += 4;
										case 3:
											OP_VG4_2( tcol, p_ );
											tcol += 4;
										case 2:
											OP_VG4_2( tcol, p_ );
											tcol += 4;
										case 1:
											OP_VG4_2( tcol, p_ );
											tcol += 4;
										} while ( --i > 0 );
									}
								}

								// Handle remaining 1, 2, or 3 columns
								switch ( rem2 & 3 )
								{
								case 3:
									OP_TAIL_VG2_2( tcol, p_ );
									tcol += 2;
									OP_TAIL_2( tcol, p_ );
									break;
								case 2:
									OP_TAIL_VG2_2( tcol, p_ );
									break;
								case 1:
									OP_TAIL_2( tcol, p_ );
									break;
								default:
									break;
								}
							}
						}

						else
						{
							// Read - as - columns and store
							for ( uint64_t tcol = 0; tcol < SVL; tcol += 4 )
							{
								svbool_t p0 = svptrue_b32();

								// Tiles 0 and 2
								svfloat32_t z0_0 = svread_ver_za32_m( svundef_f32(), p0, 0, tcol + 0 );
								svfloat32_t z0_1 = svread_ver_za32_m( svundef_f32(), p0, 2, tcol + 0 );
								svfloat32_t z0_2 = svread_ver_za32_m( svundef_f32(), p0, 0, tcol + 1 );
								svfloat32_t z0_3 = svread_ver_za32_m( svundef_f32(), p0, 2, tcol + 1 );
								
								svfloat32_t z0_4 = svread_ver_za32_m( svundef_f32(), p0, 0, tcol + 2 );
								svfloat32_t z0_5 = svread_ver_za32_m( svundef_f32(), p0, 2, tcol + 2 );
								svfloat32_t z0_6 = svread_ver_za32_m( svundef_f32(), p0, 0, tcol + 3 );
								svfloat32_t z0_7 = svread_ver_za32_m( svundef_f32(), p0, 2, tcol + 3 );

								// Tiles 1 and 3
								svfloat32_t z1_0 = svread_ver_za32_m( svundef_f32(), p0, 1, tcol + 0 );
								svfloat32_t z1_1 = svread_ver_za32_m( svundef_f32(), p0, 3, tcol + 0 );
								svfloat32_t z1_2 = svread_ver_za32_m( svundef_f32(), p0, 1, tcol + 1 );
								svfloat32_t z1_3 = svread_ver_za32_m( svundef_f32(), p0, 3, tcol + 1 );
								
								svfloat32_t z1_4 = svread_ver_za32_m( svundef_f32(), p0, 1, tcol + 2 );
								svfloat32_t z1_5 = svread_ver_za32_m( svundef_f32(), p0, 3, tcol + 2 );
								svfloat32_t z1_6 = svread_ver_za32_m( svundef_f32(), p0, 1, tcol + 3 );
								svfloat32_t z1_7 = svread_ver_za32_m( svundef_f32(), p0, 3, tcol + 3 );

								svst1_f32( p0, &p_[0], z0_0 );
								svst1_f32( p0, &p_[1 * SVL], z0_1 );
								svst1_f32( p0, &p_[2 * SVL], z0_2 );
								svst1_f32( p0, &p_[3 * SVL], z0_3 );
								svst1_f32( p0, &p_[4 * SVL], z0_4 );
								svst1_f32( p0, &p_[5 * SVL], z0_5 );
								svst1_f32( p0, &p_[6 * SVL], z0_6 );
								svst1_f32( p0, &p_[7 * SVL], z0_7 );

								svst1_f32( p0, &p_[2 * SVL * SVL + 0], z1_0 );
								svst1_f32( p0, &p_[2 * SVL * SVL + 1 * SVL], z1_1 );
								svst1_f32( p0, &p_[2 * SVL * SVL + 2 * SVL], z1_2 );
								svst1_f32( p0, &p_[2 * SVL * SVL + 3 * SVL], z1_3 );
								svst1_f32( p0, &p_[2 * SVL * SVL + 4 * SVL], z1_4 );
								svst1_f32( p0, &p_[2 * SVL * SVL + 5 * SVL], z1_5 );
								svst1_f32( p0, &p_[2 * SVL * SVL + 6 * SVL], z1_6 );
								svst1_f32( p0, &p_[2 * SVL * SVL + 7 * SVL], z1_7 );

								p_ += ( 8 * SVL );
							}
							p_ += ( 2 * SVL * SVL );
						}
					}
				}

				p_ = (float*)p;
			}
		}
		else
		{
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
	}
	else
	{
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