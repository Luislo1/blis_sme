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

#include <arm_acle.h>
#include <arm_sme.h>
#include "blis.h"

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

#include <arm_acle.h>
#include <arm_sme.h>
#include "blis.h"

__arm_new( "za" ) __arm_locally_streaming void bli_sgemm_m4sme_int_2SVLx2SVL
	(
			  dim_t      m,
			  dim_t      n,
			  dim_t      k,
		const void*      alpha,
		const void*      a,
		const void*      b,
		const void*      beta,
			  void*		 c, inc_t rs_c, inc_t cs_c,
		const auxinfo_t* data,
		const cntx_t*    cntx
	) 
{
	uint64_t SVL = svcntsw();

	GEMM_UKR_SETUP_CT_AMBI( s, 2 * SVL, 2 * SVL, false );

	float *a_ = (float *)a;
	float *b_ = (float *)b;

	float *a_next = (float *)bli_auxinfo_next_a( data );
	float *b_next = (float *)bli_auxinfo_next_b( data );

	float *c_ = (float *)c;

	const uint64_t result_tile_TL_corner_ = 0;
	const uint64_t result_tile_TR_corner_ = result_tile_TL_corner_ + SVL;

	if ( cs_c != 1 )
	{
		__pldx( 1, 1, 0,
			(float *)&c_[result_tile_TL_corner_ + ( ( ( 0 + 0 ) * cs_c ) )] );
		__pldx( 1, 1, 0,
			(float *)&c_[result_tile_TL_corner_ + ( ( ( 1 + 0 ) * cs_c ) )] );
		__pldx( 1, 1, 0,
			(float *)&c_[result_tile_TL_corner_ + ( ( ( 2 + 0 ) * cs_c ) )] );
		__pldx( 1, 1, 0,
			(float *)&c_[result_tile_TL_corner_ + ( ( ( 3 + 0 ) * cs_c ) )] );

		__pldx( 1, 1, 0,
			(float *)&c_[result_tile_TL_corner_ + ( ( ( 4 + 0 ) * cs_c ) )] );
		__pldx( 1, 1, 0,
			(float *)&c_[result_tile_TL_corner_ + ( ( ( 5 + 0 ) * cs_c ) )] );
		__pldx( 1, 1, 0,
			(float *)&c_[result_tile_TL_corner_ + ( ( ( 6 + 0 ) * cs_c ) )] );
		__pldx( 1, 1, 0,
			(float *)&c_[result_tile_TL_corner_ + ( ( ( 7 + 0 ) * cs_c ) )] );

		__pldx( 1, 1, 0,
			(float *)&c_[result_tile_TR_corner_ + ( ( ( 0 + 0 ) * cs_c ) )] );
		__pldx( 1, 1, 0,
			(float *)&c_[result_tile_TR_corner_ + ( ( ( 1 + 0 ) * cs_c ) )] );
		__pldx( 1, 1, 0,
			(float *)&c_[result_tile_TR_corner_ + ( ( ( 2 + 0 ) * cs_c ) )] );
		__pldx( 1, 1, 0,
			(float *)&c_[result_tile_TR_corner_ + ( ( ( 3 + 0 ) * cs_c ) )] );

		__pldx( 1, 1, 0,
			(float *)&c_[result_tile_TR_corner_ + ( ( ( 4 + 0 ) * cs_c ) )] );
		__pldx( 1, 1, 0,
			(float *)&c_[result_tile_TR_corner_ + ( ( ( 5 + 0 ) * cs_c ) )] );
		__pldx( 1, 1, 0,
			(float *)&c_[result_tile_TR_corner_ + ( ( ( 6 + 0 ) * cs_c ) )] );
		__pldx( 1, 1, 0,
			(float *)&c_[result_tile_TR_corner_ + ( ( ( 7 + 0 ) * cs_c ) )] );
	}
	else
	{
		__pldx( 1, 1, 0,
			(float *)&c_[result_tile_TL_corner_ + ( ( ( 0 + 0 ) * rs_c ) )] );
		__pldx( 1, 1, 0,
			(float *)&c_[result_tile_TL_corner_ + ( ( ( 1 + 0 ) * rs_c ) )] );
		__pldx( 1, 1, 0,
			(float *)&c_[result_tile_TL_corner_ + ( ( ( 2 + 0 ) * rs_c ) )] );
		__pldx( 1, 1, 0,
			(float *)&c_[result_tile_TL_corner_ + ( ( ( 3 + 0 ) * rs_c ) )] );

		__pldx( 1, 1, 0,
			(float *)&c_[result_tile_TL_corner_ + ( ( ( 4 + 0 ) * rs_c ) )] );
		__pldx( 1, 1, 0,
			(float *)&c_[result_tile_TL_corner_ + ( ( ( 5 + 0 ) * rs_c ) )] );
		__pldx( 1, 1, 0,
			(float *)&c_[result_tile_TL_corner_ + ( ( ( 6 + 0 ) * rs_c ) )] );
		__pldx( 1, 1, 0,
			(float *)&c_[result_tile_TL_corner_ + ( ( ( 7 + 0 ) * rs_c ) )] );

		__pldx( 1, 1, 0,
			(float *)&c_[result_tile_TR_corner_ + ( ( ( 0 + 0 ) * rs_c ) )] );
		__pldx( 1, 1, 0,
			(float *)&c_[result_tile_TR_corner_ + ( ( ( 1 + 0 ) * rs_c ) )] );
		__pldx( 1, 1, 0,
			(float *)&c_[result_tile_TR_corner_ + ( ( ( 2 + 0 ) * rs_c ) )] );
		__pldx( 1, 1, 0,
			(float *)&c_[result_tile_TR_corner_ + ( ( ( 3 + 0 ) * rs_c ) )] );

		__pldx( 1, 1, 0,
			(float *)&c_[result_tile_TR_corner_ + ( ( ( 4 + 0 ) * rs_c ) )] );
		__pldx( 1, 1, 0,
			(float *)&c_[result_tile_TR_corner_ + ( ( ( 5 + 0 ) * rs_c ) )] );
		__pldx( 1, 1, 0,
			(float *)&c_[result_tile_TR_corner_ + ( ( ( 6 + 0 ) * rs_c ) )] );
		__pldx( 1, 1, 0,
			(float *)&c_[result_tile_TR_corner_ + ( ( ( 7 + 0 ) * rs_c ) )] );
	}

	svzero_za();

	uint64_t k_;
	uint64_t k_iter = k / 8;
	uint64_t k_left = k % 8;

	for ( k_ = 0; k_ < k_iter; k_++ )
	{
		svfloat32_t zL00 = svld1(svptrue_b32(), (float32_t*)(&a_[0        ]));
		svfloat32_t zL01 = svld1(svptrue_b32(), (float32_t*)(&a_[(SVL)    ]));
		svfloat32_t zL02 = svld1(svptrue_b32(), (float32_t*)(&a_[2 * (SVL)]));
		svfloat32_t zL03 = svld1(svptrue_b32(), (float32_t*)(&a_[3 * (SVL)]));

		svfloat32_t zR00 = svld1(svptrue_b32(), (float32_t*)(&b_[0        ]));
		svfloat32_t zR01 = svld1(svptrue_b32(), (float32_t*)(&b_[(SVL)    ]));
		svfloat32_t zR02 = svld1(svptrue_b32(), (float32_t*)(&b_[2 * (SVL)]));
		svfloat32_t zR03 = svld1(svptrue_b32(), (float32_t*)(&b_[3 * (SVL)]));


		svmopa_za32_m( 0, svptrue_b32(), svptrue_b32(), zL00, zR00);
		svmopa_za32_m( 1, svptrue_b32(), svptrue_b32(), zL01, zR00);

		__pldx( 0, 1, 1, (float *)&a_next[0] );
		__pldx( 0, 1, 1, (float *)&b_next[0] );

		svmopa_za32_m( 2, svptrue_b32(), svptrue_b32(), zL00, zR01);
		svmopa_za32_m( 3, svptrue_b32(), svptrue_b32(), zL01, zR01);

		svmopa_za32_m( 0, svptrue_b32(), svptrue_b32(), zL02, zR02);
		svmopa_za32_m( 1, svptrue_b32(), svptrue_b32(), zL03, zR02);

		svmopa_za32_m( 2, svptrue_b32(), svptrue_b32(), zL02, zR03);
		svmopa_za32_m( 3, svptrue_b32(), svptrue_b32(), zL03, zR03);

		svfloat32_t zL04 = svld1(svptrue_b32(), (float32_t*)(&a_[4 * (SVL)]));
		svfloat32_t zL05 = svld1(svptrue_b32(), (float32_t*)(&a_[5 * (SVL)]));
		svfloat32_t zL06 = svld1(svptrue_b32(), (float32_t*)(&a_[6 * (SVL)]));
		svfloat32_t zL07 = svld1(svptrue_b32(), (float32_t*)(&a_[7 * (SVL)]));

		svfloat32_t zR04 = svld1(svptrue_b32(), (float32_t*)(&b_[4 * (SVL)]));
		svfloat32_t zR05 = svld1(svptrue_b32(), (float32_t*)(&b_[5 * (SVL)]));
		svfloat32_t zR06 = svld1(svptrue_b32(), (float32_t*)(&b_[6 * (SVL)]));
		svfloat32_t zR07 = svld1(svptrue_b32(), (float32_t*)(&b_[7 * (SVL)]));

		svmopa_za32_m( 0, svptrue_b32(), svptrue_b32(), zL04, zR04);
		svmopa_za32_m( 1, svptrue_b32(), svptrue_b32(), zL05, zR04);

		__pldx( 0, 1, 1, (float *)&a_next[4 * SVL] );
		__pldx( 0, 1, 1, (float *)&b_next[4 * SVL] );

		svmopa_za32_m( 2, svptrue_b32(), svptrue_b32(), zL04, zR05);
		svmopa_za32_m( 3, svptrue_b32(), svptrue_b32(), zL05, zR05);

		svmopa_za32_m( 0, svptrue_b32(), svptrue_b32(), zL06, zR06);
		svmopa_za32_m( 1, svptrue_b32(), svptrue_b32(), zL07, zR06);

		svmopa_za32_m( 2, svptrue_b32(), svptrue_b32(), zL06, zR07);
		svmopa_za32_m( 3, svptrue_b32(), svptrue_b32(), zL07, zR07);

		svfloat32_t zL08 = svld1(svptrue_b32(), (float32_t*)(&a_[8 * (SVL)]));
		svfloat32_t zL09 = svld1(svptrue_b32(), (float32_t*)(&a_[9 * (SVL)]));
		svfloat32_t zL10 = svld1(svptrue_b32(), (float32_t*)(&a_[10 * (SVL)]));
		svfloat32_t zL11 = svld1(svptrue_b32(), (float32_t*)(&a_[11 * (SVL)]));

		svfloat32_t zR08 = svld1(svptrue_b32(), (float32_t*)(&b_[8 * (SVL)]));
		svfloat32_t zR09 = svld1(svptrue_b32(), (float32_t*)(&b_[9 * (SVL)]));
		svfloat32_t zR10 = svld1(svptrue_b32(), (float32_t*)(&b_[10 * (SVL)]));
		svfloat32_t zR11 = svld1(svptrue_b32(), (float32_t*)(&b_[11 * (SVL)]));

		svmopa_za32_m( 0, svptrue_b32(), svptrue_b32(), zL08, zR08);
		svmopa_za32_m( 1, svptrue_b32(), svptrue_b32(), zL09, zR08);

		__pldx( 0, 1, 1, (float *)&a_next[8 * SVL] );
		__pldx( 0, 1, 1, (float *)&b_next[8 * SVL] );

		svmopa_za32_m( 2, svptrue_b32(), svptrue_b32(), zL08, zR09);
		svmopa_za32_m( 3, svptrue_b32(), svptrue_b32(), zL09, zR09);

		svmopa_za32_m( 0, svptrue_b32(), svptrue_b32(), zL10, zR10);
		svmopa_za32_m( 1, svptrue_b32(), svptrue_b32(), zL11, zR10);

		svmopa_za32_m( 2, svptrue_b32(), svptrue_b32(), zL10, zR11);
		svmopa_za32_m( 3, svptrue_b32(), svptrue_b32(), zL11, zR11);

		svfloat32_t zL12 = svld1(svptrue_b32(), (float32_t*)(&a_[12 * (SVL)]));
		svfloat32_t zL13 = svld1(svptrue_b32(), (float32_t*)(&a_[13 * (SVL)]));
		svfloat32_t zL14 = svld1(svptrue_b32(), (float32_t*)(&a_[14 * (SVL)]));
		svfloat32_t zL15 = svld1(svptrue_b32(), (float32_t*)(&a_[15 * (SVL)]));

		svfloat32_t zR12 = svld1(svptrue_b32(), (float32_t*)(&b_[12 * (SVL)]));
		svfloat32_t zR13 = svld1(svptrue_b32(), (float32_t*)(&b_[13 * (SVL)]));
		svfloat32_t zR14 = svld1(svptrue_b32(), (float32_t*)(&b_[14 * (SVL)]));
		svfloat32_t zR15 = svld1(svptrue_b32(), (float32_t*)(&b_[15 * (SVL)]));

		svmopa_za32_m( 0, svptrue_b32(), svptrue_b32(), zL12, zR12);
		svmopa_za32_m( 1, svptrue_b32(), svptrue_b32(), zL13, zR12);

		__pldx( 0, 1, 1, (float *)&a_next[12 * SVL] );
		__pldx( 0, 1, 1, (float *)&b_next[12 * SVL] );

		svmopa_za32_m( 2, svptrue_b32(), svptrue_b32(), zL12, zR13);
		svmopa_za32_m( 3, svptrue_b32(), svptrue_b32(), zL13, zR13);

		svmopa_za32_m( 0, svptrue_b32(), svptrue_b32(), zL14, zR14);
		svmopa_za32_m( 1, svptrue_b32(), svptrue_b32(), zL15, zR14);

		svmopa_za32_m( 2, svptrue_b32(), svptrue_b32(), zL14, zR15);
		svmopa_za32_m( 3, svptrue_b32(), svptrue_b32(), zL15, zR15);

		a_ += ( 2 * 8 * SVL );
		b_ += ( 2 * 8 * SVL );

		a_next += ( 2 * 8 * SVL );
		b_next += ( 2 * 8 * SVL );
	}

	for ( k_ = 0; k_ < k_left; k_ += 1 )
	{
		svfloat32_t zL00 = svld1(svptrue_b32(), (float32_t*)(&a_[0        ]));
		svfloat32_t zL01 = svld1(svptrue_b32(), (float32_t*)(&a_[(SVL)    ]));

		svfloat32_t zR00 = svld1(svptrue_b32(), (float32_t*)(&b_[0        ]));
		svfloat32_t zR01 = svld1(svptrue_b32(), (float32_t*)(&b_[(SVL)    ]));


		svmopa_za32_m( 0, svptrue_b32(), svptrue_b32(), zL00, zR00);
		svmopa_za32_m( 1, svptrue_b32(), svptrue_b32(), zL01, zR00);

		svmopa_za32_m( 2, svptrue_b32(), svptrue_b32(), zL00, zR01);
		svmopa_za32_m( 3, svptrue_b32(), svptrue_b32(), zL01, zR01);


		a_ += ( 2 * SVL );
		b_ += ( 2 * SVL );
	}

	// Store ZA to matResult.

	const uint64_t result_tile_TL_corner = 0;

	float beta_ = *(float *)beta;
	float alpha_ = *(float *)alpha;

	svfloat32_t zbeta = svdup_f32( beta_ );
	svfloat32_t zalpha = svdup_f32( alpha_ );

	if ( rs_c == 1 )
	{
	const uint64_t result_tile_BL_corner = SVL;
	const uint64_t result_tile_TR_corner = SVL * cs_c;
	const uint64_t result_tile_BR_corner = SVL * cs_c + SVL;
		if ( beta_ == 0 )
		{
			for ( uint64_t tcol = 0; tcol < SVL; tcol += 4 )
			{
				// Read ZA slices into Z regs
				svfloat32_t z0 = svread_ver_za32_m( z0, svptrue_b32(),
					/* tile: */ 0, /* slice: */ tcol + 0 );
				svfloat32_t z1 = svread_ver_za32_m( z1, svptrue_b32(),
					/* tile: */ 1, /* slice: */ tcol + 0 );
				svfloat32_t z2 = svread_ver_za32_m( z2, svptrue_b32(),
					/* tile: */ 2, /* slice: */ tcol + 0 );
				svfloat32_t z3 = svread_ver_za32_m( z3, svptrue_b32(),
					/* tile: */ 3, /* slice: */ tcol + 0 );

				// Scale Z regs by broadcast alpha
				z0 = svmul_f32_z( svptrue_b32(), z0, zalpha );
				z1 = svmul_f32_z( svptrue_b32(), z1, zalpha );
				z2 = svmul_f32_z( svptrue_b32(), z2, zalpha );
				z3 = svmul_f32_z( svptrue_b32(), z3, zalpha );

				// Store full result into C
				svst1( svptrue_b32(),
					&c_[result_tile_TL_corner + ( tcol + 0 ) * cs_c], z0 );
				svst1( svptrue_b32(),
					&c_[result_tile_BL_corner + ( tcol + 0 ) * cs_c], z1 );
				svst1( svptrue_b32(),
					&c_[result_tile_TR_corner + ( tcol + 0 ) * cs_c], z2 );
				svst1( svptrue_b32(),
					&c_[result_tile_BR_corner + ( tcol + 0 ) * cs_c], z3 );

				// Repeat unfolded x4
				z0 = svread_ver_za32_m( z0, svptrue_b32(),
					/* tile: */ 0, /* slice: */ tcol + 1 );
				z1 = svread_ver_za32_m( z1, svptrue_b32(),
					/* tile: */ 1, /* slice: */ tcol + 1 );
				z2 = svread_ver_za32_m( z2, svptrue_b32(),
					/* tile: */ 2, /* slice: */ tcol + 1 );
				z3 = svread_ver_za32_m( z3, svptrue_b32(),
					/* tile: */ 3, /* slice: */ tcol + 1 );

				z0 = svmul_f32_z( svptrue_b32(), z0, zalpha );
				z1 = svmul_f32_z( svptrue_b32(), z1, zalpha );
				z2 = svmul_f32_z( svptrue_b32(), z2, zalpha );
				z3 = svmul_f32_z( svptrue_b32(), z3, zalpha );

				svst1( svptrue_b32(),
					&c_[result_tile_TL_corner + ( tcol + 1 ) * cs_c], z0 );
				svst1( svptrue_b32(),
					&c_[result_tile_BL_corner + ( tcol + 1 ) * cs_c], z1 );
				svst1( svptrue_b32(),
					&c_[result_tile_TR_corner + ( tcol + 1 ) * cs_c], z2 );
				svst1( svptrue_b32(),
					&c_[result_tile_BR_corner + ( tcol + 1 ) * cs_c], z3 );

				z0 = svread_ver_za32_m( z0, svptrue_b32(),
					/* tile: */ 0, /* slice: */ tcol + 2 );
				z1 = svread_ver_za32_m( z1, svptrue_b32(),
					/* tile: */ 1, /* slice: */ tcol + 2 );
				z2 = svread_ver_za32_m( z2, svptrue_b32(),
					/* tile: */ 2, /* slice: */ tcol + 2 );
				z3 = svread_ver_za32_m( z3, svptrue_b32(),
					/* tile: */ 3, /* slice: */ tcol + 2 );

				z0 = svmul_f32_z( svptrue_b32(), z0, zalpha );
				z1 = svmul_f32_z( svptrue_b32(), z1, zalpha );
				z2 = svmul_f32_z( svptrue_b32(), z2, zalpha );
				z3 = svmul_f32_z( svptrue_b32(), z3, zalpha );

				svst1( svptrue_b32(),
					&c_[result_tile_TL_corner + ( tcol + 2 ) * cs_c], z0 );
				svst1( svptrue_b32(),
					&c_[result_tile_BL_corner + ( tcol + 2 ) * cs_c], z1 );
				svst1( svptrue_b32(),
					&c_[result_tile_TR_corner + ( tcol + 2 ) * cs_c], z2 );
				svst1( svptrue_b32(),
					&c_[result_tile_BR_corner + ( tcol + 2 ) * cs_c], z3 );

				z0 = svread_ver_za32_m( z0, svptrue_b32(),
					/* tile: */ 0, /* slice: */ tcol + 3 );
				z1 = svread_ver_za32_m( z1, svptrue_b32(),
					/* tile: */ 1, /* slice: */ tcol + 3 );
				z2 = svread_ver_za32_m( z2, svptrue_b32(),
					/* tile: */ 2, /* slice: */ tcol + 3 );
				z3 = svread_ver_za32_m( z3, svptrue_b32(),
					/* tile: */ 3, /* slice: */ tcol + 3 );

				z0 = svmul_f32_z( svptrue_b32(), z0, zalpha );
				z1 = svmul_f32_z( svptrue_b32(), z1, zalpha );
				z2 = svmul_f32_z( svptrue_b32(), z2, zalpha );
				z3 = svmul_f32_z( svptrue_b32(), z3, zalpha );

				svst1( svptrue_b32(),
					&c_[result_tile_TL_corner + ( tcol + 3 ) * cs_c], z0 );
				svst1( svptrue_b32(),
					&c_[result_tile_BL_corner + ( tcol + 3 ) * cs_c], z1 );
				svst1( svptrue_b32(),
					&c_[result_tile_TR_corner + ( tcol + 3 ) * cs_c], z2 );
				svst1( svptrue_b32(),
					&c_[result_tile_BR_corner + ( tcol + 3 ) * cs_c], z3 );
			}
		}

		// beta != 0
		else
		{
			for ( uint64_t tcol = 0; tcol < SVL; tcol += 4 )
			{
				// Read ZA slices into Z regs
				svfloat32_t z0 = svread_ver_za32_m( z0, svptrue_b32(),
					/* tile: */ 0, /* slice: */ tcol + 0 );
				svfloat32_t z1 = svread_ver_za32_m( z1, svptrue_b32(),
					/* tile: */ 1, /* slice: */ tcol + 0 );
				svfloat32_t z2 = svread_ver_za32_m( z2, svptrue_b32(),
					/* tile: */ 2, /* slice: */ tcol + 0 );
				svfloat32_t z3 = svread_ver_za32_m( z3, svptrue_b32(),
					/* tile: */ 3, /* slice: */ tcol + 0 );

				// Scale Z regs by broadcast alpha
				svfloat32_t z00 = svmul_f32_z( svptrue_b32(), z0, zalpha );
				svfloat32_t z10 = svmul_f32_z( svptrue_b32(), z1, zalpha );
				svfloat32_t z20 = svmul_f32_z( svptrue_b32(), z2, zalpha );
				svfloat32_t z30 = svmul_f32_z( svptrue_b32(), z3, zalpha );

				// Load C into Z regs
				svfloat32_t zq5 = svld1_f32( svptrue_b32(),
					&c_[result_tile_TL_corner + ( tcol + 0 ) * cs_c] );
				svfloat32_t zq6 = svld1_f32( svptrue_b32(),
					&c_[result_tile_BL_corner + ( tcol + 0 ) * cs_c] );
				svfloat32_t zq7 = svld1_f32( svptrue_b32(),
					&c_[result_tile_TR_corner + ( tcol + 0 ) * cs_c] );
				svfloat32_t zq8 = svld1_f32( svptrue_b32(),
					&c_[result_tile_BR_corner + ( tcol + 0 ) * cs_c] );

				// Scale Z regs by broadcast beta
				svfloat32_t z40 = svmla_m( svptrue_b32(), z00, zq5, zbeta );
				svfloat32_t z50 = svmla_m( svptrue_b32(), z10, zq6, zbeta );
				svfloat32_t z60 = svmla_m( svptrue_b32(), z20, zq7, zbeta );
				svfloat32_t z70 = svmla_m( svptrue_b32(), z30, zq8,	zbeta );

				// Store full result into C
				svst1( svptrue_b32(),
					&c_[result_tile_TL_corner + ( tcol + 0 ) * cs_c], z40 );
				svst1( svptrue_b32(),
					&c_[result_tile_BL_corner + ( tcol + 0 ) * cs_c], z50 );
				svst1( svptrue_b32(),
					&c_[result_tile_TR_corner + ( tcol + 0 ) * cs_c], z60 );
				svst1( svptrue_b32(),
					&c_[result_tile_BR_corner + ( tcol + 0 ) * cs_c], z70 );

				// Repeat unfolded x4
				svfloat32_t z01 = svread_ver_za32_m( z01, svptrue_b32(),
					/* tile: */ 0, /* slice: */ tcol + 1 );
				svfloat32_t z11 = svread_ver_za32_m( z11, svptrue_b32(),
					/* tile: */ 1, /* slice: */ tcol + 1 );
				svfloat32_t z21 = svread_ver_za32_m( z21, svptrue_b32(),
					/* tile: */ 2, /* slice: */ tcol + 1 );
				svfloat32_t z31 = svread_ver_za32_m( z31, svptrue_b32(),
					/* tile: */ 3, /* slice: */ tcol + 1 );

				svfloat32_t z02 = svmul_f32_z( svptrue_b32(), z01, zalpha );
				svfloat32_t z12 = svmul_f32_z( svptrue_b32(), z11, zalpha );
				svfloat32_t z22 = svmul_f32_z( svptrue_b32(), z21, zalpha );
				svfloat32_t z32 = svmul_f32_z( svptrue_b32(), z31, zalpha );

				svfloat32_t zq51 = svld1_f32( svptrue_b32(),
					&c_[result_tile_TL_corner + ( tcol + 1 ) * cs_c] );
				svfloat32_t zq61 = svld1_f32( svptrue_b32(),
					&c_[result_tile_BL_corner + ( tcol + 1 ) * cs_c] );
				svfloat32_t zq71 = svld1_f32( svptrue_b32(),
					&c_[result_tile_TR_corner + ( tcol + 1 ) * cs_c] );
				svfloat32_t zq81 = svld1_f32( svptrue_b32(),
					&c_[result_tile_BR_corner + ( tcol + 1 ) * cs_c] );

				svfloat32_t z401 = svmla_m( svptrue_b32(), z02, zq51, zbeta );
				svfloat32_t z501 = svmla_m( svptrue_b32(), z12, zq61, zbeta );
				svfloat32_t z601 = svmla_m( svptrue_b32(), z22, zq71, zbeta );
				svfloat32_t z701 = svmla_m( svptrue_b32(), z32, zq81, zbeta );

				svst1( svptrue_b32(),
					&c_[result_tile_TL_corner + ( tcol + 1 ) * cs_c], z401 );
				svst1( svptrue_b32(),
					&c_[result_tile_BL_corner + ( tcol + 1 ) * cs_c], z501 );
				svst1( svptrue_b32(),
					&c_[result_tile_TR_corner + ( tcol + 1 ) * cs_c], z601 );
				svst1( svptrue_b32(),
					&c_[result_tile_BR_corner + ( tcol + 1 ) * cs_c], z701 );

				svfloat32_t z03 = svread_ver_za32_m( z03, svptrue_b32(),
					/* tile: */ 0, /* slice: */ tcol + 2 );
				svfloat32_t z13 = svread_ver_za32_m( z13, svptrue_b32(),
					/* tile: */ 1, /* slice: */ tcol + 2 );
				svfloat32_t z23 = svread_ver_za32_m( z23, svptrue_b32(),
					/* tile: */ 2, /* slice: */ tcol + 2 );
				svfloat32_t z33 = svread_ver_za32_m( z33, svptrue_b32(),
					/* tile: */ 3, /* slice: */ tcol + 2 );

				svfloat32_t z04 = svmul_f32_z( svptrue_b32(), z03, zalpha );
				svfloat32_t z14 = svmul_f32_z( svptrue_b32(), z13, zalpha );
				svfloat32_t z24 = svmul_f32_z( svptrue_b32(), z23, zalpha );
				svfloat32_t z34 = svmul_f32_z( svptrue_b32(), z33, zalpha );
				
				svfloat32_t zq52 = svld1_f32( svptrue_b32(),
					&c_[result_tile_TL_corner + ( tcol + 2 ) * cs_c] );
				svfloat32_t zq62 = svld1_f32( svptrue_b32(),
					&c_[result_tile_BL_corner + ( tcol + 2 ) * cs_c] );
				svfloat32_t zq72 = svld1_f32( svptrue_b32(),
					&c_[result_tile_TR_corner + ( tcol + 2 ) * cs_c] );
				svfloat32_t zq82 = svld1_f32( svptrue_b32(),
					&c_[result_tile_BR_corner + ( tcol + 2 ) * cs_c] );

				// Scale Z regs by broadcast beta
				svfloat32_t z402 = svmla_m( svptrue_b32(), z04, zq52, zbeta );
				svfloat32_t z502 = svmla_m( svptrue_b32(), z14, zq62, zbeta );
				svfloat32_t z602 = svmla_m( svptrue_b32(), z24, zq72, zbeta );
				svfloat32_t z702 = svmla_m( svptrue_b32(), z34, zq82, zbeta );

				svst1( svptrue_b32(),
					&c_[result_tile_TL_corner + ( tcol + 2 ) * cs_c], z402 );
				svst1( svptrue_b32(),
					&c_[result_tile_BL_corner + ( tcol + 2 ) * cs_c], z502 );
				svst1( svptrue_b32(),
					&c_[result_tile_TR_corner + ( tcol + 2 ) * cs_c], z602 );
				svst1( svptrue_b32(),
					&c_[result_tile_BR_corner + ( tcol + 2 ) * cs_c], z702 );

				svfloat32_t z05 = svread_ver_za32_m( z05, svptrue_b32(),
					/* tile: */ 0, /* slice: */ tcol + 3 );
				svfloat32_t z15 = svread_ver_za32_m( z15, svptrue_b32(),
					/* tile: */ 1, /* slice: */ tcol + 3 );
				svfloat32_t z25 = svread_ver_za32_m( z25, svptrue_b32(),
					/* tile: */ 2, /* slice: */ tcol + 3 );
				svfloat32_t z35 = svread_ver_za32_m( z35, svptrue_b32(),
					/* tile: */ 3, /* slice: */ tcol + 3 );

				svfloat32_t z06 = svmul_f32_z( svptrue_b32(), z05, zalpha );
				svfloat32_t z16 = svmul_f32_z( svptrue_b32(), z15, zalpha );
				svfloat32_t z26 = svmul_f32_z( svptrue_b32(), z25, zalpha );
				svfloat32_t z36 = svmul_f32_z( svptrue_b32(), z35, zalpha );

				svfloat32_t zq53 = svld1_f32( svptrue_b32(),
					&c_[result_tile_TL_corner + ( tcol + 3 ) * cs_c] );
				svfloat32_t zq63 = svld1_f32( svptrue_b32(),
					&c_[result_tile_BL_corner + ( tcol + 3 ) * cs_c] );
				svfloat32_t zq73 = svld1_f32( svptrue_b32(),
					&c_[result_tile_TR_corner + ( tcol + 3 ) * cs_c] );
				svfloat32_t zq83 = svld1_f32( svptrue_b32(),
					&c_[result_tile_BR_corner + ( tcol + 3 ) * cs_c] );

				// Scale Z regs by broadcast beta
				svfloat32_t z403 = svmla_m( svptrue_b32(), z06, zq53, zbeta );
				svfloat32_t z503 = svmla_m( svptrue_b32(), z16, zq63, zbeta );
				svfloat32_t z603 = svmla_m( svptrue_b32(), z26, zq73, zbeta );
				svfloat32_t z703 = svmla_m( svptrue_b32(), z36, zq83, zbeta );

				svst1( svptrue_b32(),
					&c_[result_tile_TL_corner + ( tcol + 3 ) * cs_c], z403 );
				svst1( svptrue_b32(),
					&c_[result_tile_BL_corner + ( tcol + 3 ) * cs_c], z503 );
				svst1( svptrue_b32(),
					&c_[result_tile_TR_corner + ( tcol + 3 ) * cs_c], z603 );
				svst1( svptrue_b32(),
					&c_[result_tile_BR_corner + ( tcol + 3 ) * cs_c], z703 );
			}
		}
	}
	else
	{
		const uint64_t result_tile_BL_corner = SVL * rs_c;
		const uint64_t result_tile_TR_corner = SVL;
		const uint64_t result_tile_BR_corner = SVL * rs_c + SVL;

		if ( beta_ == 0 )
		{
			for ( uint64_t tcol = 0; tcol < SVL; tcol += 4 )
			{
				// Read ZA slices into Z regs
				svfloat32_t z0 = svread_hor_za32_m( z0, svptrue_b32(),
					/* tile: */ 0, /* slice: */ tcol + 0 );
				svfloat32_t z1 = svread_hor_za32_m( z1, svptrue_b32(),
					/* tile: */ 1, /* slice: */ tcol + 0 );
				svfloat32_t z2 = svread_hor_za32_m( z2, svptrue_b32(),
					/* tile: */ 2, /* slice: */ tcol + 0 );
				svfloat32_t z3 = svread_hor_za32_m( z3, svptrue_b32(),
					/* tile: */ 3, /* slice: */ tcol + 0 );

				// Scale Z regs by broadcast alpha
				z0 = svmul_f32_z( svptrue_b32(), z0, zalpha );
				z1 = svmul_f32_z( svptrue_b32(), z1, zalpha );
				z2 = svmul_f32_z( svptrue_b32(), z2, zalpha );
				z3 = svmul_f32_z( svptrue_b32(), z3, zalpha );

				// Store full result into C
				svst1( svptrue_b32(),
					&c_[result_tile_TL_corner + ( tcol + 0 ) * rs_c], z0 );
				svst1( svptrue_b32(),
					&c_[result_tile_BL_corner + ( tcol + 0 ) * rs_c], z1 );
				svst1( svptrue_b32(),
					&c_[result_tile_TR_corner + ( tcol + 0 ) * rs_c], z2 );
				svst1( svptrue_b32(),
					&c_[result_tile_BR_corner + ( tcol + 0 ) * rs_c], z3 );

				// Repeat unfolded x4
				z0 = svread_hor_za32_m( z0, svptrue_b32(),
					/* tile: */ 0, /* slice: */ tcol + 1 );
				z1 = svread_hor_za32_m( z1, svptrue_b32(),
					/* tile: */ 1, /* slice: */ tcol + 1 );
				z2 = svread_hor_za32_m( z2, svptrue_b32(),
					/* tile: */ 2, /* slice: */ tcol + 1 );
				z3 = svread_hor_za32_m( z3, svptrue_b32(),
					/* tile: */ 3, /* slice: */ tcol + 1 );

				z0 = svmul_f32_z( svptrue_b32(), z0, zalpha );
				z1 = svmul_f32_z( svptrue_b32(), z1, zalpha );
				z2 = svmul_f32_z( svptrue_b32(), z2, zalpha );
				z3 = svmul_f32_z( svptrue_b32(), z3, zalpha );

				svst1( svptrue_b32(),
					&c_[result_tile_TL_corner + ( tcol + 1 ) * rs_c], z0 );
				svst1( svptrue_b32(),
					&c_[result_tile_BL_corner + ( tcol + 1 ) * rs_c], z1 );
				svst1( svptrue_b32(),
					&c_[result_tile_TR_corner + ( tcol + 1 ) * rs_c], z2 );
				svst1( svptrue_b32(),
					&c_[result_tile_BR_corner + ( tcol + 1 ) * rs_c], z3 );

				z0 = svread_hor_za32_m( z0, svptrue_b32(),
					/* tile: */ 0, /* slice: */ tcol + 2 );
				z1 = svread_hor_za32_m( z1, svptrue_b32(),
					/* tile: */ 1, /* slice: */ tcol + 2 );
				z2 = svread_hor_za32_m( z2, svptrue_b32(),
					/* tile: */ 2, /* slice: */ tcol + 2 );
				z3 = svread_hor_za32_m( z3, svptrue_b32(),
					/* tile: */ 3, /* slice: */ tcol + 2 );

				z0 = svmul_f32_z( svptrue_b32(), z0, zalpha );
				z1 = svmul_f32_z( svptrue_b32(), z1, zalpha );
				z2 = svmul_f32_z( svptrue_b32(), z2, zalpha );
				z3 = svmul_f32_z( svptrue_b32(), z3, zalpha );

				svst1( svptrue_b32(),
					&c_[result_tile_TL_corner + ( tcol + 2 ) * rs_c], z0 );
				svst1( svptrue_b32(),
					&c_[result_tile_BL_corner + ( tcol + 2 ) * rs_c], z1 );
				svst1( svptrue_b32(),
					&c_[result_tile_TR_corner + ( tcol + 2 ) * rs_c], z2 );
				svst1( svptrue_b32(),
					&c_[result_tile_BR_corner + ( tcol + 2 ) * rs_c], z3 );

				z0 = svread_hor_za32_m( z0, svptrue_b32(),
					/* tile: */ 0, /* slice: */ tcol + 3 );
				z1 = svread_hor_za32_m( z1, svptrue_b32(),
					/* tile: */ 1, /* slice: */ tcol + 3 );
				z2 = svread_hor_za32_m( z2, svptrue_b32(),
					/* tile: */ 2, /* slice: */ tcol + 3 );
				z3 = svread_hor_za32_m( z3, svptrue_b32(),
					/* tile: */ 3, /* slice: */ tcol + 3 );

				z0 = svmul_f32_z( svptrue_b32(), z0, zalpha );
				z1 = svmul_f32_z( svptrue_b32(), z1, zalpha );
				z2 = svmul_f32_z( svptrue_b32(), z2, zalpha );
				z3 = svmul_f32_z( svptrue_b32(), z3, zalpha );

				svst1( svptrue_b32(),
					&c_[result_tile_TL_corner + ( tcol + 3 ) * rs_c], z0 );
				svst1( svptrue_b32(),
					&c_[result_tile_BL_corner + ( tcol + 3 ) * rs_c], z1 );
				svst1( svptrue_b32(),
					&c_[result_tile_TR_corner + ( tcol + 3 ) * rs_c], z2 );
				svst1( svptrue_b32(),
					&c_[result_tile_BR_corner + ( tcol + 3 ) * rs_c], z3 );
			}
		}
		else
		{
			for ( uint64_t tcol = 0; tcol < SVL; tcol += 4 )
			{
				// Read ZA slices into Z regs
				svfloat32_t z0 = svread_hor_za32_m( z0, svptrue_b32(),
					/* tile: */ 0, /* slice: */ tcol + 0 );
				svfloat32_t z1 = svread_hor_za32_m( z1, svptrue_b32(),
					/* tile: */ 1, /* slice: */ tcol + 0 );
				svfloat32_t z2 = svread_hor_za32_m( z2, svptrue_b32(),
					/* tile: */ 2, /* slice: */ tcol + 0 );
				svfloat32_t z3 = svread_hor_za32_m( z3, svptrue_b32(),
					/* tile: */ 3, /* slice: */ tcol + 0 );

				// Scale Z regs by broadcast alpha
				svfloat32_t z00 = svmul_f32_z( svptrue_b32(), z0, zalpha );
				svfloat32_t z10 = svmul_f32_z( svptrue_b32(), z1, zalpha );
				svfloat32_t z20 = svmul_f32_z( svptrue_b32(), z2, zalpha );
				svfloat32_t z30 = svmul_f32_z( svptrue_b32(), z3, zalpha );

				// Load C into Z regs
				svfloat32_t zq5 = svld1_f32( svptrue_b32(),
					&c_[result_tile_TL_corner + ( tcol + 0 ) * rs_c] );
				svfloat32_t zq6 = svld1_f32( svptrue_b32(),
					&c_[result_tile_BL_corner + ( tcol + 0 ) * rs_c] );
				svfloat32_t zq7 = svld1_f32( svptrue_b32(),
					&c_[result_tile_TR_corner + ( tcol + 0 ) * rs_c] );
				svfloat32_t zq8 = svld1_f32( svptrue_b32(),
					&c_[result_tile_BR_corner + ( tcol + 0 ) * rs_c] );

				// Scale Z regs by broadcast beta
				svfloat32_t z40 = svmla_m( svptrue_b32(), z00, zq5, zbeta );
				svfloat32_t z50 = svmla_m( svptrue_b32(), z10, zq6, zbeta );
				svfloat32_t z60 = svmla_m( svptrue_b32(), z20, zq7, zbeta );
				svfloat32_t z70 = svmla_m( svptrue_b32(), z30, zq8,	zbeta );

				// Store full result into C
				svst1( svptrue_b32(),
					&c_[result_tile_TL_corner + ( tcol + 0 ) * rs_c], z40 );
				svst1( svptrue_b32(),
					&c_[result_tile_BL_corner + ( tcol + 0 ) * rs_c], z50 );
				svst1( svptrue_b32(),
					&c_[result_tile_TR_corner + ( tcol + 0 ) * rs_c], z60 );
				svst1( svptrue_b32(),
					&c_[result_tile_BR_corner + ( tcol + 0 ) * rs_c], z70 );

				// Repeat unfolded x4
				svfloat32_t z01 = svread_hor_za32_m( z01, svptrue_b32(),
					/* tile: */ 0, /* slice: */ tcol + 1 );
				svfloat32_t z11 = svread_hor_za32_m( z11, svptrue_b32(),
					/* tile: */ 1, /* slice: */ tcol + 1 );
				svfloat32_t z21 = svread_hor_za32_m( z21, svptrue_b32(),
					/* tile: */ 2, /* slice: */ tcol + 1 );
				svfloat32_t z31 = svread_hor_za32_m( z31, svptrue_b32(),
					/* tile: */ 3, /* slice: */ tcol + 1 );

				svfloat32_t z02 = svmul_f32_z( svptrue_b32(), z01, zalpha );
				svfloat32_t z12 = svmul_f32_z( svptrue_b32(), z11, zalpha );
				svfloat32_t z22 = svmul_f32_z( svptrue_b32(), z21, zalpha );
				svfloat32_t z32 = svmul_f32_z( svptrue_b32(), z31, zalpha );

				svfloat32_t zq51 = svld1_f32( svptrue_b32(),
					&c_[result_tile_TL_corner + ( tcol + 1 ) * rs_c] );
				svfloat32_t zq61 = svld1_f32( svptrue_b32(),
					&c_[result_tile_BL_corner + ( tcol + 1 ) * rs_c] );
				svfloat32_t zq71 = svld1_f32( svptrue_b32(),
					&c_[result_tile_TR_corner + ( tcol + 1 ) * rs_c] );
				svfloat32_t zq81 = svld1_f32( svptrue_b32(),
					&c_[result_tile_BR_corner + ( tcol + 1 ) * rs_c] );

				svfloat32_t z401 = svmla_m( svptrue_b32(), z02, zq51, zbeta );
				svfloat32_t z501 = svmla_m( svptrue_b32(), z12, zq61, zbeta );
				svfloat32_t z601 = svmla_m( svptrue_b32(), z22, zq71, zbeta );
				svfloat32_t z701 = svmla_m( svptrue_b32(), z32, zq81, zbeta );

				svst1( svptrue_b32(),
					&c_[result_tile_TL_corner + ( tcol + 1 ) * rs_c], z401 );
				svst1( svptrue_b32(),
					&c_[result_tile_BL_corner + ( tcol + 1 ) * rs_c], z501 );
				svst1( svptrue_b32(),
					&c_[result_tile_TR_corner + ( tcol + 1 ) * rs_c], z601 );
				svst1( svptrue_b32(),
					&c_[result_tile_BR_corner + ( tcol + 1 ) * rs_c], z701 );

				svfloat32_t z03 = svread_hor_za32_m( z03, svptrue_b32(),
					/* tile: */ 0, /* slice: */ tcol + 2 );
				svfloat32_t z13 = svread_hor_za32_m( z13, svptrue_b32(),
					/* tile: */ 1, /* slice: */ tcol + 2 );
				svfloat32_t z23 = svread_hor_za32_m( z23, svptrue_b32(),
					/* tile: */ 2, /* slice: */ tcol + 2 );
				svfloat32_t z33 = svread_hor_za32_m( z33, svptrue_b32(),
					/* tile: */ 3, /* slice: */ tcol + 2 );

				svfloat32_t z04 = svmul_f32_z( svptrue_b32(), z03, zalpha );
				svfloat32_t z14 = svmul_f32_z( svptrue_b32(), z13, zalpha );
				svfloat32_t z24 = svmul_f32_z( svptrue_b32(), z23, zalpha );
				svfloat32_t z34 = svmul_f32_z( svptrue_b32(), z33, zalpha );

				svfloat32_t zq52 = svld1_f32( svptrue_b32(),
					&c_[result_tile_TL_corner + ( tcol + 2 ) * rs_c] );
				svfloat32_t zq62 = svld1_f32( svptrue_b32(),
					&c_[result_tile_BL_corner + ( tcol + 2 ) * rs_c] );
				svfloat32_t zq72 = svld1_f32( svptrue_b32(),
					&c_[result_tile_TR_corner + ( tcol + 2 ) * rs_c] );
				svfloat32_t zq82 = svld1_f32( svptrue_b32(),
					&c_[result_tile_BR_corner + ( tcol + 2 ) * rs_c] );

				// Scale Z regs by broadcast beta
				svfloat32_t z402 = svmla_m( svptrue_b32(), z04, zq52, zbeta );
				svfloat32_t z502 = svmla_m( svptrue_b32(), z14, zq62, zbeta );
				svfloat32_t z602 = svmla_m( svptrue_b32(), z24, zq72, zbeta );
				svfloat32_t z702 = svmla_m( svptrue_b32(), z34, zq82, zbeta );

				svst1( svptrue_b32(),
					&c_[result_tile_TL_corner + ( tcol + 2 ) * rs_c], z402 );
				svst1( svptrue_b32(),
					&c_[result_tile_BL_corner + ( tcol + 2 ) * rs_c], z502 );
				svst1( svptrue_b32(),
					&c_[result_tile_TR_corner + ( tcol + 2 ) * rs_c], z602 );
				svst1( svptrue_b32(),
					&c_[result_tile_BR_corner + ( tcol + 2 ) * rs_c], z702 );

				svfloat32_t z05 = svread_hor_za32_m( z05, svptrue_b32(),
					/* tile: */ 0, /* slice: */ tcol + 3 );
				svfloat32_t z15 = svread_hor_za32_m( z15, svptrue_b32(),
					/* tile: */ 1, /* slice: */ tcol + 3 );
				svfloat32_t z25 = svread_hor_za32_m( z25, svptrue_b32(),
					/* tile: */ 2, /* slice: */ tcol + 3 );
				svfloat32_t z35 = svread_hor_za32_m( z35, svptrue_b32(),
					/* tile: */ 3, /* slice: */ tcol + 3 );

				svfloat32_t z06 = svmul_f32_z( svptrue_b32(), z05, zalpha );
				svfloat32_t z16 = svmul_f32_z( svptrue_b32(), z15, zalpha );
				svfloat32_t z26 = svmul_f32_z( svptrue_b32(), z25, zalpha );
				svfloat32_t z36 = svmul_f32_z( svptrue_b32(), z35, zalpha );

				svfloat32_t zq53 = svld1_f32( svptrue_b32(),
					&c_[result_tile_TL_corner + ( tcol + 3 ) * rs_c] );
				svfloat32_t zq63 = svld1_f32( svptrue_b32(),
					&c_[result_tile_BL_corner + ( tcol + 3 ) * rs_c] );
				svfloat32_t zq73 = svld1_f32( svptrue_b32(),
					&c_[result_tile_TR_corner + ( tcol + 3 ) * rs_c] );
				svfloat32_t zq83 = svld1_f32( svptrue_b32(),
					&c_[result_tile_BR_corner + ( tcol + 3 ) * rs_c] );

				// Scale Z regs by broadcast beta
				svfloat32_t z403 = svmla_m( svptrue_b32(), z06, zq53, zbeta );
				svfloat32_t z503 = svmla_m( svptrue_b32(), z16, zq63, zbeta );
				svfloat32_t z603 = svmla_m( svptrue_b32(), z26, zq73, zbeta );
				svfloat32_t z703 = svmla_m( svptrue_b32(), z36, zq83, zbeta );

				svst1( svptrue_b32(),
					&c_[result_tile_TL_corner + ( tcol + 3 ) * rs_c], z403 );
				svst1( svptrue_b32(),
					&c_[result_tile_BL_corner + ( tcol + 3 ) * rs_c], z503 );
				svst1( svptrue_b32(),
					&c_[result_tile_TR_corner + ( tcol + 3 ) * rs_c], z603 );
				svst1( svptrue_b32(),
					&c_[result_tile_BR_corner + ( tcol + 3 ) * rs_c], z703 );
			}
		}
	}

	GEMM_UKR_FLUSH_CT( s );

	return;
}