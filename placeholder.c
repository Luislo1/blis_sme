void bli_sgemm_m4sme_asm_8x12
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

	const void* a_next = bli_auxinfo_next_a( data );
	const void* b_next = bli_auxinfo_next_b( data );

	// Typecast local copies of integers in case dim_t and inc_t are a
	// different size than is expected by load instructions.
	uint64_t k_iter = k / 4;
	uint64_t k_left = k % 4;
	uint64_t rs_c   = rs_c0;
	uint64_t cs_c   = cs_c0;

	GEMM_UKR_SETUP_CT_ANY( s, 8, 12, false );


    __asm__ volatile
	(
	" smstart                                    \n\t"
	" smstop                                     \n\t"
	" ldr x0,%[aaddr]                            \n\t" // Load address of A.
	" ldr x1,%[baddr]                            \n\t" // Load address of B.
	" ldr x2,%[caddr]                            \n\t" // Load address of C.
	"                                            \n\t"
	" ldr x10,%[cs_c]                            \n\t" // Load cs_c.
	" lsl x10,x10,#2                             \n\t" // cs_c * sizeof(float) -- AUX.
	"                                            \n\t"
	" ldr x5,%[k_iter]                           \n\t" // Number of unrolled iterations (k_iter).
	" ldr x6,%[k_left]                           \n\t" // Number of remaining iterations (k_left).
	" add x16,x2,x10                             \n\t" //Load address Column 1 of C
	"                                            \n\t"
	" ldr x14,%[rs_c]                            \n\t" // Load rs_c.
	" lsl x14,x14,#2                             \n\t" // rs_c * sizeof(float).
	"                                            \n\t"
	"                                            \n\t"
	" dup  v8.4s, wzr                            \n\t" // Vector for accummulating column 0
	" add x17,x16,x10                            \n\t" //Load address Column 2 of C
	" dup  v9.4s, wzr                            \n\t" // Vector for accummulating column 0
	"                                            \n\t" // Since the columns can cross a cache line boundary,
	                                                   // we also need to prefetch the "ends"
	" add x19,x17,x10                            \n\t" //Load address Column 3 of C
	" dup  v10.4s, wzr                           \n\t" // Vector for accummulating column 1
	" add x20,x19,x10                            \n\t" //Load address Column 4 of C
	" dup  v11.4s, wzr                           \n\t" // Vector for accummulating column 1
	" dup  v12.4s, wzr                           \n\t" // Vector for accummulating column 2
	" add x21,x20,x10                            \n\t" //Load address Column 5 of C
	" dup  v13.4s, wzr                           \n\t" // Vector for accummulating column 2
	"                                            \n\t"
	" dup  v14.4s, wzr                           \n\t" // Vector for accummulating column 3
	" add x22,x21,x10                            \n\t" //Load address Column 6 of C
	" dup  v15.4s, wzr                           \n\t" // Vector for accummulating column 3
	" dup  v16.4s, wzr                           \n\t" // Vector for accummulating column 4
	" add x23,x22,x10                            \n\t" //Load address Column 7 of C
	" dup  v17.4s, wzr                           \n\t" // Vector for accummulating column 4
	" dup  v18.4s, wzr                           \n\t" // Vector for accummulating column 5
	" add x24,x23,x10                            \n\t" //Load address Column 8 of C
	" dup  v19.4s, wzr                           \n\t" // Vector for accummulating column 5
	"                                            \n\t"
	" dup  v20.4s, wzr                           \n\t" // Vector for accummulating column 6
	" add x25,x24,x10                            \n\t" //Load address Column 9 of C
	" dup  v21.4s, wzr                           \n\t" // Vector for accummulating column 6
	" dup  v22.4s, wzr                           \n\t" // Vector for accummulating column 7
	" add x26,x25,x10                            \n\t" //Load address Column 10 of C
	" dup  v23.4s, wzr                           \n\t" // Vector for accummulating column 7
	" dup  v24.4s, wzr                           \n\t" // Vector for accummulating column 8
	" add x27,x26,x10                            \n\t" //Load address Column 11 of C
	" dup  v25.4s, wzr                           \n\t" // Vector for accummulating column 8
	"                                            \n\t"
	" dup  v26.4s, wzr                           \n\t" // Vector for accummulating column 9
	" dup  v27.4s, wzr                           \n\t" // Vector for accummulating column 9
	" dup  v28.4s, wzr                           \n\t" // Vector for accummulating column 10
	" dup  v29.4s, wzr                           \n\t" // Vector for accummulating column 10
	" dup  v30.4s, wzr                           \n\t" // Vector for accummulating column 11
	" dup  v31.4s, wzr                           \n\t" // Vector for accummulating column 11
	"                                            \n\t"
	"                                            \n\t"
	" cmp x5,#0                                  \n\t" // If k_iter == 0, jump to k_left.
	BEQ (SCONSIDERKLEFT)
	"                                            \n\t"
	" ldr q0, [x0]                               \n\t"
	" ldr q1, [x0, #16]                          \n\t" // Load a
	"                                            \n\t"
	" ldr q2, [x1]                               \n\t" // Load b
	" ldr q3, [x1, #16]                          \n\t"
	" ldr q4, [x1, #32]                          \n\t"
	"                                            \n\t"
	" add x0, x0, #32                            \n\t" //update address of A
	" add x1, x1, #48                            \n\t" //update address of B
	"                                            \n\t"
	" cmp x5,1                                   \n\t" // If there is just one k_iter, jump to that one.
	BEQ (SLASTITER)                                    // (as loop is do-while-like).
	"                                            \n\t"
	LABEL (SLOOPKITER)                                 // Body of the k_iter loop.
	"                                            \n\t"
	" ldr q5, [x0]                               \n\t"
	" fmla v8.4s, v0.4s,v2.s[0]                  \n\t" // Accummulate.
	" fmla v9.4s, v1.4s,v2.s[0]                  \n\t" // Accummulate.
	" ldr q6, [x0, #16]                          \n\t"
	" fmla v10.4s,v0.4s,v2.s[1]                  \n\t" // Accummulate.
	" fmla v11.4s,v1.4s,v2.s[1]                  \n\t" // Accummulate.
	" fmla v12.4s,v0.4s,v2.s[2]                  \n\t" // Accummulate.
	" fmla v13.4s,v1.4s,v2.s[2]                  \n\t" // Accummulate.
	" fmla v14.4s,v0.4s,v2.s[3]                  \n\t" // Accummulate.
	" fmla v15.4s,v1.4s,v2.s[3]                  \n\t" // Accummulate.
	" ldr q2, [x1]                               \n\t"
	"                                            \n\t"
	" fmla v16.4s,v0.4s,v3.s[0]                  \n\t" // Accummulate.
	" fmla v17.4s,v1.4s,v3.s[0]                  \n\t" // Accummulate.
	" fmla v18.4s,v0.4s,v3.s[1]                  \n\t" // Accummulate.
	" fmla v19.4s,v1.4s,v3.s[1]                  \n\t" // Accummulate.
	" fmla v20.4s,v0.4s,v3.s[2]                  \n\t" // Accummulate.
	" fmla v21.4s,v1.4s,v3.s[2]                  \n\t" // Accummulate.
	" fmla v22.4s,v0.4s,v3.s[3]                  \n\t" // Accummulate.
	" fmla v23.4s,v1.4s,v3.s[3]                  \n\t" // Accummulate.
	"                                            \n\t"
	" fmla v24.4s,v0.4s,v4.s[0]                  \n\t" // Accummulate.
	" fmla v26.4s,v0.4s,v4.s[1]                  \n\t" // Accummulate.
	" fmla v28.4s,v0.4s,v4.s[2]                  \n\t" // Accummulate.
	" fmla v30.4s,v0.4s,v4.s[3]                  \n\t" // Accummulate.
	" ldr q3, [x1, #16]                          \n\t"
	"                                            \n\t"
	" fmla v25.4s,v1.4s,v4.s[0]                  \n\t" // Accummulate.
	" fmla v27.4s,v1.4s,v4.s[1]                  \n\t" // Accummulate.
	" fmla v29.4s,v1.4s,v4.s[2]                  \n\t" // Accummulate.
	" fmla v31.4s,v1.4s,v4.s[3]                  \n\t" // Accummulate.
	" ldr q4, [x1, #32]                          \n\t"
	"                                            \n\t" //End It 1
	"                                            \n\t"
	" ldr q0, [x0, #32]                          \n\t"
	" fmla v8.4s,v5.4s,v2.s[0]                   \n\t" // Accummulate.
	" fmla v9.4s,v6.4s,v2.s[0]                   \n\t" // Accummulate.
	" ldr q1, [x0, #48]                          \n\t"
	" fmla v10.4s,v5.4s,v2.s[1]                  \n\t" // Accummulate.
	" fmla v11.4s,v6.4s,v2.s[1]                  \n\t" // Accummulate.
	" fmla v12.4s,v5.4s,v2.s[2]                  \n\t" // Accummulate.
	" fmla v13.4s,v6.4s,v2.s[2]                  \n\t" // Accummulate.
	" fmla v14.4s,v5.4s,v2.s[3]                  \n\t" // Accummulate.
	" fmla v15.4s,v6.4s,v2.s[3]                  \n\t" // Accummulate.
	" ldr q2, [x1, #48]                          \n\t"
	"                                            \n\t"
	" fmla v16.4s,v5.4s,v3.s[0]                  \n\t" // Accummulate.
	" fmla v17.4s,v6.4s,v3.s[0]                  \n\t" // Accummulate.
	" fmla v18.4s,v5.4s,v3.s[1]                  \n\t" // Accummulate.
	" fmla v19.4s,v6.4s,v3.s[1]                  \n\t" // Accummulate.
	" fmla v20.4s,v5.4s,v3.s[2]                  \n\t" // Accummulate.
	" fmla v21.4s,v6.4s,v3.s[2]                  \n\t" // Accummulate.
	" fmla v22.4s,v5.4s,v3.s[3]                  \n\t" // Accummulate.
	" fmla v23.4s,v6.4s,v3.s[3]                  \n\t" // Accummulate.
	"                                            \n\t"
	" fmla v24.4s,v5.4s,v4.s[0]                  \n\t" // Accummulate.
	" fmla v26.4s,v5.4s,v4.s[1]                  \n\t" // Accummulate.
	" fmla v28.4s,v5.4s,v4.s[2]                  \n\t" // Accummulate.
	" fmla v30.4s,v5.4s,v4.s[3]                  \n\t" // Accummulate.
	" ldr q3, [x1, #64]                          \n\t"
	"                                            \n\t"
	" fmla v25.4s,v6.4s,v4.s[0]                  \n\t" // Accummulate.
	" fmla v27.4s,v6.4s,v4.s[1]                  \n\t" // Accummulate.
	" fmla v29.4s,v6.4s,v4.s[2]                  \n\t" // Accummulate.
	" fmla v31.4s,v6.4s,v4.s[3]                  \n\t" // Accummulate.
	" ldr q4, [x1, #80]                          \n\t"
	"                                            \n\t" //End It 2
	"                                            \n\t"
	" ldr q5, [x0, #64]                          \n\t"
	" fmla v8.4s,v0.4s,v2.s[0]                   \n\t" // Accummulate.
	" fmla v9.4s,v1.4s,v2.s[0]                   \n\t" // Accummulate.
	" ldr q6, [x0, #80]                          \n\t"
	" fmla v10.4s,v0.4s,v2.s[1]                  \n\t" // Accummulate.
	" fmla v11.4s,v1.4s,v2.s[1]                  \n\t" // Accummulate.
	" fmla v12.4s,v0.4s,v2.s[2]                  \n\t" // Accummulate.
	" fmla v13.4s,v1.4s,v2.s[2]                  \n\t" // Accummulate.
	" fmla v14.4s,v0.4s,v2.s[3]                  \n\t" // Accummulate.
	" fmla v15.4s,v1.4s,v2.s[3]                  \n\t" // Accummulate.
	" ldr q2, [x1, #96]                          \n\t"
	"                                            \n\t"
	" fmla v16.4s,v0.4s,v3.s[0]                  \n\t" // Accummulate.
	" fmla v17.4s,v1.4s,v3.s[0]                  \n\t" // Accummulate.
	" fmla v18.4s,v0.4s,v3.s[1]                  \n\t" // Accummulate.
	" fmla v19.4s,v1.4s,v3.s[1]                  \n\t" // Accummulate.
	" fmla v20.4s,v0.4s,v3.s[2]                  \n\t" // Accummulate.
	" fmla v21.4s,v1.4s,v3.s[2]                  \n\t" // Accummulate.
	" fmla v22.4s,v0.4s,v3.s[3]                  \n\t" // Accummulate.
	" fmla v23.4s,v1.4s,v3.s[3]                  \n\t" // Accummulate.
	"                                            \n\t"
	" fmla v24.4s,v0.4s,v4.s[0]                  \n\t" // Accummulate.
	" fmla v26.4s,v0.4s,v4.s[1]                  \n\t" // Accummulate.
	" fmla v28.4s,v0.4s,v4.s[2]                  \n\t" // Accummulate.
	" fmla v30.4s,v0.4s,v4.s[3]                  \n\t" // Accummulate.
	" ldr q3, [x1, #112]                         \n\t"
	"                                            \n\t"
	" fmla v25.4s,v1.4s,v4.s[0]                  \n\t" // Accummulate.
	" fmla v27.4s,v1.4s,v4.s[1]                  \n\t" // Accummulate.
	" fmla v29.4s,v1.4s,v4.s[2]                  \n\t" // Accummulate.
	" fmla v31.4s,v1.4s,v4.s[3]                  \n\t" // Accummulate.
	" ldr q4, [x1, #128]                         \n\t"
	"                                            \n\t" //End It 3
	"                                            \n\t"
	" ldr q0, [x0, #96]                          \n\t"
	" fmla v8.4s,v5.4s,v2.s[0]                   \n\t" // Accummulate.
	" fmla v9.4s,v6.4s,v2.s[0]                   \n\t" // Accummulate.
	" ldr q1, [x0, #112]                         \n\t"
	" fmla v10.4s,v5.4s,v2.s[1]                  \n\t" // Accummulate.
	" fmla v11.4s,v6.4s,v2.s[1]                  \n\t" // Accummulate.
	" fmla v12.4s,v5.4s,v2.s[2]                  \n\t" // Accummulate.
	" fmla v13.4s,v6.4s,v2.s[2]                  \n\t" // Accummulate.
	" fmla v14.4s,v5.4s,v2.s[3]                  \n\t" // Accummulate.
	" fmla v15.4s,v6.4s,v2.s[3]                  \n\t" // Accummulate.
	" ldr q2, [x1, #144]                         \n\t"
	"                                            \n\t"
	" fmla v16.4s,v5.4s,v3.s[0]                  \n\t" // Accummulate.
	" fmla v17.4s,v6.4s,v3.s[0]                  \n\t" // Accummulate.
	" fmla v18.4s,v5.4s,v3.s[1]                  \n\t" // Accummulate.
	" fmla v19.4s,v6.4s,v3.s[1]                  \n\t" // Accummulate.
	" fmla v20.4s,v5.4s,v3.s[2]                  \n\t" // Accummulate.
	" fmla v21.4s,v6.4s,v3.s[2]                  \n\t" // Accummulate.
	" fmla v22.4s,v5.4s,v3.s[3]                  \n\t" // Accummulate.
	" fmla v23.4s,v6.4s,v3.s[3]                  \n\t" // Accummulate.
	"                                            \n\t"
	" fmla v24.4s,v5.4s,v4.s[0]                  \n\t" // Accummulate.
	" fmla v26.4s,v5.4s,v4.s[1]                  \n\t" // Accummulate.
	" fmla v28.4s,v5.4s,v4.s[2]                  \n\t" // Accummulate.
	" fmla v30.4s,v5.4s,v4.s[3]                  \n\t" // Accummulate.
	" ldr q3, [x1, #160]                         \n\t"
	"                                            \n\t"
	" fmla v25.4s,v6.4s,v4.s[0]                  \n\t" // Accummulate.
	" fmla v27.4s,v6.4s,v4.s[1]                  \n\t" // Accummulate.
	" fmla v29.4s,v6.4s,v4.s[2]                  \n\t" // Accummulate.
	" fmla v31.4s,v6.4s,v4.s[3]                  \n\t" // Accummulate.
	" ldr q4, [x1, #176]                         \n\t"
	" add x1, x1, #192                           \n\t"
	" add x0, x0, #128                           \n\t"
	"                                            \n\t" //End It 4
	" sub x5,x5,1                                \n\t" // i-=1.
	" cmp x5,1                                   \n\t" // Iterate again if we are not in k_iter == 1.
	BNE (SLOOPKITER)
	"                                            \n\t"
	LABEL (SLASTITER)                                  // Last iteration of k_iter loop.
	"                                            \n\t"
	"                                            \n\t"
	" ldr q5, [x0]                               \n\t"
	" fmla v8.4s,v0.4s,v2.s[0]                   \n\t" // Accummulate.
	" fmla v9.4s,v1.4s,v2.s[0]                   \n\t" // Accummulate.
	" ldr q6, [x0, #16]                          \n\t"
	" fmla v10.4s,v0.4s,v2.s[1]                  \n\t" // Accummulate.
	" fmla v11.4s,v1.4s,v2.s[1]                  \n\t" // Accummulate.
	" fmla v12.4s,v0.4s,v2.s[2]                  \n\t" // Accummulate.
	" fmla v13.4s,v1.4s,v2.s[2]                  \n\t" // Accummulate.
	" fmla v14.4s,v0.4s,v2.s[3]                  \n\t" // Accummulate.
	" fmla v15.4s,v1.4s,v2.s[3]                  \n\t" // Accummulate.
	" ldr q2, [x1]                               \n\t"
	"                                            \n\t"
	" fmla v16.4s,v0.4s,v3.s[0]                  \n\t" // Accummulate.
	" fmla v17.4s,v1.4s,v3.s[0]                  \n\t" // Accummulate.
	" fmla v18.4s,v0.4s,v3.s[1]                  \n\t" // Accummulate.
	" fmla v19.4s,v1.4s,v3.s[1]                  \n\t" // Accummulate.
	" fmla v20.4s,v0.4s,v3.s[2]                  \n\t" // Accummulate.
	" fmla v21.4s,v1.4s,v3.s[2]                  \n\t" // Accummulate.
	" fmla v22.4s,v0.4s,v3.s[3]                  \n\t" // Accummulate.
	" fmla v23.4s,v1.4s,v3.s[3]                  \n\t" // Accummulate.
	"                                            \n\t"
	" fmla v24.4s,v0.4s,v4.s[0]                  \n\t" // Accummulate.
	" fmla v26.4s,v0.4s,v4.s[1]                  \n\t" // Accummulate.
	" fmla v28.4s,v0.4s,v4.s[2]                  \n\t" // Accummulate.
	" fmla v30.4s,v0.4s,v4.s[3]                  \n\t" // Accummulate.
	" ldr q3, [x1, #16]                          \n\t"
	"                                            \n\t"
	" fmla v25.4s,v1.4s,v4.s[0]                  \n\t" // Accummulate.
	" fmla v27.4s,v1.4s,v4.s[1]                  \n\t" // Accummulate.
	" fmla v29.4s,v1.4s,v4.s[2]                  \n\t" // Accummulate.
	" fmla v31.4s,v1.4s,v4.s[3]                  \n\t" // Accummulate.
	" ldr q4, [x1, #32]                          \n\t"
	"                                            \n\t" //End It 1
	"                                            \n\t"
	" ldr q0, [x0, #32]                          \n\t"
	" fmla v8.4s,v5.4s,v2.s[0]                   \n\t" // Accummulate.
	" fmla v9.4s,v6.4s,v2.s[0]                   \n\t" // Accummulate.
	" ldr q1, [x0, #48]                          \n\t"
	" fmla v10.4s,v5.4s,v2.s[1]                  \n\t" // Accummulate.
	" fmla v11.4s,v6.4s,v2.s[1]                  \n\t" // Accummulate.
	" fmla v12.4s,v5.4s,v2.s[2]                  \n\t" // Accummulate.
	" fmla v13.4s,v6.4s,v2.s[2]                  \n\t" // Accummulate.
	" fmla v14.4s,v5.4s,v2.s[3]                  \n\t" // Accummulate.
	" fmla v15.4s,v6.4s,v2.s[3]                  \n\t" // Accummulate.
	" ldr q2, [x1, #48]                          \n\t"
	"                                            \n\t"
	" fmla v16.4s,v5.4s,v3.s[0]                  \n\t" // Accummulate.
	" fmla v17.4s,v6.4s,v3.s[0]                  \n\t" // Accummulate.
	" fmla v18.4s,v5.4s,v3.s[1]                  \n\t" // Accummulate.
	" fmla v19.4s,v6.4s,v3.s[1]                  \n\t" // Accummulate.
	" fmla v20.4s,v5.4s,v3.s[2]                  \n\t" // Accummulate.
	" fmla v21.4s,v6.4s,v3.s[2]                  \n\t" // Accummulate.
	" fmla v22.4s,v5.4s,v3.s[3]                  \n\t" // Accummulate.
	" fmla v23.4s,v6.4s,v3.s[3]                  \n\t" // Accummulate.
	"                                            \n\t"
	" fmla v24.4s,v5.4s,v4.s[0]                  \n\t" // Accummulate.
	" fmla v26.4s,v5.4s,v4.s[1]                  \n\t" // Accummulate.
	" fmla v28.4s,v5.4s,v4.s[2]                  \n\t" // Accummulate.
	" fmla v30.4s,v5.4s,v4.s[3]                  \n\t" // Accummulate.
	" ldr q3, [x1, #64]                          \n\t"
	"                                            \n\t"
	" fmla v25.4s,v6.4s,v4.s[0]                  \n\t" // Accummulate.
	" fmla v27.4s,v6.4s,v4.s[1]                  \n\t" // Accummulate.
	" fmla v29.4s,v6.4s,v4.s[2]                  \n\t" // Accummulate.
	" fmla v31.4s,v6.4s,v4.s[3]                  \n\t" // Accummulate.
	" ldr q4, [x1, #80]                          \n\t"
	"                                            \n\t" //End It 2
	"                                            \n\t"
	" ldr q5, [x0, #64]                          \n\t"
	" fmla v8.4s,v0.4s,v2.s[0]                   \n\t" // Accummulate.
	" fmla v9.4s,v1.4s,v2.s[0]                   \n\t" // Accummulate.
	" ldr q6, [x0, #80]                          \n\t"
	" fmla v10.4s,v0.4s,v2.s[1]                  \n\t" // Accummulate.
	" fmla v11.4s,v1.4s,v2.s[1]                  \n\t" // Accummulate.
	" fmla v12.4s,v0.4s,v2.s[2]                  \n\t" // Accummulate.
	" fmla v13.4s,v1.4s,v2.s[2]                  \n\t" // Accummulate.
	" fmla v14.4s,v0.4s,v2.s[3]                  \n\t" // Accummulate.
	" fmla v15.4s,v1.4s,v2.s[3]                  \n\t" // Accummulate.
	" ldr q2, [x1, #96]                          \n\t"
	"                                            \n\t"
	" fmla v16.4s,v0.4s,v3.s[0]                  \n\t" // Accummulate.
	" fmla v17.4s,v1.4s,v3.s[0]                  \n\t" // Accummulate.
	" fmla v18.4s,v0.4s,v3.s[1]                  \n\t" // Accummulate.
	" fmla v19.4s,v1.4s,v3.s[1]                  \n\t" // Accummulate.
	" fmla v20.4s,v0.4s,v3.s[2]                  \n\t" // Accummulate.
	" fmla v21.4s,v1.4s,v3.s[2]                  \n\t" // Accummulate.
	" fmla v22.4s,v0.4s,v3.s[3]                  \n\t" // Accummulate.
	" fmla v23.4s,v1.4s,v3.s[3]                  \n\t" // Accummulate.
	"                                            \n\t"
	" fmla v24.4s,v0.4s,v4.s[0]                  \n\t" // Accummulate.
	" fmla v26.4s,v0.4s,v4.s[1]                  \n\t" // Accummulate.
	" fmla v28.4s,v0.4s,v4.s[2]                  \n\t" // Accummulate.
	" fmla v30.4s,v0.4s,v4.s[3]                  \n\t" // Accummulate.
	" ldr q3, [x1, #112]                         \n\t"
	"                                            \n\t"
	" fmla v25.4s,v1.4s,v4.s[0]                  \n\t" // Accummulate.
	" fmla v27.4s,v1.4s,v4.s[1]                  \n\t" // Accummulate.
	" fmla v29.4s,v1.4s,v4.s[2]                  \n\t" // Accummulate.
	" fmla v31.4s,v1.4s,v4.s[3]                  \n\t" // Accummulate.
	" ldr q4, [x1, #128]                         \n\t"
	"                                            \n\t" //End It 3
	"                                            \n\t"
	" fmla v8.4s,v5.4s,v2.s[0]                   \n\t" // Accummulate.
	" fmla v9.4s,v6.4s,v2.s[0]                   \n\t" // Accummulate.
	" fmla v10.4s,v5.4s,v2.s[1]                  \n\t" // Accummulate.
	" fmla v11.4s,v6.4s,v2.s[1]                  \n\t" // Accummulate.
	" fmla v12.4s,v5.4s,v2.s[2]                  \n\t" // Accummulate.
	" fmla v13.4s,v6.4s,v2.s[2]                  \n\t" // Accummulate.
	" fmla v14.4s,v5.4s,v2.s[3]                  \n\t" // Accummulate.
	" fmla v15.4s,v6.4s,v2.s[3]                  \n\t" // Accummulate.
	"                                            \n\t"
	" fmla v16.4s,v5.4s,v3.s[0]                  \n\t" // Accummulate.
	" fmla v17.4s,v6.4s,v3.s[0]                  \n\t" // Accummulate.
	" fmla v18.4s,v5.4s,v3.s[1]                  \n\t" // Accummulate.
	" fmla v19.4s,v6.4s,v3.s[1]                  \n\t" // Accummulate.
	" fmla v20.4s,v5.4s,v3.s[2]                  \n\t" // Accummulate.
	" fmla v21.4s,v6.4s,v3.s[2]                  \n\t" // Accummulate.
	" fmla v22.4s,v5.4s,v3.s[3]                  \n\t" // Accummulate.
	" fmla v23.4s,v6.4s,v3.s[3]                  \n\t" // Accummulate.
	"                                            \n\t"
	" fmla v24.4s,v5.4s,v4.s[0]                  \n\t" // Accummulate.
	" fmla v26.4s,v5.4s,v4.s[1]                  \n\t" // Accummulate.
	" fmla v28.4s,v5.4s,v4.s[2]                  \n\t" // Accummulate.
	" fmla v30.4s,v5.4s,v4.s[3]                  \n\t" // Accummulate.
	"                                            \n\t"
	" fmla v25.4s,v6.4s,v4.s[0]                  \n\t" // Accummulate.
	" fmla v27.4s,v6.4s,v4.s[1]                  \n\t" // Accummulate.
	" fmla v29.4s,v6.4s,v4.s[2]                  \n\t" // Accummulate.
	" fmla v31.4s,v6.4s,v4.s[3]                  \n\t" // Accummulate.
	" add x1, x1, #144                           \n\t"
	" add x0, x0, #96                            \n\t"
	"                                            \n\t" //End It 4
	"                                            \n\t"
	LABEL (SCONSIDERKLEFT)
	" cmp x6,0                                   \n\t" // If k_left == 0, we are done.
	BEQ (SPOSTACCUM)                                   // else, we enter the k_left loop.
	"                                            \n\t"
	LABEL (SLOOPKLEFT)                                 // Body of the left iterations
	"                                            \n\t"
	" ldr q0, [x0],#16                           \n\t"
	" ldr q1, [x0],#16                           \n\t" // Load a
	"                                            \n\t"
	" ldr q2, [x1],#16                           \n\t" // Load b
	" ldr q3, [x1],#16                           \n\t"
	" ldr q4, [x1],#16                           \n\t"
	"                                            \n\t"
	" sub x6,x6,1                                \n\t" // i = i-1.
	"                                            \n\t"
	" fmla v8.4s,v0.4s,v2.s[0]                   \n\t" // Accummulate.
	" fmla v9.4s,v1.4s,v2.s[0]                   \n\t" // Accummulate.
	" fmla v10.4s,v0.4s,v2.s[1]                  \n\t" // Accummulate.
	" fmla v11.4s,v1.4s,v2.s[1]                  \n\t" // Accummulate.
	" fmla v12.4s,v0.4s,v2.s[2]                  \n\t" // Accummulate.
	" fmla v13.4s,v1.4s,v2.s[2]                  \n\t" // Accummulate.
	" fmla v14.4s,v0.4s,v2.s[3]                  \n\t" // Accummulate.
	" fmla v15.4s,v1.4s,v2.s[3]                  \n\t" // Accummulate.
	"                                            \n\t"
	" fmla v16.4s,v0.4s,v3.s[0]                  \n\t" // Accummulate.
	" fmla v17.4s,v1.4s,v3.s[0]                  \n\t" // Accummulate.
	" fmla v18.4s,v0.4s,v3.s[1]                  \n\t" // Accummulate.
	" fmla v19.4s,v1.4s,v3.s[1]                  \n\t" // Accummulate.
	" fmla v20.4s,v0.4s,v3.s[2]                  \n\t" // Accummulate.
	" fmla v21.4s,v1.4s,v3.s[2]                  \n\t" // Accummulate.
	" fmla v22.4s,v0.4s,v3.s[3]                  \n\t" // Accummulate.
	" fmla v23.4s,v1.4s,v3.s[3]                  \n\t" // Accummulate.
	"                                            \n\t"
	" fmla v24.4s,v0.4s,v4.s[0]                  \n\t" // Accummulate.
	" fmla v26.4s,v0.4s,v4.s[1]                  \n\t" // Accummulate.
	" fmla v28.4s,v0.4s,v4.s[2]                  \n\t" // Accummulate.
	" fmla v30.4s,v0.4s,v4.s[3]                  \n\t" // Accummulate.
	" fmla v25.4s,v1.4s,v4.s[0]                  \n\t" // Accummulate.
	" fmla v27.4s,v1.4s,v4.s[1]                  \n\t" // Accummulate.
	" fmla v29.4s,v1.4s,v4.s[2]                  \n\t" // Accummulate.
	" fmla v31.4s,v1.4s,v4.s[3]                  \n\t" // Accummulate.
	"                                            \n\t"
	" cmp x6,0                                   \n\t" // Iterate again.
	BNE (SLOOPKLEFT)                                   // if i!=0.
	"                                            \n\t"
	LABEL (SPOSTACCUM)
	"                                            \n\t"
	" ldr x0,%[alpha]                            \n\t" // Alpha address.
	" ldr x1,%[beta]                             \n\t" // Beta address.
	"                                            \n\t"
	" ld1r {v6.4s},[x0]                          \n\t" // Load alpha.
	" ld1r {v7.4s},[x1]                          \n\t" // Load beta
	"                                            \n\t"
	" ldr x0,%[a_next]                           \n\t" // Pointer to next block of A.
	" ldr x1,%[b_next]                           \n\t" // Pointer to next pointer of B.
	"                                            \n\t"
	" cmp x14,#4                                 \n\t" // If rs_c != 1 (column-major)
	BNE (SGENSTORED)
	"                                            \n\t"
	LABEL (SCOLSTORED)                                 // C is column-major.
	"                                            \n\t"
	" dup  v0.4s, wzr                            \n\t"
	" dup  v1.4s, wzr                            \n\t"
	" dup  v2.4s, wzr                            \n\t"
	" dup  v3.4s, wzr                            \n\t"
	" dup  v4.4s, wzr                            \n\t"
	" dup  v5.4s, wzr                            \n\t"
	"                                            \n\t"
	" fcmp s7,#0.0                               \n\t"
	BEQ (SBETAZEROCOLSTOREDS1)                         // Taking care of the beta==0 case.
	"                                            \n\t"
	" ldr q0, [x2]                               \n\t" //Load column 0 of C
	" ldr q1, [x2, #16]                          \n\t"
	" ldr q2, [x16]                              \n\t" //Load column 1 of C
	" ldr q3, [x16, #16]                         \n\t"
	" ldr q4, [x17]                              \n\t" //Load column 2 of C
	" ldr q5, [x17, #16]                         \n\t"
	"                                            \n\t"
	" fmul v0.4s,v0.4s,v7.s[0]                   \n\t" // Scale by beta
	" fmul v1.4s,v1.4s,v7.s[0]                   \n\t" // Scale by beta
	" fmul v2.4s,v2.4s,v7.s[0]                   \n\t" // Scale by beta
	" fmul v3.4s,v3.4s,v7.s[0]                   \n\t" // Scale by beta
	" fmul v4.4s,v4.4s,v7.s[0]                   \n\t" // Scale by beta
	" fmul v5.4s,v5.4s,v7.s[0]                   \n\t" // Scale by beta
	"                                            \n\t"
	LABEL (SBETAZEROCOLSTOREDS1)
	"                                            \n\t"
	" fmla v0.4s,v8.4s,v6.s[0]                   \n\t" // Scale by alpha
	" fmla v1.4s,v9.4s,v6.s[0]                   \n\t" // Scale by alpha
	" fmla v2.4s,v10.4s,v6.s[0]                  \n\t" // Scale by alpha
	" fmla v3.4s,v11.4s,v6.s[0]                  \n\t" // Scale by alpha
	" fmla v4.4s,v12.4s,v6.s[0]                  \n\t" // Scale by alpha
	" fmla v5.4s,v13.4s,v6.s[0]                  \n\t" // Scale by alpha
	"                                            \n\t"
	" str q0, [x2]                               \n\t" //Store column 0 of C
	" str q1, [x2, #16]                          \n\t"
	" str q2, [x16]                              \n\t" //Store column 1 of C
	" str q3, [x16, #16]                         \n\t"
	" str q4, [x17]                              \n\t" //Store column 2 of C
	" str q5, [x17, #16]                         \n\t"
	"                                            \n\t"
	" dup  v8.4s, wzr                            \n\t"
	" dup  v9.4s, wzr                            \n\t"
	" dup  v10.4s, wzr                           \n\t"
	" dup  v11.4s, wzr                           \n\t"
	" dup  v12.4s, wzr                           \n\t"
	" dup  v13.4s, wzr                           \n\t"
	"                                            \n\t"
	" fcmp s7,#0.0                               \n\t"
	BEQ (SBETAZEROCOLSTOREDS2)                         // Taking care of the beta==0 case.
	"                                            \n\t"
	" ldr q8, [x19]                              \n\t" //Load column 3 of C
	" ldr q9, [x19, #16]                         \n\t"
	" ldr q10, [x20]                             \n\t" //Load column 4 of C
	" ldr q11, [x20, #16]                        \n\t"
	" ldr q12, [x21]                             \n\t" //Load column 5 of C
	" ldr q13, [x21, #16]                        \n\t"
	"                                            \n\t"
	" fmul v8.4s, v8.4s, v7.s[0]                 \n\t" // Scale by beta
	" fmul v9.4s, v9.4s, v7.s[0]                 \n\t" // Scale by beta
	" fmul v10.4s,v10.4s,v7.s[0]                 \n\t" // Scale by beta
	" fmul v11.4s,v11.4s,v7.s[0]                 \n\t" // Scale by beta
	" fmul v12.4s,v12.4s,v7.s[0]                 \n\t" // Scale by beta
	" fmul v13.4s,v13.4s,v7.s[0]                 \n\t" // Scale by beta
	"                                            \n\t"
	LABEL (SBETAZEROCOLSTOREDS2)
	"                                            \n\t"
	" fmla v8.4s, v14.4s,v6.s[0]                 \n\t" // Scale by alpha
	" fmla v9.4s, v15.4s,v6.s[0]                 \n\t" // Scale by alpha
	" fmla v10.4s,v16.4s,v6.s[0]                 \n\t" // Scale by alpha
	" fmla v11.4s,v17.4s,v6.s[0]                 \n\t" // Scale by alpha
	" fmla v12.4s,v18.4s,v6.s[0]                 \n\t" // Scale by alpha
	" fmla v13.4s,v19.4s,v6.s[0]                 \n\t" // Scale by alpha
	"                                            \n\t"
	" str q8, [x19]                              \n\t" //Store column 3 of C
	" str q9, [x19, #16]                         \n\t"
	" str q10, [x20]                             \n\t" //Store column 4 of C
	" str q11, [x20, #16]                        \n\t"
	" str q12, [x21]                             \n\t" //Store column 5 of C
	" str q13, [x21, #16]                        \n\t"
	"                                            \n\t"
	" dup  v0.4s, wzr                            \n\t"
	" dup  v1.4s, wzr                            \n\t"
	" dup  v2.4s, wzr                            \n\t"
	" dup  v3.4s, wzr                            \n\t"
	" dup  v4.4s, wzr                            \n\t"
	" dup  v5.4s, wzr                            \n\t"
	"                                            \n\t"
	" fcmp s7,#0.0                               \n\t"
	BEQ (SBETAZEROCOLSTOREDS3)                         // Taking care of the beta==0 case.
	"                                            \n\t"
	" ldr q0, [x22]                              \n\t" //Load column 6 of C
	" ldr q1, [x22, #16]                         \n\t"
	" ldr q2, [x23]                              \n\t" //Load column 7 of C
	" ldr q3, [x23, #16]                         \n\t"
	" ldr q4, [x24]                              \n\t" //Load column 8 of C
	" ldr q5, [x24, #16]                         \n\t"
	"                                            \n\t"
	" fmul v0.4s,v0.4s,v7.s[0]                   \n\t" // Scale by beta
	" fmul v1.4s,v1.4s,v7.s[0]                   \n\t" // Scale by beta
	" fmul v2.4s,v2.4s,v7.s[0]                   \n\t" // Scale by beta
	" fmul v3.4s,v3.4s,v7.s[0]                   \n\t" // Scale by beta
	" fmul v4.4s,v4.4s,v7.s[0]                   \n\t" // Scale by beta
	" fmul v5.4s,v5.4s,v7.s[0]                   \n\t" // Scale by beta
	"                                            \n\t"
	LABEL (SBETAZEROCOLSTOREDS3)
	"                                            \n\t"
	" fmla v0.4s,v20.4s,v6.s[0]                  \n\t" // Scale by alpha
	" fmla v1.4s,v21.4s,v6.s[0]                  \n\t" // Scale by alpha
	" fmla v2.4s,v22.4s,v6.s[0]                  \n\t" // Scale by alpha
	" fmla v3.4s,v23.4s,v6.s[0]                  \n\t" // Scale by alpha
	" fmla v4.4s,v24.4s,v6.s[0]                  \n\t" // Scale by alpha
	" fmla v5.4s,v25.4s,v6.s[0]                  \n\t" // Scale by alpha
	"                                            \n\t"
	" str q0, [x22]                              \n\t" //Store column 6 of C
	" str q1, [x22, #16]                         \n\t"
	" str q2, [x23]                              \n\t" //Store column 7 of C
	" str q3, [x23, #16]                         \n\t"
	" str q4, [x24]                              \n\t" //Store column 8 of C
	" str q5, [x24, #16]                         \n\t"
	"                                            \n\t"
	" dup  v8.4s, wzr                            \n\t"
	" dup  v9.4s, wzr                            \n\t"
	" dup  v10.4s, wzr                            \n\t"
	" dup  v11.4s, wzr                            \n\t"
	" dup  v12.4s, wzr                            \n\t"
	" dup  v13.4s, wzr                            \n\t"
	"                                            \n\t"
	" fcmp s7,#0.0                               \n\t"
	BEQ (SBETAZEROCOLSTOREDS4)                         // Taking care of the beta==0 case.
	"                                            \n\t"
	" ldr q8, [x25]                              \n\t" //Load column 9 of C
	" ldr q9, [x25, #16]                         \n\t"
	" ldr q10, [x26]                             \n\t" //Load column 10 of C
	" ldr q11, [x26, #16]                        \n\t"
	" ldr q12, [x27]                             \n\t" //Load column 11 of C
	" ldr q13, [x27, #16]                        \n\t"
	"                                            \n\t"
	" fmul v8.4s, v8.4s, v7.s[0]                 \n\t" // Scale by beta
	" fmul v9.4s, v9.4s, v7.s[0]                 \n\t" // Scale by beta
	" fmul v10.4s,v10.4s,v7.s[0]                 \n\t" // Scale by beta
	" fmul v11.4s,v11.4s,v7.s[0]                 \n\t" // Scale by beta
	" fmul v12.4s,v12.4s,v7.s[0]                 \n\t" // Scale by beta
	" fmul v13.4s,v13.4s,v7.s[0]                 \n\t" // Scale by beta
	"                                            \n\t"
	LABEL (SBETAZEROCOLSTOREDS4)
	"                                            \n\t"
	"                                            \n\t"
	" fmla v8.4s, v26.4s,v6.s[0]                 \n\t" // Scale by alpha
	" fmla v9.4s, v27.4s,v6.s[0]                 \n\t" // Scale by alpha
	" fmla v10.4s,v28.4s,v6.s[0]                 \n\t" // Scale by alpha
	" fmla v11.4s,v29.4s,v6.s[0]                 \n\t" // Scale by alpha
	" fmla v12.4s,v30.4s,v6.s[0]                 \n\t" // Scale by alpha
	" fmla v13.4s,v31.4s,v6.s[0]                 \n\t" // Scale by alpha
	"                                            \n\t"
	" str q8, [x25]                              \n\t" //Store column 9 of C
	" str q9, [x25, #16]                         \n\t"
	" str q10, [x26]                             \n\t" //Store column 10 of C
	" str q11, [x26, #16]                        \n\t"
	" str q12, [x27]                             \n\t" //Store column 11 of C
	" str q13, [x27, #16]                        \n\t"
	"                                            \n\t"
	"                                            \n\t"
	BRANCH (SEND)                                      // Done.
	"                                            \n\t"
	"                                            \n\t"
	LABEL (SGENSTORED)                                 // C is general-stride stored.
	"                                            \n\t"
	"                                            \n\t"
	" dup  v0.4s, wzr                            \n\t"
	" dup  v1.4s, wzr                            \n\t"
	" dup  v2.4s, wzr                            \n\t"
	" dup  v3.4s, wzr                            \n\t"
	" dup  v4.4s, wzr                            \n\t"
	" dup  v5.4s, wzr                            \n\t"
	"                                            \n\t"
	" fcmp s7,#0.0                               \n\t"
	BEQ (SBETAZEROGENSTOREDS1)                         // Taking care of the beta==0 case.
	"                                            \n\t"
	" mov x5, x2                                 \n\t"
	"                                            \n\t"
	" ld1 {v0.s}[0],[x5],x14                     \n\t" // Load c00  into quad and increment by rs_c.
	" ld1 {v0.s}[1],[x5],x14                     \n\t" // Load c01  into quad and increment by rs_c.
	" ld1 {v0.s}[2],[x5],x14                     \n\t" // Load c02  into quad and increment by rs_c.
	" ld1 {v0.s}[3],[x5],x14                     \n\t" // Load c03  into quad and increment by rs_c.
	" ld1 {v1.s}[0],[x5],x14                     \n\t" // Load c04  into quad and increment by rs_c.
	" ld1 {v1.s}[1],[x5],x14                     \n\t" // Load c05  into quad and increment by rs_c.
	" ld1 {v1.s}[2],[x5],x14                     \n\t" // Load c06  into quad and increment by rs_c.
	" ld1 {v1.s}[3],[x5],x14                     \n\t" // Load c07  into quad and increment by rs_c.
	"                                            \n\t"
	" mov x5, x16                                \n\t"
	"                                            \n\t"
	" ld1 {v2.s}[0],[x5],x14                     \n\t" // Load c10  into quad and increment by rs_c.
	" ld1 {v2.s}[1],[x5],x14                     \n\t" // Load c11  into quad and increment by rs_c.
	" ld1 {v2.s}[2],[x5],x14                     \n\t" // Load c12  into quad and increment by rs_c.
	" ld1 {v2.s}[3],[x5],x14                     \n\t" // Load c13  into quad and increment by rs_c.
	" ld1 {v3.s}[0],[x5],x14                     \n\t" // Load c14  into quad and increment by rs_c.
	" ld1 {v3.s}[1],[x5],x14                     \n\t" // Load c15  into quad and increment by rs_c.
	" ld1 {v3.s}[2],[x5],x14                     \n\t" // Load c16  into quad and increment by rs_c.
	" ld1 {v3.s}[3],[x5],x14                     \n\t" // Load c17  into quad and increment by rs_c.
	"                                            \n\t"
	" mov x5, x17                                \n\t"
	"                                            \n\t"
	" ld1 {v4.s}[0],[x5],x14                     \n\t" // Load c20  into quad and increment by rs_c.
	" ld1 {v4.s}[1],[x5],x14                     \n\t" // Load c21  into quad and increment by rs_c.
	" ld1 {v4.s}[2],[x5],x14                     \n\t" // Load c22  into quad and increment by rs_c.
	" ld1 {v4.s}[3],[x5],x14                     \n\t" // Load c23  into quad and increment by rs_c.
	" ld1 {v5.s}[0],[x5],x14                     \n\t" // Load c24  into quad and increment by rs_c.
	" ld1 {v5.s}[1],[x5],x14                     \n\t" // Load c25  into quad and increment by rs_c.
	" ld1 {v5.s}[2],[x5],x14                     \n\t" // Load c26  into quad and increment by rs_c.
	" ld1 {v5.s}[3],[x5],x14                     \n\t" // Load c27  into quad and increment by rs_c.
	"                                            \n\t"
	" fmul v0.4s,v0.4s,v7.s[0]                   \n\t" // Scale by beta
	" fmul v1.4s,v1.4s,v7.s[0]                   \n\t" // Scale by beta
	" fmul v2.4s,v2.4s,v7.s[0]                   \n\t" // Scale by beta
	" fmul v3.4s,v3.4s,v7.s[0]                   \n\t" // Scale by beta
	" fmul v4.4s,v4.4s,v7.s[0]                   \n\t" // Scale by beta
	" fmul v5.4s,v5.4s,v7.s[0]                   \n\t" // Scale by beta
	"                                            \n\t"
	LABEL (SBETAZEROGENSTOREDS1)
	"                                            \n\t"
	" fmla v0.4s, v8.4s,v6.s[0]                  \n\t" // Scale by alpha
	" fmla v1.4s, v9.4s,v6.s[0]                  \n\t" // Scale by alpha
	" fmla v2.4s,v10.4s,v6.s[0]                  \n\t" // Scale by alpha
	" fmla v3.4s,v11.4s,v6.s[0]                  \n\t" // Scale by alpha
	" fmla v4.4s,v12.4s,v6.s[0]                  \n\t" // Scale by alpha
	" fmla v5.4s,v13.4s,v6.s[0]                  \n\t" // Scale by alpha
	"                                            \n\t"
	" mov x5, x2                                 \n\t"
	"                                            \n\t"
	" st1 {v0.s}[0],[x5],x14                     \n\t" // Store c00  into quad and increment by rs_c.
	" st1 {v0.s}[1],[x5],x14                     \n\t" // Store c01  into quad and increment by rs_c.
	" st1 {v0.s}[2],[x5],x14                     \n\t" // Store c02  into quad and increment by rs_c.
	" st1 {v0.s}[3],[x5],x14                     \n\t" // Store c03  into quad and increment by rs_c.
	" st1 {v1.s}[0],[x5],x14                     \n\t" // Store c04  into quad and increment by rs_c.
	" st1 {v1.s}[1],[x5],x14                     \n\t" // Store c05  into quad and increment by rs_c.
	" st1 {v1.s}[2],[x5],x14                     \n\t" // Store c06  into quad and increment by rs_c.
	" st1 {v1.s}[3],[x5],x14                     \n\t" // Store c07  into quad and increment by rs_c.
	"                                            \n\t"
	" mov x5, x16                                \n\t"
	"                                            \n\t"
	" st1 {v2.s}[0],[x5],x14                     \n\t" // Store c10  into quad and increment by rs_c.
	" st1 {v2.s}[1],[x5],x14                     \n\t" // Store c11  into quad and increment by rs_c.
	" st1 {v2.s}[2],[x5],x14                     \n\t" // Store c12  into quad and increment by rs_c.
	" st1 {v2.s}[3],[x5],x14                     \n\t" // Store c13  into quad and increment by rs_c.
	" st1 {v3.s}[0],[x5],x14                     \n\t" // Store c14  into quad and increment by rs_c.
	" st1 {v3.s}[1],[x5],x14                     \n\t" // Store c15  into quad and increment by rs_c.
	" st1 {v3.s}[2],[x5],x14                     \n\t" // Store c16  into quad and increment by rs_c.
	" st1 {v3.s}[3],[x5],x14                     \n\t" // Store c17  into quad and increment by rs_c.
	"                                            \n\t"
	" mov x5, x17                                \n\t"
	"                                            \n\t"
	" st1 {v4.s}[0],[x5],x14                     \n\t" // Store c20  into quad and increment by rs_c.
	" st1 {v4.s}[1],[x5],x14                     \n\t" // Store c21  into quad and increment by rs_c.
	" st1 {v4.s}[2],[x5],x14                     \n\t" // Store c22  into quad and increment by rs_c.
	" st1 {v4.s}[3],[x5],x14                     \n\t" // Store c23  into quad and increment by rs_c.
	" st1 {v5.s}[0],[x5],x14                     \n\t" // Store c24  into quad and increment by rs_c.
	" st1 {v5.s}[1],[x5],x14                     \n\t" // Store c25  into quad and increment by rs_c.
	" st1 {v5.s}[2],[x5],x14                     \n\t" // Store c26  into quad and increment by rs_c.
	" st1 {v5.s}[3],[x5],x14                     \n\t" // Store c27  into quad and increment by rs_c.
	"                                            \n\t"
	" dup  v8.4s, wzr                            \n\t"
	" dup  v9.4s, wzr                            \n\t"
	" dup  v10.4s, wzr                           \n\t"
	" dup  v11.4s, wzr                           \n\t"
	" dup  v12.4s, wzr                           \n\t"
	" dup  v13.4s, wzr                           \n\t"
	"                                            \n\t"
	" fcmp s7,#0.0                               \n\t"
	BEQ (SBETAZEROGENSTOREDS2)                         // Taking care of the beta==0 case.
	"                                            \n\t"
	" mov x5, x19                                \n\t"
	"                                            \n\t"
	" ld1 {v8.s}[0],[x5],x14                     \n\t" // Load c30  into quad and increment by rs_c.
	" ld1 {v8.s}[1],[x5],x14                     \n\t" // Load c31  into quad and increment by rs_c.
	" ld1 {v8.s}[2],[x5],x14                     \n\t" // Load c32  into quad and increment by rs_c.
	" ld1 {v8.s}[3],[x5],x14                     \n\t" // Load c33  into quad and increment by rs_c.
	" ld1 {v9.s}[0],[x5],x14                     \n\t" // Load c34  into quad and increment by rs_c.
	" ld1 {v9.s}[1],[x5],x14                     \n\t" // Load c35  into quad and increment by rs_c.
	" ld1 {v9.s}[2],[x5],x14                     \n\t" // Load c36  into quad and increment by rs_c.
	" ld1 {v9.s}[3],[x5],x14                     \n\t" // Load c37  into quad and increment by rs_c.
	"                                            \n\t"
	" mov x5, x20                                \n\t"
	"                                            \n\t"
	" ld1 {v10.s}[0],[x5],x14                    \n\t" // Load c40  into quad and increment by rs_c.
	" ld1 {v10.s}[1],[x5],x14                    \n\t" // Load c41  into quad and increment by rs_c.
	" ld1 {v10.s}[2],[x5],x14                    \n\t" // Load c42  into quad and increment by rs_c.
	" ld1 {v10.s}[3],[x5],x14                    \n\t" // Load c43  into quad and increment by rs_c.
	" ld1 {v11.s}[0],[x5],x14                    \n\t" // Load c44  into quad and increment by rs_c.
	" ld1 {v11.s}[1],[x5],x14                    \n\t" // Load c45  into quad and increment by rs_c.
	" ld1 {v11.s}[2],[x5],x14                    \n\t" // Load c46  into quad and increment by rs_c.
	" ld1 {v11.s}[3],[x5],x14                    \n\t" // Load c47  into quad and increment by rs_c.
	"                                            \n\t"
	" mov x5, x21                                \n\t"
	"                                            \n\t"
	" ld1 {v12.s}[0],[x5],x14                    \n\t" // Load c50  into quad and increment by rs_c.
	" ld1 {v12.s}[1],[x5],x14                    \n\t" // Load c51  into quad and increment by rs_c.
	" ld1 {v12.s}[2],[x5],x14                    \n\t" // Load c52  into quad and increment by rs_c.
	" ld1 {v12.s}[3],[x5],x14                    \n\t" // Load c53  into quad and increment by rs_c.
	" ld1 {v13.s}[0],[x5],x14                    \n\t" // Load c54  into quad and increment by rs_c.
	" ld1 {v13.s}[1],[x5],x14                    \n\t" // Load c55  into quad and increment by rs_c.
	" ld1 {v13.s}[2],[x5],x14                    \n\t" // Load c56  into quad and increment by rs_c.
	" ld1 {v13.s}[3],[x5],x14                    \n\t" // Load c57  into quad and increment by rs_c.
	"                                            \n\t"
	" fmul v8.4s, v8.4s, v7.s[0]                 \n\t" // Scale by beta
	" fmul v9.4s, v9.4s, v7.s[0]                 \n\t" // Scale by beta
	" fmul v10.4s,v10.4s,v7.s[0]                 \n\t" // Scale by beta
	" fmul v11.4s,v11.4s,v7.s[0]                 \n\t" // Scale by beta
	" fmul v12.4s,v12.4s,v7.s[0]                 \n\t" // Scale by beta
	" fmul v13.4s,v13.4s,v7.s[0]                 \n\t" // Scale by beta
	"                                            \n\t"
	LABEL (SBETAZEROGENSTOREDS2)
	"                                            \n\t"
	" fmla v8.4s, v14.4s,v6.s[0]                 \n\t" // Scale by alpha
	" fmla v9.4s, v15.4s,v6.s[0]                 \n\t" // Scale by alpha
	" fmla v10.4s,v16.4s,v6.s[0]                 \n\t" // Scale by alpha
	" fmla v11.4s,v17.4s,v6.s[0]                 \n\t" // Scale by alpha
	" fmla v12.4s,v18.4s,v6.s[0]                 \n\t" // Scale by alpha
	" fmla v13.4s,v19.4s,v6.s[0]                 \n\t" // Scale by alpha
	"                                            \n\t"
	" mov x5, x19                                \n\t"
	"                                            \n\t"
	" st1 {v8.s}[0],[x5],x14                     \n\t" // Store c30  into quad and increment by rs_c.
	" st1 {v8.s}[1],[x5],x14                     \n\t" // Store c31  into quad and increment by rs_c.
	" st1 {v8.s}[2],[x5],x14                     \n\t" // Store c32  into quad and increment by rs_c.
	" st1 {v8.s}[3],[x5],x14                     \n\t" // Store c33  into quad and increment by rs_c.
	" st1 {v9.s}[0],[x5],x14                     \n\t" // Store c34  into quad and increment by rs_c.
	" st1 {v9.s}[1],[x5],x14                     \n\t" // Store c35  into quad and increment by rs_c.
	" st1 {v9.s}[2],[x5],x14                     \n\t" // Store c36  into quad and increment by rs_c.
	" st1 {v9.s}[3],[x5],x14                     \n\t" // Store c37  into quad and increment by rs_c.
	"                                            \n\t"
	" mov x5, x20                                \n\t"
	"                                            \n\t"
	" st1 {v10.s}[0],[x5],x14                    \n\t" // Store c40  into quad and increment by rs_c.
	" st1 {v10.s}[1],[x5],x14                    \n\t" // Store c41  into quad and increment by rs_c.
	" st1 {v10.s}[2],[x5],x14                    \n\t" // Store c42  into quad and increment by rs_c.
	" st1 {v10.s}[3],[x5],x14                    \n\t" // Store c43  into quad and increment by rs_c.
	" st1 {v11.s}[0],[x5],x14                    \n\t" // Store c44  into quad and increment by rs_c.
	" st1 {v11.s}[1],[x5],x14                    \n\t" // Store c45  into quad and increment by rs_c.
	" st1 {v11.s}[2],[x5],x14                    \n\t" // Store c46  into quad and increment by rs_c.
	" st1 {v11.s}[3],[x5],x14                    \n\t" // Store c47  into quad and increment by rs_c.
	"                                            \n\t"
	" mov x5, x21                                \n\t"
	"                                            \n\t"
	" st1 {v12.s}[0],[x5],x14                    \n\t" // Store c50  into quad and increment by rs_c.
	" st1 {v12.s}[1],[x5],x14                    \n\t" // Store c51  into quad and increment by rs_c.
	" st1 {v12.s}[2],[x5],x14                    \n\t" // Store c52  into quad and increment by rs_c.
	" st1 {v12.s}[3],[x5],x14                    \n\t" // Store c53  into quad and increment by rs_c.
	" st1 {v13.s}[0],[x5],x14                    \n\t" // Store c54  into quad and increment by rs_c.
	" st1 {v13.s}[1],[x5],x14                    \n\t" // Store c55  into quad and increment by rs_c.
	" st1 {v13.s}[2],[x5],x14                    \n\t" // Store c56  into quad and increment by rs_c.
	" st1 {v13.s}[3],[x5],x14                    \n\t" // Store c57  into quad and increment by rs_c.
	"                                            \n\t"
	" dup  v0.4s, wzr                            \n\t"
	" dup  v1.4s, wzr                            \n\t"
	" dup  v2.4s, wzr                            \n\t"
	" dup  v3.4s, wzr                            \n\t"
	" dup  v4.4s, wzr                            \n\t"
	" dup  v5.4s, wzr                            \n\t"
	"                                            \n\t"
	" fcmp s7,#0.0                               \n\t"
	BEQ (SBETAZEROGENSTOREDS3)                         // Taking care of the beta==0 case.
	"                                            \n\t"
	" mov x5, x22                                \n\t"
	"                                            \n\t"
	" ld1 {v0.s}[0],[x5],x14                     \n\t" // Load c60  into quad and increment by rs_c.
	" ld1 {v0.s}[1],[x5],x14                     \n\t" // Load c61  into quad and increment by rs_c.
	" ld1 {v0.s}[2],[x5],x14                     \n\t" // Load c62  into quad and increment by rs_c.
	" ld1 {v0.s}[3],[x5],x14                     \n\t" // Load c63  into quad and increment by rs_c.
	" ld1 {v1.s}[0],[x5],x14                     \n\t" // Load c64  into quad and increment by rs_c.
	" ld1 {v1.s}[1],[x5],x14                     \n\t" // Load c65  into quad and increment by rs_c.
	" ld1 {v1.s}[2],[x5],x14                     \n\t" // Load c66  into quad and increment by rs_c.
	" ld1 {v1.s}[3],[x5],x14                     \n\t" // Load c67  into quad and increment by rs_c.
	"                                            \n\t"
	" mov x5, x23                                \n\t"
	"                                            \n\t"
	" ld1 {v2.s}[0],[x5],x14                     \n\t" // Load c70  into quad and increment by rs_c.
	" ld1 {v2.s}[1],[x5],x14                     \n\t" // Load c71  into quad and increment by rs_c.
	" ld1 {v2.s}[2],[x5],x14                     \n\t" // Load c72  into quad and increment by rs_c.
	" ld1 {v2.s}[3],[x5],x14                     \n\t" // Load c73  into quad and increment by rs_c.
	" ld1 {v3.s}[0],[x5],x14                     \n\t" // Load c74  into quad and increment by rs_c.
	" ld1 {v3.s}[1],[x5],x14                     \n\t" // Load c75  into quad and increment by rs_c.
	" ld1 {v3.s}[2],[x5],x14                     \n\t" // Load c76  into quad and increment by rs_c.
	" ld1 {v3.s}[3],[x5],x14                     \n\t" // Load c77  into quad and increment by rs_c.
	"                                            \n\t"
	" mov x5, x24                                \n\t"
	"                                            \n\t"
	" ld1 {v4.s}[0],[x5],x14                     \n\t" // Load c80  into quad and increment by rs_c.
	" ld1 {v4.s}[1],[x5],x14                     \n\t" // Load c81  into quad and increment by rs_c.
	" ld1 {v4.s}[2],[x5],x14                     \n\t" // Load c82  into quad and increment by rs_c.
	" ld1 {v4.s}[3],[x5],x14                     \n\t" // Load c83  into quad and increment by rs_c.
	" ld1 {v5.s}[0],[x5],x14                     \n\t" // Load c84  into quad and increment by rs_c.
	" ld1 {v5.s}[1],[x5],x14                     \n\t" // Load c85  into quad and increment by rs_c.
	" ld1 {v5.s}[2],[x5],x14                     \n\t" // Load c86  into quad and increment by rs_c.
	" ld1 {v5.s}[3],[x5],x14                     \n\t" // Load c87  into quad and increment by rs_c.
	"                                            \n\t"
	" fmul v0.4s,v0.4s,v7.s[0]                   \n\t" // Scale by beta
	" fmul v1.4s,v1.4s,v7.s[0]                   \n\t" // Scale by beta
	" fmul v2.4s,v2.4s,v7.s[0]                   \n\t" // Scale by beta
	" fmul v3.4s,v3.4s,v7.s[0]                   \n\t" // Scale by beta
	" fmul v4.4s,v4.4s,v7.s[0]                   \n\t" // Scale by beta
	" fmul v5.4s,v5.4s,v7.s[0]                   \n\t" // Scale by beta
	"                                            \n\t"
	LABEL (SBETAZEROGENSTOREDS3)
	"                                            \n\t"
	" fmla v0.4s,v20.4s,v6.s[0]                  \n\t" // Scale by alpha
	" fmla v1.4s,v21.4s,v6.s[0]                  \n\t" // Scale by alpha
	" fmla v2.4s,v22.4s,v6.s[0]                  \n\t" // Scale by alpha
	" fmla v3.4s,v23.4s,v6.s[0]                  \n\t" // Scale by alpha
	" fmla v4.4s,v24.4s,v6.s[0]                  \n\t" // Scale by alpha
	" fmla v5.4s,v25.4s,v6.s[0]                  \n\t" // Scale by alpha
	"                                            \n\t"
	" mov x5, x22                                \n\t"
	"                                            \n\t"
	" st1 {v0.s}[0],[x5],x14                     \n\t" // Store c60  into quad and increment by rs_c.
	" st1 {v0.s}[1],[x5],x14                     \n\t" // Store c61  into quad and increment by rs_c.
	" st1 {v0.s}[2],[x5],x14                     \n\t" // Store c62  into quad and increment by rs_c.
	" st1 {v0.s}[3],[x5],x14                     \n\t" // Store c63  into quad and increment by rs_c.
	" st1 {v1.s}[0],[x5],x14                     \n\t" // Store c64  into quad and increment by rs_c.
	" st1 {v1.s}[1],[x5],x14                     \n\t" // Store c65  into quad and increment by rs_c.
	" st1 {v1.s}[2],[x5],x14                     \n\t" // Store c66  into quad and increment by rs_c.
	" st1 {v1.s}[3],[x5],x14                     \n\t" // Store c67  into quad and increment by rs_c.
	"                                            \n\t"
	" mov x5, x23                                \n\t"
	"                                            \n\t"
	" st1 {v2.s}[0],[x5],x14                     \n\t" // Store c70  into quad and increment by rs_c.
	" st1 {v2.s}[1],[x5],x14                     \n\t" // Store c71  into quad and increment by rs_c.
	" st1 {v2.s}[2],[x5],x14                     \n\t" // Store c72  into quad and increment by rs_c.
	" st1 {v2.s}[3],[x5],x14                     \n\t" // Store c73  into quad and increment by rs_c.
	" st1 {v3.s}[0],[x5],x14                     \n\t" // Store c74  into quad and increment by rs_c.
	" st1 {v3.s}[1],[x5],x14                     \n\t" // Store c75  into quad and increment by rs_c.
	" st1 {v3.s}[2],[x5],x14                     \n\t" // Store c76  into quad and increment by rs_c.
	" st1 {v3.s}[3],[x5],x14                     \n\t" // Store c77  into quad and increment by rs_c.
	"                                            \n\t"
	" mov x5, x24                                \n\t"
	"                                            \n\t"
	" st1 {v4.s}[0],[x5],x14                     \n\t" // Store c80  into quad and increment by rs_c.
	" st1 {v4.s}[1],[x5],x14                     \n\t" // Store c81  into quad and increment by rs_c.
	" st1 {v4.s}[2],[x5],x14                     \n\t" // Store c82  into quad and increment by rs_c.
	" st1 {v4.s}[3],[x5],x14                     \n\t" // Store c83  into quad and increment by rs_c.
	" st1 {v5.s}[0],[x5],x14                     \n\t" // Store c84  into quad and increment by rs_c.
	" st1 {v5.s}[1],[x5],x14                     \n\t" // Store c85  into quad and increment by rs_c.
	" st1 {v5.s}[2],[x5],x14                     \n\t" // Store c86  into quad and increment by rs_c.
	" st1 {v5.s}[3],[x5],x14                     \n\t" // Store c87  into quad and increment by rs_c.
	"                                            \n\t"
	" dup  v8.4s, wzr                            \n\t"
	" dup  v9.4s, wzr                            \n\t"
	" dup  v10.4s, wzr                           \n\t"
	" dup  v11.4s, wzr                           \n\t"
	" dup  v12.4s, wzr                           \n\t"
	" dup  v13.4s, wzr                           \n\t"
	"                                            \n\t"
	" fcmp s7,#0.0                               \n\t"
	BEQ (SBETAZEROGENSTOREDS4)                         // Taking care of the beta==0 case.
	"                                            \n\t"
	" mov x5, x25                                \n\t"
	"                                            \n\t"
	" ld1 {v8.s}[0],[x5],x14                     \n\t" // Load c90  into quad and increment by rs_c.
	" ld1 {v8.s}[1],[x5],x14                     \n\t" // Load c91  into quad and increment by rs_c.
	" ld1 {v8.s}[2],[x5],x14                     \n\t" // Load c92  into quad and increment by rs_c.
	" ld1 {v8.s}[3],[x5],x14                     \n\t" // Load c93  into quad and increment by rs_c.
	" ld1 {v9.s}[0],[x5],x14                     \n\t" // Load c94  into quad and increment by rs_c.
	" ld1 {v9.s}[1],[x5],x14                     \n\t" // Load c95  into quad and increment by rs_c.
	" ld1 {v9.s}[2],[x5],x14                     \n\t" // Load c96  into quad and increment by rs_c.
	" ld1 {v9.s}[3],[x5],x14                     \n\t" // Load c97  into quad and increment by rs_c.
	"                                            \n\t"
	" mov x5, x26                                \n\t"
	"                                            \n\t"
	" ld1 {v10.s}[0],[x5],x14                    \n\t" // Load c100  into quad and increment by rs_c.
	" ld1 {v10.s}[1],[x5],x14                    \n\t" // Load c101  into quad and increment by rs_c.
	" ld1 {v10.s}[2],[x5],x14                    \n\t" // Load c102  into quad and increment by rs_c.
	" ld1 {v10.s}[3],[x5],x14                    \n\t" // Load c103  into quad and increment by rs_c.
	" ld1 {v11.s}[0],[x5],x14                    \n\t" // Load c104  into quad and increment by rs_c.
	" ld1 {v11.s}[1],[x5],x14                    \n\t" // Load c105  into quad and increment by rs_c.
	" ld1 {v11.s}[2],[x5],x14                    \n\t" // Load c106  into quad and increment by rs_c.
	" ld1 {v11.s}[3],[x5],x14                    \n\t" // Load c107  into quad and increment by rs_c.
	"                                            \n\t"
	" mov x5, x27                                \n\t"
	"                                            \n\t"
	" ld1 {v12.s}[0],[x5],x14                    \n\t" // Load c110  into quad and increment by rs_c.
	" ld1 {v12.s}[1],[x5],x14                    \n\t" // Load c111  into quad and increment by rs_c.
	" ld1 {v12.s}[2],[x5],x14                    \n\t" // Load c112  into quad and increment by rs_c.
	" ld1 {v12.s}[3],[x5],x14                    \n\t" // Load c113  into quad and increment by rs_c.
	" ld1 {v13.s}[0],[x5],x14                    \n\t" // Load c114  into quad and increment by rs_c.
	" ld1 {v13.s}[1],[x5],x14                    \n\t" // Load c115  into quad and increment by rs_c.
	" ld1 {v13.s}[2],[x5],x14                    \n\t" // Load c116  into quad and increment by rs_c.
	" ld1 {v13.s}[3],[x5],x14                    \n\t" // Load c117  into quad and increment by rs_c.
	"                                            \n\t"
	" fmul v8.4s, v8.4s, v7.s[0]                 \n\t" // Scale by beta
	" fmul v9.4s, v9.4s, v7.s[0]                 \n\t" // Scale by beta
	" fmul v10.4s,v10.4s,v7.s[0]                 \n\t" // Scale by beta
	" fmul v11.4s,v11.4s,v7.s[0]                 \n\t" // Scale by beta
	" fmul v12.4s,v12.4s,v7.s[0]                 \n\t" // Scale by beta
	" fmul v13.4s,v13.4s,v7.s[0]                 \n\t" // Scale by beta
	"                                            \n\t"
	LABEL (SBETAZEROGENSTOREDS4)
	"                                            \n\t"
	"                                            \n\t"
	" fmla v8.4s, v26.4s,v6.s[0]                 \n\t" // Scale by alpha
	" fmla v9.4s, v27.4s,v6.s[0]                 \n\t" // Scale by alpha
	" fmla v10.4s,v28.4s,v6.s[0]                 \n\t" // Scale by alpha
	" fmla v11.4s,v29.4s,v6.s[0]                 \n\t" // Scale by alpha
	" fmla v12.4s,v30.4s,v6.s[0]                 \n\t" // Scale by alpha
	" fmla v13.4s,v31.4s,v6.s[0]                 \n\t" // Scale by alpha
	"                                            \n\t"
	" mov x5, x25                                \n\t"
	"                                            \n\t"
	" st1 {v8.s}[0],[x5],x14                     \n\t" // Store c90  into quad and increment by rs_c.
	" st1 {v8.s}[1],[x5],x14                     \n\t" // Store c91  into quad and increment by rs_c.
	" st1 {v8.s}[2],[x5],x14                     \n\t" // Store c92  into quad and increment by rs_c.
	" st1 {v8.s}[3],[x5],x14                     \n\t" // Store c93  into quad and increment by rs_c.
	" st1 {v9.s}[0],[x5],x14                     \n\t" // Store c94  into quad and increment by rs_c.
	" st1 {v9.s}[1],[x5],x14                     \n\t" // Store c95  into quad and increment by rs_c.
	" st1 {v9.s}[2],[x5],x14                     \n\t" // Store c96  into quad and increment by rs_c.
	" st1 {v9.s}[3],[x5],x14                     \n\t" // Store c97  into quad and increment by rs_c.
	"                                            \n\t"
	" mov x5, x26                                \n\t"
	"                                            \n\t"
	" st1 {v10.s}[0],[x5],x14                    \n\t" // Store c100  into quad and increment by rs_c.
	" st1 {v10.s}[1],[x5],x14                    \n\t" // Store c101  into quad and increment by rs_c.
	" st1 {v10.s}[2],[x5],x14                    \n\t" // Store c102  into quad and increment by rs_c.
	" st1 {v10.s}[3],[x5],x14                    \n\t" // Store c103  into quad and increment by rs_c.
	" st1 {v11.s}[0],[x5],x14                    \n\t" // Store c104  into quad and increment by rs_c.
	" st1 {v11.s}[1],[x5],x14                    \n\t" // Store c105  into quad and increment by rs_c.
	" st1 {v11.s}[2],[x5],x14                    \n\t" // Store c106  into quad and increment by rs_c.
	" st1 {v11.s}[3],[x5],x14                    \n\t" // Store c107  into quad and increment by rs_c.
	"                                            \n\t"
	" mov x5, x27                                \n\t"
	"                                            \n\t"
	" st1 {v12.s}[0],[x5],x14                    \n\t" // Store c110  into quad and increment by rs_c.
	" st1 {v12.s}[1],[x5],x14                    \n\t" // Store c111  into quad and increment by rs_c.
	" st1 {v12.s}[2],[x5],x14                    \n\t" // Store c112  into quad and increment by rs_c.
	" st1 {v12.s}[3],[x5],x14                    \n\t" // Store c113  into quad and increment by rs_c.
	" st1 {v13.s}[0],[x5],x14                    \n\t" // Store c114  into quad and increment by rs_c.
	" st1 {v13.s}[1],[x5],x14                    \n\t" // Store c115  into quad and increment by rs_c.
	" st1 {v13.s}[2],[x5],x14                    \n\t" // Store c116  into quad and increment by rs_c.
	" st1 {v13.s}[3],[x5],x14                    \n\t" // Store c147  into quad and increment by rs_c.
	"                                            \n\t"
	LABEL (SEND)                                       // Done!
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

	GEMM_UKR_FLUSH_CT( s );
}