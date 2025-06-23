# Objetivo

El objetivo del proyecto es realizar un *port* de la biblioteca de álgebra lineal BLIS (https://github.com/flame/blis) al acelerador AMX de Apple, programable a partir de la generación M4 a través de la extensión SME *(Scalable Matrix Extension*) de ARM.
# Máquina de trabajo

Máquina de trabajo (Apple M4). Con acceso físico (solicítanos tarjeta de acceso). Acceso remoto a través de:

```sh
ssh remoto.dacya.ucm.es -p 16998 -l USUARIO
```

Te pasaremos el usuario por otra vía. Puedes utilizar cualquier emulador de terminal de Linux (todos tienen el cliente `ssh` instalado, o bien putty o similar en Windows).
# Primeros pasos

## Compilación de BLIS para Cortex-A57

La configuración más cercana al Apple M4 de la que dispone BLIS se denomina cortex-a57. Hace uso de las instrucciones SIMD NEON de Apple, y por tanto puede vectorizar código utilizando las unidades SIMD que existen en cada core del M4. Este no es nuestro objetivo final, pero servirá para hacernos una idea del proceso de compilación de blis, del funcionamiento de la testsuite y del nivel de rendimiento que podemos obtener en la CPU de esta máquina.

```sh
git clone https://github.com/flame/blis
cd blis
./configure cortexa57
make -j
...
...
Archiving lib/cortexa57/libblis.a
Dynamically linking lib/cortexa57/libblis.dylib
Creating symlink lib/cortexa57/libblis.4.dylib
```

Con esto, tenemos BLIS listo para utilizar desde cualquier binario. BLIS proporciona una testsuite para comprobar que la funcionalidad es correcta, y también para evaluar el rendimiento de cada una de las rutinas que componen la biblioteca, con todas sus variantes.

Esta testsuite reside en el directorio `testsuite` del código fuente. Podemos compilarla:

```sh
cd testsuite
make -j
```

Y ejecutarla:

```sh
./test_libblis.x
```
Verás que la salida es muy extensa. Básicamente, estamos testeando todas las rutinas que componen BLIS (cualquier biblioteca BLAS, realmente), para distintas variantes, tipos de operandos, tipos de datos, dimensiones, etc.

El número de tests que se ejecutan viene dado por dos ficheros:

- `input.general`, con generalidades sobre la ejecución de los tests, comunes a todos.
- `input.operations`, seleccionando qué operaciones concretas se testearán.

En nuestro caso, nos interesa el producto de matrices para simple y doble precisión para tamaños de matriz relativamente grandes, por lo que haremos los siguientes cambios en ambos ficheros:

- En `input.general`, nos quedaremos con esto (analiza el contenido tú mismo, es muy sencillo. Básicamente limitamos a almacenamiento de datos por columnas, dimensiones de 100 a 5000, en saltos de 400 en 400, para simple y doble precisión):
```sh
# ----------------------------------------------------------------------
#
#  input.general
#  BLIS test suite
#
#  This file contains input values that control how BLIS operations are
#  tested. Comments explain the purpose of each parameter as well as
#  accepted values.
#

1       # Number of repeats per experiment (best result is reported)
c       # Matrix storage scheme(s) to test:
        #   'c' = col-major storage; 'g' = general stride storage;
        #   'r' = row-major storage
cj      # Vector storage scheme(s) to test:
        #   'c' = colvec / unit stride; 'j' = colvec / non-unit stride;
        #   'r' = rowvec / unit stride; 'i' = rowvec / non-unit stride
0       # Test all combinations of storage schemes?
1       # Perform all tests with alignment?
        #   '0' = do NOT align buffers/ldims; '1' = align buffers/ldims
0       # Randomize vectors and matrices using:
        #   '0' = real values on [-1,1];
        #   '1' = powers of 2 in narrow precision range
32      # General stride spacing (for cases when testing general stride)
sd      # Datatype(s) to test:
        #   's' = single real; 'c' = single complex;
        #   'd' = double real; 'z' = double complex
0       # Test gemm with mixed-domain operands?
0       # Test gemm with mixed-precision operands?
100     # Problem size: first to test
5000    # Problem size: maximum to test
400     # Problem size: increment between experiments
        # Complex level-3 implementations to test:
1       #   1m   ('1' = enable; '0' = disable)
1       #   native ('1' = enable; '0' = disable)
1       # Simulate application-level threading:
        #   '1' = disable / use one testsuite thread;
        #   'n' = enable and use n testsuite threads
1       # Error-checking level:
        #   '0' = disable error checking; '1' = full error checking
i       # Reaction to test failure:
        #   'i' = ignore; 's' = sleep() and continue; 'a' = abort
0       # Output results in matlab/octave format? ('1' = yes; '0' = no)
0       # Output results to stdout AND files? ('1' = yes; '0' = no)
```
- En `input.operations`, seleccionaremos sólo el producto de matrices (GEMM) para testear. Aquí hay más cambios, pero básicamente debes seleccionar sólo evaluar en nivel 3 de BLAS (rutinas matriz-matriz, donde se encuentra GEMM):
```sh
# --- Section overrides ----------------------------------------------------

0        # Utility
0        # Level-1v kernels
0        # Level-1m
0        # Level-1f kernels
0        # Level-2
0        # Level-3 micro-kernels
1        # Level-3
```
     Y a continuación solamente quedarte con GEMM, utilizando matrices sin transponer (parte final del fichero):
```sh
     # --- Level-3 --------------------------------------------------------------

1        # gemm
-1 -1 -1 #   dimensions: m n k
nn       #   parameters: transa transb

0        # gemmt
-1 -1    #   dimensions: m k
???      #   parameters: uploc transa transb

0        # hemm
-1 -1    #   dimensions: m n
????     #   parameters: side uploa conja transb

0        # herk
-1 -1    #   dimensions: m k
??       #   parameters: uploc transa

0        # her2k
-1 -1    #   dimensions: m k
???      #   parameters: uploc transa transb

0        # symm
-1 -1    #   dimensions: m n
????     #   parameters: side uploa conja transb

0        # syrk
-1 -1    #   dimensions: m k
??       #   parameters: uploc transa

0        # syr2k
-1 -1    #   dimensions: m k
???      #   parameters: uploc transa transb

0        # trmm
-1 -1    #   dimensions: m n
????     #   parameters: side uploa transa diaga

0        # trmm3
-1 -1    #   dimensions: m n
????n    #   parameters: side uploa transa diaga transb

0        # trsm
-1 -1    #   dimensions: m n
????     #   parameters: side uploa transa diaga
```

Con esto, estás listo para ejecutar la testsuite:

```sh
figual@MacMiniM4pro testsuite % ./test_libblis.x
% no -g option given; defaulting to "input.general" for parameters filename.
% no -o option given; defaulting to "input.operations" for operations filename.
%
% --- BLIS library info -------------------------------------
%
% version string                 0.9.0-184
%
% --- BLIS configuration info ---
%
% active sub-configuration       cortexa57
%
% BLIS integer type size (bits)  64
%
% Assumed max # of SIMD regs     32
% SIMD size (bytes)              64
% SIMD alignment (bytes)         16
% Max stack buffer size (bytes)  8192
% Page size (bytes)              4096
%
% memory pools
%   enabled for packing blocks?  1
%   enabled for small blocks?    1
%
% memory alignment (bytes)
%   stack address                16
%   obj_t address                16
%   obj_t stride                 16
%   pool block addr A (+offset)  4096 (+0)
%   pool block addr B (+offset)  4096 (+0)
%
% BLAS/CBLAS compatibility layers
%   BLAS API enabled?            1
%   CBLAS API enabled?           0
%   integer type size (bits)     32
%
% libmemkind
%   enabled?                     0
%
% gemm sandbox
%   enabled?                     0
%
% floating-point types           s       d       c       z
%   sizes (bytes)                4       8       8      16
%
%
% --- BLIS parallelization info ---
%
% thread-local storage (TLS)     1
%
% multithreading modes           single only
%   default mode                 single
%   current mode                 single
%
% thread auto-factorization
%   m dim thread ratio           1
%   n dim thread ratio           1
%   jr max threads               4
%   ir max threads               1
%
% ways of parallelism     nt    jc    pc    ic    jr    ir
%   environment            1     1     1     1     1     1
%
% thread partitioning
%   jr/ir loops                  slab
%
%
% --- BLIS default implementations ---
%
% level-3 implementations        s       d       c       z
%   gemm                    native  native      1m      1m
%   hemm                    native  native      1m      1m
%   herk                    native  native      1m      1m
%   her2k                   native  native      1m      1m
%   symm                    native  native      1m      1m
%   syrk                    native  native      1m      1m
%   syr2k                   native  native      1m      1m
%   trmm                    native  native      1m      1m
%   trmm3                   native  native      1m      1m
%   trsm                    native  native      1m      1m
%
%
% --- BLIS native implementation info ---
%
%                                                c       z
% complex implementation                    native  native
%
% level-3 blocksizes             s       d       c       z
%   mc                         120     120     128      64
%   kc                         640     240     256     256
%   nc                        3072    3072    4096    4096
%
%   mc maximum                 120     120     128      64
%   kc maximum                 640     240     256     256
%   nc maximum                3072    3072    4096    4096
%
%   mr                           8       6       4       4
%   nr                          12       8       8       4
%
%   mr packdim                   8       6       4       4
%   nr packdim                  12       8       8       4
%
% micro-kernel types             s       d       c       z
%   gemm                   optimzd optimzd refrnce refrnce
%   gemmtrsm_l             refrnce refrnce refrnce refrnce
%   gemmtrsm_u             refrnce refrnce refrnce refrnce
%   trsm_l                 refrnce refrnce refrnce refrnce
%   trsm_u                 refrnce refrnce refrnce refrnce
%
%
% micro-kernel prefers rows?     s       d       c       z
%   gemm                         0       0       1       1
%   gemmtrsm_l                   0       0       0       0
%   gemmtrsm_u                   0       0       0       0
%   trsm_l                       0       0       0       0
%   trsm_u                       0       0       0       0
%
%
%
% --- BLIS misc. other info ---
%
% level-2 cache blocksizes       s       d       c       z
%   m dimension               1000    1000    1000    1000
%   n dimension               1000    1000    1000    1000
%
% level-1f fusing factors        s       d       c       z
%   axpyf                        8       8       8       8
%   dotxf                        6       6       6       6
%   dotxaxpyf                    4       4       4       4
%

%
% --- BLIS test suite parameters ----------------------------
%
% num repeats per experiment   1
% num matrix storage schemes   1
% storage[ matrix ]            c
% num vector storage schemes   2
% storage[ vector ]            cj
% mix all storage schemes?     0
% test with aligned memory?    1
% randomization method         0
% general stride spacing       32
% num datatypes                2
% datatype[0]                  0 (s)
%         [1]                  2 (d)
% mix domains for gemm?        0
% mix precisions for gemm?     0
% problem size: first to test  100
% problem size: max to test    5000
% problem size increment       400
% complex implementations
%   1m?                        1
%   native?                    1
% simulated app-level threads  1
% error-checking level         1
% reaction to failure          i
% output in matlab format?     0
% output to stdout AND files?  0
%

%
% --- Section overrides ---
%
% Utility operations           0
% Level-1v operations          0
% Level-1m operations          0
% Level-1f operations          0
% Level-2 operations           0
% Level-3 micro-kernels        0
% Level-3 operations           1
%

% --- gemm ---
%
% gemm m n k                  -1 -1 -1
% gemm operand params         nn
%

% blis_<dt><op>_<params>_<stor>            m     n     k   gflops   resid      result
blis_sgemm_nn_ccc                        100   100   100     8.01   2.23e-08   PASS
blis_sgemm_nn_ccc                        500   500   500    50.78   1.34e-08   PASS
blis_sgemm_nn_ccc                        900   900   900   113.81   1.73e-08   PASS
blis_sgemm_nn_ccc                       1300  1300  1300   130.73   1.44e-08   PASS
blis_sgemm_nn_ccc                       1700  1700  1700   131.20   2.07e-08   PASS
blis_sgemm_nn_ccc                       2100  2100  2100   132.49   1.41e-08   PASS
blis_sgemm_nn_ccc                       2500  2500  2500   132.10   1.87e-08   PASS
blis_sgemm_nn_ccc                       2900  2900  2900   132.30   2.52e-08   PASS
blis_sgemm_nn_ccc                       3300  3300  3300   131.59   1.53e-08   PASS
blis_sgemm_nn_ccc                       3700  3700  3700   131.41   1.74e-08   PASS
blis_sgemm_nn_ccc                       4100  4100  4100   131.51   1.04e-08   PASS
blis_sgemm_nn_ccc                       4500  4500  4500   131.70   1.19e-08   PASS
blis_sgemm_nn_ccc                       4900  4900  4900   131.57   1.33e-08   PASS

% blis_<dt><op>_<params>_<stor>            m     n     k   gflops   resid      result
blis_dgemm_nn_ccc                        100   100   100    44.73   4.05e-17   PASS
blis_dgemm_nn_ccc                        500   500   500    62.84   2.66e-17   PASS
blis_dgemm_nn_ccc                        900   900   900    63.47   3.08e-17   PASS
blis_dgemm_nn_ccc                       1300  1300  1300    63.01   2.62e-17   PASS
blis_dgemm_nn_ccc                       1700  1700  1700    62.90   4.04e-17   PASS
blis_dgemm_nn_ccc                       2100  2100  2100    63.44   2.80e-17   PASS
blis_dgemm_nn_ccc                       2500  2500  2500    62.96   3.65e-17   PASS
blis_dgemm_nn_ccc                       2900  2900  2900    62.56   4.59e-17   PASS
blis_dgemm_nn_ccc                       3300  3300  3300    62.87   2.64e-17   PASS
blis_dgemm_nn_ccc                       3700  3700  3700    62.81   3.23e-17   PASS
blis_dgemm_nn_ccc                       4100  4100  4100    62.93   1.92e-17   PASS
blis_dgemm_nn_ccc                       4500  4500  4500    63.11   2.17e-17   PASS
blis_dgemm_nn_ccc                       4900  4900  4900    62.58   2.46e-17   PASS

%
% Exiting normally.
%
figual@MacMiniM4pro testsuite % 
```

Como ves, la salida te reporta rendimiento (en términos de GFLOPS) para SGEMM (simple precisión) y DGEMM (doble precisión), junto con la corrección del test y el residuo obtenido en el chequeo (que debería ser muy pequeño, del orden de 1e-7 para simple precisión, 1e-16 para doble).
# Links de interés

- BLIS (https://github.com/flame/blis)
- BLIS wiki (https://github.com/flame/blis/wiki), específicamente las secciones dedicadas a crear una nueva configuración (lo haremos para SME) y diseño de micro-kernels (más sobre esto cuando hayas compilado y probado la biblioteca).