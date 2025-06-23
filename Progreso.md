# Progreso semanal

## Semana 9 Junio

### Adaptar a lo de Fran

Esta es la primera semana en el proyecto. Seguí las instrucciones de Fran apra configurar todo. Fui capaz de ejecutar la testsuite con la nueva arquitectura pero sin optimizaciones. El resultado fue este: 

```sh
dlopez@MacMiniM4pro testsuite % ./test_libblis.x

...
...

% blis_<dt><op>_<params>_<stor>            m     n     k   gflops   resid      result
blis_sgemm_nn_ccc                        100   100   100     5.44   2.32e-08   PASS
blis_sgemm_nn_ccc                        500   500   500    39.60   1.32e-08   PASS
blis_sgemm_nn_ccc                        900   900   900    59.66   1.74e-08   PASS
blis_sgemm_nn_ccc                       1300  1300  1300    71.26   1.47e-08   PASS
blis_sgemm_nn_ccc                       1700  1700  1700    80.10   2.10e-08   PASS
blis_sgemm_nn_ccc                       2100  2100  2100    84.77   1.41e-08   PASS
blis_sgemm_nn_ccc                       2500  2500  2500    91.08   1.85e-08   PASS
blis_sgemm_nn_ccc                       2900  2900  2900    94.95   2.48e-08   PASS
blis_sgemm_nn_ccc                       3300  3300  3300    87.28   1.51e-08   PASS
blis_sgemm_nn_ccc                       3700  3700  3700    90.30   1.73e-08   PASS
blis_sgemm_nn_ccc                       4100  4100  4100    91.88   1.03e-08   PASS
blis_sgemm_nn_ccc                       4500  4500  4500    94.86   1.20e-08   PASS
blis_sgemm_nn_ccc                       4900  4900  4900    96.94   1.32e-08   PASS

% blis_<dt><op>_<params>_<stor>            m     n     k   gflops   resid      result
blis_dgemm_nn_ccc                        100   100   100     8.89   4.12e-17   PASS
blis_dgemm_nn_ccc                        500   500   500    29.37   2.63e-17   PASS
blis_dgemm_nn_ccc                        900   900   900    38.45   3.15e-17   PASS
blis_dgemm_nn_ccc                       1300  1300  1300    43.10   2.63e-17   PASS
blis_dgemm_nn_ccc                       1700  1700  1700    46.32   4.04e-17   PASS
blis_dgemm_nn_ccc                       2100  2100  2100    48.99   2.84e-17   PASS
blis_dgemm_nn_ccc                       2500  2500  2500    50.37   3.69e-17   PASS
blis_dgemm_nn_ccc                       2900  2900  2900    51.32   4.61e-17   PASS
blis_dgemm_nn_ccc                       3300  3300  3300    49.23   2.65e-17   PASS
blis_dgemm_nn_ccc                       3700  3700  3700    50.13   3.22e-17   PASS
blis_dgemm_nn_ccc                       4100  4100  4100    50.55   1.91e-17   PASS
blis_dgemm_nn_ccc                       4500  4500  4500    51.76   2.17e-17   PASS
blis_dgemm_nn_ccc                       4900  4900  4900    52.02   2.46e-17   PASS
```

Como vemos obtenemos menores GFLOPS al bajar de ~130 a ~95 en simple precisión (-27%) y de ~62 a ~52 en el del doble (-16%).

### Cambio de configuracion para activar las instrucciones del modo Streaming SVE

Cambiando el código del cortexa57 para que cada vez que se mete al microkernel ejecute las instrucciones `smstart` y `smstop` para ver el impacto que este tipo de cambio tiene vemos como pasan los GFLOPS de ~95 a ~76 (-20%) y de ~52 a ~41 (-21%), por lo que igual si conviene hacerlo en mas arriba en y no en el microkernel.
