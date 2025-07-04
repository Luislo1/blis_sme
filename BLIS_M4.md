# BLIS_M4

## Registro ZA y parámetros de BLIS

El registro ZA se compone de N vectores de tamaño VL (Vector Length), siendo N el numero de bytes de VL (i.e. VL = 64 bits -> N = 8 vectores). En el caso del M4 de Apple VL = 512 bits por lo tanto Z4 es de 64 vectores de 64 bytes (512 bits) cada uno. Si tratamos con floats de simple precisión (4 bytes) nos quedaría una matriz 16 x 64 y si fuera con doble precisión sería de 8 x 64, frente a los 8x12 y 6x12 previos respectivamente.

## Pasos del pseudocódigo

El proceso del pseudocódigo quedaría de la siguiente forma:
1. Cargar C en ZA, ya que pasos previos pueden haber escrito ya sobre ellos. Todavía por investigar como funciona la carga y escritura de este registro. 
2. Cargar A y B en los registros. A siempre irá en Z0, pero B irá en los registros Z1-4 en caso de tratarse de s y Z1-8 en caso de tratarse de d.
3. Realizar las operaciones `FMOPA` habiendo tantas como registros necesarios para almacenar B. Esto es debido a que vamos rellenando de forma cuadrada y no todo el registro se llenará y las sucesibas llamadas irán haciendo que se actualice la parte restante. Tomando como ejemplo los flotantes simples hemos dicho que se queda una matriz 16x64. La primera operación `FMOPA` tendrá como resultado el primer cuarto de ZA (`ZA[0:15][0:15]`), la siguiente operacion el segundo cuarto (`ZA[0:15][16:31]`), y así hasta el final.
4. Guardar ZA en C. 

## Observaciones y posibles optimizaciones

- Habiendo estudiado el comportamiento me he dado cuenta de que si entre cada llamada al kernel-3 fijamos el tramo del B y vamos iterando hasta el final del bloque de A, estaríamos operando sobre la misma región de ZA pudiendo ahorranos traer de nuevo cada invocación del kernel-3 tanto ZA, como los registros de B y solo cargaríamos cada vez que fuera necesario cambiar al siguiente.

- En la línea del comentario anterior como no podemos aumentar el ancho de los registros de A al estar limitados por el ancho del registro ZA, podríamos aumentar la profundidad de A al tener mayor cantidad de registros Z libres al no ser usados para almacenar C, de esta forma podríamos hacer más de una llamada previa al kernel-3 pudiendo hacer desenrollado o buscando optimizaciones para ahorrarnos las cargas innecesarias.