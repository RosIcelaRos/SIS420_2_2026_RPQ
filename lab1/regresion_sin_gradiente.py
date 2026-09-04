import random
import csv

# -----------------------------------------------------------------
# 1) DATASET: Cargar datos del archivo CSV
# -----------------------------------------------------------------
mileage_train = []
selling_price_train = []
mileage_test = []
selling_price_test = []

with open('automobile_dataset.csv', 'r') as file:
    reader = csv.DictReader(file)
    data = list(reader)
    
    # Primeros 100 datos para entrenamiento
    for row in data[:100]:
        mileage_train.append(float(row['Mileage']))
        selling_price_train.append(float(row['Selling_Price']))
    
    # Siguientes 10 datos para pruebas
    for row in data[100:110]:
        mileage_test.append(float(row['Mileage']))
        selling_price_test.append(float(row['Selling_Price']))

kilometraje = mileage_train
precio = selling_price_train
n = len(kilometraje)


# -----------------------------------------------------------------
# 2) PREDICCION Y COSTO (las mismas de siempre, sin derivadas)
# -----------------------------------------------------------------
def predecir(b, w, x):
    return b + w * x


def calcular_costo(b, w, x_datos, y_datos):
    suma_error_cuadrado = 0
    for i in range(len(x_datos)):                    # <-- repetitiva
        error = predecir(b, w, x_datos[i]) - y_datos[i]
        suma_error_cuadrado += error ** 2
    return suma_error_cuadrado / len(x_datos)


# -----------------------------------------------------------------
# 3) BUSQUEDA ALEATORIA (sin gradiente, sin derivadas)
# -----------------------------------------------------------------
def busqueda_aleatoria(x_datos, y_datos,
                        intentos_totales=200000,
                        paso_inicial=50.0,
                        paciencia=500,
                        paso_minimo=0.0001):
    """
    paso_inicial : tamano maximo del "salto" aleatorio al inicio
    paciencia    : cuantos intentos seguidos sin mejora se toleran
                   antes de reducir el paso
    paso_minimo  : cuando el paso se vuelve mas pequeno que esto,
                   se detiene la busqueda (ya no vale la pena seguir)
    """

    random.seed(42)   # para que el resultado sea reproducible

    b = 0.0
    w = 0.0
    mejor_costo = calcular_costo(b, w, x_datos, y_datos)
    paso = paso_inicial
    intentos_sin_mejora = 0

    for intento in range(intentos_totales):          # <-- REPETITIVA

        # Generar una variacion aleatoria dentro de [-paso, +paso]
        b_candidato = b + random.uniform(-paso, paso)
        w_candidato = w + random.uniform(-paso, paso)

        costo_candidato = calcular_costo(b_candidato, w_candidato,
                                          x_datos, y_datos)

        # ---- Estructura DECISIVA: aceptar o descartar el cambio ----
        if costo_candidato < mejor_costo:
            b = b_candidato
            w = w_candidato
            mejor_costo = costo_candidato
            intentos_sin_mejora = 0
        else:
            intentos_sin_mejora += 1

        # ---- Estructura DECISIVA: reducir el paso si no hay mejora ----
        if intentos_sin_mejora >= paciencia:
            paso = paso / 2
            intentos_sin_mejora = 0

        # ---- Estructura DECISIVA: reportar avance ----
        if intento % 20000 == 0:
            print(f"Intento {intento:>6} | Costo (MSE): {mejor_costo:,.2f} "
                  f"| b={b:.4f}  w={w:.4f}  | paso={paso:.6f}")

        # ---- Estructura DECISIVA: detener si el paso ya es minusculo ----
        if paso < paso_minimo:
            print(f"Paso demasiado pequeno, busqueda detenida en el intento {intento}")
            break

    return b, w, mejor_costo


# -----------------------------------------------------------------
# 4) ENTRENAMIENTO DEL MODELO
# -----------------------------------------------------------------
print("=" * 70)
print("BUSCANDO b Y w POR ENSAYO Y ERROR (SIN GRADIENTE)...")
print("=" * 70)

b_final, w_final, costo_final = busqueda_aleatoria(kilometraje, precio)

print("\n" + "=" * 70)
print("PARAMETROS ENCONTRADOS")
print("=" * 70)
print(f"b (intercepto) = {b_final:.4f}")
print(f"w (pendiente)  = {w_final:.4f}")
print(f"\nEcuacion final del modelo:")
print(f"   precio = {b_final:.2f} + ({w_final:.4f}) * kilometraje(miles km)")
print(f"\nCosto final (MSE): {costo_final:,.2f}")
print(f"Raiz del error cuadratico medio (RMSE): {costo_final ** 0.5:,.2f} USD")


# -----------------------------------------------------------------
# 5) INFERENCIA PARA LOS 10 DATOS DE PRUEBA
# -----------------------------------------------------------------
print("\n" + "=" * 70)
print("INFERENCIA (PREDICCION DE PRECIO) PARA 10 DATOS DE PRUEBA")
print("=" * 70)
print(f"{'Mileage':>15} | {'Precio Real (USD)':>20} | {'Precio Estimado (USD)':>22} | {'Error':>15}")
print("-" * 78)

errores = []
for i in range(len(mileage_test)):
    mileage = mileage_test[i]
    precio_real = selling_price_test[i]
    precio_estimado = predecir(b_final, w_final, mileage)
    
    if precio_estimado < 0:
        precio_estimado = 0
    
    error = abs(precio_real - precio_estimado)
    errores.append(error)
    
    print(f"{mileage:>15,.0f} | {precio_real:>20,.2f} | {precio_estimado:>22,.2f} | {error:>15,.2f}")

print("=" * 70)
print(f"Error promedio (MAE): {sum(errores)/len(errores):,.2f} USD")
print("=" * 70)
