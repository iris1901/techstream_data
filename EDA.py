import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Carga del dataset
df = pd.read_csv("server_sensor_data.csv")

# Inspeccion
print("\nDataset info:")
print(df.info())

print("\nSummary statistics:")
print(df.describe())

print("\nFirst rows:")
print(df.head())

# Target variables
plt.figure(figsize=(6,4))
sns.countplot(x="failure", data=df)
plt.title("Failure distribution")
plt.xlabel("Failure (0 = normal, 1 = failure)")
plt.ylabel("Count")
plt.show()

print("\nFailure distribution:")
print(df["failure"].value_counts(normalize=True))

# El dataset presenta un cierto desbalanceo, con aproximadamente un 75% de observaciones normales frente a un 25% de fallos.
# Este comportamiento es bastante coherente con un escenario real, donde los fallos suelen ser menos frecuentes que el 
# funcionamiento normal. Además, el desbalanceo no es excesivo, por lo que el modelo debería poder aprender patrones de ambas
# clases sin necesidad de aplicar técnicas como resampling o ponderación de clases.

# Bivariante analisis
features = df.columns[:-1]

plt.figure(figsize=(14, 10))
for i, col in enumerate(features):
    plt.subplot(3, 3, i+1)
    sns.histplot(df[col], kde=True)
    plt.title(col)
plt.tight_layout()
plt.show()

# Las variables presentan distribuciones suaves y bastante continuas, en muchos casos cercanas a una normal, lo que sugiere que el
# proceso de generación de datos es estable y no introduce valores erráticos o extremos de forma artificial.
# En general, no se aprecian valores atípicos extremos ni distribuciones anómalas, lo que indica que los datos son consistentes y
# adecuados para el entrenamiento de modelos.

# Correlacion analisis
plt.figure(figsize=(10, 8))
corr = df.corr()

sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation matrix")
plt.show()

# La matriz de correlación muestra relaciones fuertes entre muchas de las variables, especialmente entre cpu_usage_pct, temperature_c 
# y power_consumption_w, lo cual es coherente desde el punto de vista físico, ya que un mayor uso de CPU suele implicar mayor temperatura y consumo energético.
# En relación con la variable objetivo, se aprecia una correlación positiva moderada con prácticamente todas las variables.

# Relación con el objetivo
plt.figure(figsize=(14, 10))

for i, col in enumerate(features):
    plt.subplot(3, 3, i+1)
    sns.boxplot(x="failure", y=col, data=df)
    plt.title(f"{col} vs failure")

plt.tight_layout()
plt.show()

# Los boxplots muestran una separación clara entre las clases para la mayoría de variables. En los casos de fallo (failure = 1), se observan valores sistemáticamente
# más altos en temperatura, uso de CPU, memoria, latencia y tasa de error, lo cual es coherente con situaciones de sobrecarga del sistema.
# Aunque existe cierto solapamiento entre ambas clases, las medianas y distribuciones están claramente desplazadas, lo que indica que las variables contienen 
# información relevante para discriminar entre estados normales y fallos.
# En conjunto, estos patrones sugieren que el modelo debería ser capaz de capturar relaciones útiles sin que el problema resulte trivial.

# Pairplot
sns.pairplot(df.sample(500), hue="failure")
plt.show()

# El pairplot confirma las relaciones observadas previamente, mostrando patrones lineales claros entre muchas de las variables, especialmente aquellas relacionadas
# con la carga del sistema como CPU, temperatura y consumo energético.

# Se aprecia además que los casos de fallo tienden a concentrarse en regiones donde las variables toman valores más altos, aunque con cierto solapamiento con la clase
# normal. Esto refuerza la idea de que el problema no es trivial, pero sí presenta suficiente separabilidad para que un modelo pueda aprender patrones útiles.