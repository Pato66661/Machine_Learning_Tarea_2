Análisis Gerencial de Churn con Streamlit
Aplicación interactiva para predecir la pérdida de clientes (churn) en telecomunicaciones, utilizando modelos de machine learning y visualizaciones ejecutivas integradas en un dashboard profesional.

🚀 Descripción General
Este proyecto permite:

Comparar modelos de machine learning para predicción de churn.

Ingresar datos de nuevos clientes y obtener predicciones personalizadas.

Visualizar de forma intuitiva la distribución de variables clave y desempeño de modelos.

Resaltar variables más influyentes para la toma de decisiones estratégicas.

🛠️ Tecnologías Utilizadas
Python 3

Streamlit

scikit-learn

Pandas / NumPy

Plotly

Pickle (para cargar modelos y codificadores entrenados)

🎯 Funcionalidades Clave
Carga de Modelos: Comparación entre Random Forest, Regresión Logística, Gradient Boosting.

Ingreso Personalizado de Datos: Simulación en tiempo real del resultado para un cliente específico.

Evaluación de Modelos: Métricas como Accuracy, F1 Score, AUC y matrices de confusión.

Visualización Ejecutiva: Distribuciones de churn, tenure, métricas y variables numéricas.

Importancia de Variables: Presentación de top features según cada modelo.

📁 Estructura del Proyecto
📦 churn_dashboard/
├── data/
│   └── Clientes_Telecomunicaciones.csv
├── models/
│   ├── rf_all_features.pkl
│   ├── rf_top_features.pkl
│   └── ...
├── label_encoders.pkl
├── app.py
└── README.md
▶️ Ejecución Local
bash
git clone https://github.com/tu_usuario/churn_dashboard.git
cd churn_dashboard
streamlit run app.py
📌 Consideraciones
Asegúrate de tener los archivos .pkl (modelos y codificadores) en la carpeta correspondiente.

El archivo CSV debe estar limpio y codificado adecuadamente.

Compatible con navegadores modernos para visualización óptima.

📈 Público Objetivo
Esta herramienta está diseñada para:

Analistas de negocios

Gerentes de marketing o retención

Consultores en ciencia de datos

Tomadores de decisiones estratégicas
