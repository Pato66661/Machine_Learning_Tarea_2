import streamlit as st
import pandas as pd
import pickle
import numpy as np
import plotly.express as px
import plotly.figure_factory as ff
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

# 🎨 Estilo ejecutivo limpio
st.set_page_config(layout="wide")
st.markdown("""
    <style>
    #MainMenu, footer, header {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# 📊 Encabezado
st.markdown("<h1 style='text-align: center; color:#2e4053;'>📈 Análisis Gerencial de Churn</h1>", unsafe_allow_html=True)
st.markdown("---")

# 📂 Cargar datos
df = pd.read_csv(r'C:\Users\Dc\Documents\Maestría_Data_Science\Aprendizaje_Machine_Learning\Datasets\Clientes_Telecomunicaciones.csv', sep=',', encoding='latin1')
if 'customerID' in df.columns:
    df = df.drop(columns=['customerID'])

# Convertir Churn a binaria
df_corr = df.copy()
if df_corr['Churn'].dtype == 'object':
    df_corr['Churn'] = df_corr['Churn'].map({'Yes': 1, 'No': 0})
y_true = df_corr['Churn'].values

# Cargar modelos y codificadores
modelos = {
    'Random Forest - Completo': 'rf_all_features.pkl',
    'Random Forest - Top Features': 'rf_top_features.pkl',
    'Logística - Top Features': 'logit_top_features.pkl',
    'Gradient Boosting - Top Features': 'gb_top_features.pkl'
}
model_objects = {nombre: pickle.load(open(ruta, 'rb')) for nombre, ruta in modelos.items()}
label_encoders = pickle.load(open("label_encoders.pkl", "rb"))

# Preparar features y split
X_full = df.drop(columns=['Churn'])
y_full_bin = (df['Churn'] == 'Yes').astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X_full, y_full_bin, test_size=0.2, random_state=42, stratify=y_full_bin
)

# Aplicar encoding a variables categóricas
for col, le in label_encoders.items():
    if col in X_train.columns:
        X_train[col] = le.transform(X_train[col])
        X_test[col] = le.transform(X_test[col])

# ⚙️ Sidebar
st.sidebar.title("⚙️ Parámetros")
modelo_nombre = st.sidebar.selectbox("Selecciona un modelo:", list(modelos.keys()))
modelo = model_objects[modelo_nombre]
features = modelo.feature_names_in_

# 📥 Ingreso de datos cliente
st.subheader("📥 Ingreso de Datos del Cliente")
user_input = {}
for col in features:
    if col in label_encoders:
        opciones = label_encoders[col].classes_.tolist()
        seleccion = st.selectbox(f"{col}", opciones)
        user_input[col] = label_encoders[col].transform([seleccion])[0]
    else:
        user_input[col] = st.number_input(f"{col}", step=0.1)

# 🔍 Predicción
if st.button("🔍 Predecir"):
    entrada_df = pd.DataFrame([user_input])[features]
    prob = modelo.predict_proba(entrada_df)[0][1]
    pred = modelo.predict(entrada_df)[0]
    resultado = "✅ El cliente probablemente se quedará." if pred == 0 else "⚠️ Alta probabilidad de abandono."

    st.markdown("---")
    st.subheader("🧾 Resultados de Predicción")
    col1, col2 = st.columns([1, 2])
    col1.metric("Probabilidad de Churn", f"{prob:.2%}")
    col2.success(resultado if pred == 1 else resultado)
    st.markdown("---")

# 📊 Panel de pestañas
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["🎯 Churn", "📉 Tenure", "📊 Numéricas", "📈 Métricas", "🧩 Matriz", "🔬 Mejores Características"])

with tab1:
    st.subheader("📊 Distribución de Churn")
    fig_churn = px.histogram(df, x='Churn', color='Churn',
                             color_discrete_sequence=["#08304A", "#6B7C9F"],
                             title="Distribución Ejecutiva del Abandono de Clientes")
    fig_churn.update_layout(xaxis_title="Churn", yaxis_title="Cantidad", bargap=0.3)
    st.plotly_chart(fig_churn, use_container_width=True)

    fig_scatter = px.scatter(df, x='tenure', y='MonthlyCharges', color='Churn',
                             color_discrete_sequence=["#08304A", "#6B7C9F"],
                             title="Relación entre Permanencia y Cargos Mensuales",
                             labels={'tenure': 'Meses de Permanencia', 'MonthlyCharges': 'Cargos Mensuales'})
    st.plotly_chart(fig_scatter, use_container_width=True)

with tab2:
    st.subheader("📉 Distribución de Tenure")
    fig_tenure = px.histogram(df, x='tenure', color='Churn', nbins=24, opacity=0.7,
                              marginal="rug", color_discrete_sequence=px.colors.sequential.Teal)
    fig_tenure.update_layout(xaxis_title="Meses", yaxis_title="Cantidad")
    st.plotly_chart(fig_tenure, use_container_width=True)

with tab3:
    st.subheader("📊 Histograma de Variables Numéricas")
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    for col in num_cols:
        fig_num = px.histogram(df, x=col, color='Churn', nbins=20,
                               color_discrete_sequence=px.colors.sequential.Viridis, opacity=0.7)
        st.plotly_chart(fig_num, use_container_width=True)

with tab4:
    st.subheader("📈 Métricas de Desempeño en Test")
    metricas_dict = {}
    for nombre, modelo in model_objects.items():
        X_eval = X_test[modelo.feature_names_in_]
        y_pred = modelo.predict(X_eval)
        y_proba = modelo.predict_proba(X_eval)[:, 1]
        metricas_dict[nombre] = [
            accuracy_score(y_test, y_pred),
            f1_score(y_test, y_pred),
            roc_auc_score(y_test, y_proba)
        ]
    df_metrics = pd.DataFrame(metricas_dict, index=['Accuracy', 'F1 Score', 'AUC']).T.reset_index().rename(columns={'index': 'Modelo'})
    fig_metrics = px.bar(df_metrics, x='Modelo', y=['Accuracy', 'F1 Score', 'AUC'], barmode='group')
    st.plotly_chart(fig_metrics, use_container_width=True)

with tab5:
    st.subheader("🧩 Matrices de Confusión (Test Set)")
    for nombre, modelo in model_objects.items():
        st.markdown(f"**{nombre}**")
        X_eval = X_test[modelo.feature_names_in_]
        y_pred = modelo.predict(X_eval)
        cm = confusion_matrix(y_test, y_pred)
        fig_cm = ff.create_annotated_heatmap(z=cm, x=['No Churn', 'Churn'], y=['No Churn', 'Churn'],
                                             colorscale='Blues', showscale=True)
        fig_cm.update_layout(title=f'Matriz de Confusión - {nombre}')
        st.plotly_chart(fig_cm, use_container_width=True)

with tab6:
    importances_tab = st.expander("🔍 Importancia de Variables en el Modelo Seleccionado", expanded=True)

with importances_tab:
    st.subheader("🔬 Importancia de Características (Feature Importance)")
    if hasattr(modelo, 'feature_importances_'):
        feature_importances = pd.Series(modelo.feature_importances_, index=modelo.feature_names_in_).sort_values(ascending=False).head(10)
        fig_feat = px.bar(feature_importances, orientation='h', color=feature_importances,
                          color_continuous_scale='Blues_r',
                          labels={'value': 'Importancia', 'index': 'Variable'},
                          title="Top 10 Variables Más Relevantes")
        fig_feat.update_layout(yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig_feat, use_container_width=True)
    else:
        st.info("ℹ️ Este modelo no permite visualizar importancia de características directamente.")
