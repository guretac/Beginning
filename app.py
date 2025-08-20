import streamlit as st
import pandas as pd
import pydeck as pdk
import numpy as np
import io

# Configurar el diseño de la página para ocupar todo el ancho
st.set_page_config(layout="wide")

# Título de la aplicación
st.title("Dashboard de Beginning CLARO-VTR")
st.markdown("<p style='font-size:18px;'>Desarrollado por EconoDataAI</p>", unsafe_allow_html=True)
st.markdown("<a href='https://www.econodataai.cl' target='_blank'>www.econodataai.cl</a>", unsafe_allow_html=True)
st.markdown("---")

# Función específica para corregir coordenadas en formato chileno
def fix_chilean_coord(coord):
    """
    Convierte '-705.508.782' -> -70.5508782 y similares.
    Sirve para coordenadas ingresadas en formato con separador de miles '.'
    """
    s = str(coord).strip()
    if pd.isna(coord) or s.lower() in ('nan', '', 'none'):
        return np.nan
    if s.startswith('-'):
        negative = True
        s = s[1:]
    else:
        negative = False
    digits = ''.join(s.split('.'))
    if len(digits) < 3:
        return np.nan
    try:
        result = float(digits[:2] + '.' + digits[2:])
        if negative:
            result = -result
        return result
    except Exception:
        return np.nan

# Cargar el archivo CSV
@st.cache_data
def load_data(file_path):
    """
    Carga los datos del archivo CSV de forma robusta.
    """
    try:
        df = pd.read_csv(file_path, sep=';')
    except Exception:
        try:
            df = pd.read_csv(file_path, sep=',')
        except Exception as e:
            st.error(f"No se pudo cargar el archivo CSV. Por favor, verifica el formato y el delimitador. Error: {e}")
            return None

    df.columns = df.columns.str.strip()

    for col in ['longitude', 'latitude']:
        if col in df.columns:
            df[col] = df[col].apply(fix_chilean_coord)

    if 'KMS RECORRIDOS' in df.columns:
        df['KMS RECORRIDOS'] = df['KMS RECORRIDOS'].astype(str).str.replace(',', '.', regex=False)
        df['KMS RECORRIDOS'] = pd.to_numeric(df['KMS RECORRIDOS'], errors='coerce')

    columnas_numericas = ['Duracion', 'Qact', 'Q_reit', 'Tiempo de viaje', 'PxDIa']
    for col in columnas_numericas:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace(',', '.', regex=False)
            df[col] = pd.to_numeric(df[col], errors='coerce')

    if "longitude" in df.columns and "latitude" in df.columns:
        df.dropna(subset=['longitude', 'latitude'], inplace=True)
    else:
        st.warning("No se encontraron las columnas 'longitude' y/o 'latitude'. El mapa no estará disponible.")

    if 'Tecnico' in df.columns:
        df['Tecnico'] = df['Tecnico'].astype(str)

    return df

# Cargar los datos
file_path = "Beginning2.csv"
try:
    df = load_data(file_path)
    if df is None or df.empty:
        st.stop()
except FileNotFoundError:
    st.error(f"Error: El archivo '{file_path}' no se encuentra. Asegúrate de que está en la misma carpeta que el script.")
    st.stop()
except Exception as e:
    st.error(f"Ocurrió un error inesperado al cargar el archivo: {e}.")
    st.stop()

# --- FILTROS GLOBALES ---
st.sidebar.header("Filtros del Dashboard")
if 'Tecnico' in df.columns:
    tecnicos = ['TODOS'] + df['Tecnico'].unique().tolist()
    selected_tecnicos = st.sidebar.multiselect(
        "Seleccionar Técnico:",
        options=tecnicos,
        default=['TODOS']
    )
else:
    st.warning("No se encontró la columna 'Tecnico'. El filtro por técnico no estará disponible.")
    selected_tecnicos = ['TODOS']

# Aplicar los filtros al DataFrame
filtered_df = df.copy()
if 'TODOS' not in selected_tecnicos:
    filtered_df = filtered_df[filtered_df['Tecnico'].isin(selected_tecnicos)]

if filtered_df.empty:
    st.warning("No hay datos que coincidan con los filtros seleccionados.")
    st.stop()

# =========================================================================
# === INICIO DE LA REORGANIZACIÓN DEL DASHBOARD ===
# =========================================================================

# --- 1. SECCIÓN DE KPIS (Tarjetas de Indicadores Clave) ---
st.markdown("### 1. Indicadores Clave (KPIs)")

def calculate_kpis(data_frame, label_prefix=""):
    """
    Calcula y muestra los KPIs para un DataFrame dado.
    """
    avg_qact = data_frame['Qact'].mean() if 'Qact' in data_frame.columns else 0
    total_km = data_frame['KMS RECORRIDOS'].sum() if 'KMS RECORRIDOS' in data_frame.columns else 0

    # Calcular el promedio de cumplimiento de franja
    if 'cumple_franja' in data_frame.columns:
        # Convertir los valores a 1 si es 'Cumple' y 0 si no lo es
        data_frame['cumple_franja_num'] = data_frame['cumple_franja'].apply(lambda x: 1 if x == 'Cumple' else 0)
        avg_cumple_franja = data_frame['cumple_franja_num'].mean() * 100 if 'cumple_franja_num' in data_frame.columns else 0
    else:
        avg_cumple_franja = 0

    avg_duration = data_frame['Duracion'].mean() if 'Duracion' in data_frame.columns else 0

    kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
    with kpi_col1:
        st.metric(label=f"Promedio Qact{label_prefix}", value=f"{avg_qact:.2f}")

    with kpi_col2:
        st.metric(label=f"Promedio Cumplimiento Franja{label_prefix}", value=f"{avg_cumple_franja:.2f}%")

    with kpi_col3:
        st.metric(label=f"Promedio Duración (min){label_prefix}", value=f"{avg_duration:.0f}")
    
    with kpi_col4:
        st.metric(label=f"Total Kilómetros{label_prefix}", value=f"{total_km:,.0f} km")


# Lógica para mostrar KPIs en base a la selección
if 'TODOS' in selected_tecnicos:
    st.markdown("#### Resumen total de todos los técnicos")
    calculate_kpis(df)
else:
    st.markdown("#### Comparación de KPIs: Total vs. Seleccionados")
    
    st.markdown("##### Total de todos los técnicos")
    calculate_kpis(df)
    
    st.markdown("---")

    st.markdown("##### Técnicos seleccionados")
    calculate_kpis(filtered_df)

st.markdown("---")

# --- 2. DISTRIBUCIÓN DE VARIABLES ---
with st.container():
    with st.expander("2. Distribución de Variables", expanded=True):
        st.subheader("Análisis de Datos Generales")
        columns_for_filter = [col for col in filtered_df.columns if col not in ["longitude", "latitude", "Tecnico", "Ciudad"]]

        if columns_for_filter:
            filter_column = st.selectbox("Selecciona la columna para analizar:", options=columns_for_filter, key='general_filter_col_select')
            if filter_column:
                st.subheader(f"Distribución de '{filter_column}'")
                data_counts = filtered_df[filter_column].fillna('Sin Valor').value_counts().reset_index()
                data_counts.columns = [filter_column, 'Conteo']
                if not data_counts.empty:
                    st.bar_chart(data_counts, x=filter_column, y='Conteo', use_container_width=True)
                else:
                    st.info("No hay datos para mostrar la distribución.")
                st.write(f"Datos completos para los filtros seleccionados:")
                st.dataframe(filtered_df, use_container_width=True)
            else:
                st.warning("No hay columnas disponibles para analizar.")
                st.dataframe(filtered_df, use_container_width=True)
st.markdown("---")

# --- 3. KILÓMETROS RECORRIDOS ---
with st.container():
    with st.expander("3. Kilómetros Recorridos", expanded=True):
        st.subheader("Análisis de Kilómetros Recorridos")
        if 'KMS RECORRIDOS' in filtered_df.columns and 'Tecnico' in filtered_df.columns:
            st.markdown("---")
            st.markdown("#### Costo de combustible ⛽")
            precio_petroleo = st.number_input(
                "Precio del Petróleo por Litro ($CLP):",
                min_value=1,
                value=1000,
                step=1
            )
            rendimiento_vehiculo = st.number_input(
                "Rendimiento del Vehículo (km/L):",
                min_value=1.0,
                value=10.0,
                step=0.1
            )
            st.markdown("### Resumen Gráfico de Kilómetros Recorridos")

            if 'TODOS' in selected_tecnicos:
                total_km_por_tecnico = filtered_df.groupby('Tecnico')['KMS RECORRIDOS'].sum().reset_index()
                total_km_por_tecnico.rename(columns={'KMS RECORRIDOS': 'Kilómetros Recorridos'}, inplace=True)
                if not total_km_por_tecnico.empty:
                    st.bar_chart(total_km_por_tecnico, x='Tecnico', y='Kilómetros Recorridos', use_container_width=True)
                st.markdown("### Tabla de Kilómetros Totales por Técnico")
                st.dataframe(total_km_por_tecnico, use_container_width=True)
                total_general_km = total_km_por_tecnico['Kilómetros Recorridos'].sum()
                gasto_combustible = (total_general_km / rendimiento_vehiculo) * precio_petroleo if rendimiento_vehiculo > 0 else 0
                st.markdown("---")
                st.markdown(f"### **Total General de Kilómetros:**")
                st.success(f"**{total_general_km:,.2f} km**")
                st.markdown(f"### **Gasto Estimado en Combustible:**")
                st.success(f"**$ {gasto_combustible:,.2f} CLP**")
            else:
                total_km_tecnico = filtered_df['KMS RECORRIDOS'].sum()
                gasto_combustible = (total_km_tecnico / rendimiento_vehiculo) * precio_petroleo if rendimiento_vehiculo > 0 else 0
                st.write(f"Distribución de viajes para los técnicos seleccionados:")
                st.bar_chart(filtered_df.groupby('Tecnico')['KMS RECORRIDOS'].sum(), use_container_width=True)
                st.markdown("---")
                st.markdown(f"### **Kilómetros Totales para los técnicos seleccionados:**")
                st.success(f"**{total_km_tecnico:,.2f} km**")
                st.markdown(f"### **Gasto Estimado en Combustible:**")
                st.success(f"**$ {gasto_combustible:,.2f} CLP**")
                st.markdown("---")
                st.markdown("### Detalle de Viajes")
                st.dataframe(filtered_df, use_container_width=True)
        else:
            st.warning("No se encontró la columna 'KMS RECORRIDOS' o 'Tecnico' en los datos. No se puede realizar este análisis.")
st.markdown("---")

# --- 4. CALCULADORA DE REMUNERACIÓN ---
with st.container():
    with st.expander("4. Calculadora de Remuneración", expanded=True):
        st.subheader("Elige las columnas y sus pesos para el cálculo")
        metricas_rendimiento = ['Duracion', 'Qact', 'Q_reit', 'Tiempo de viaje', 'cumple_franja', 'PxDIa']
        disponible_metrics = [col for col in metricas_rendimiento if col in filtered_df.columns]
        if not disponible_metrics:
            st.warning("No se encontraron columnas de métricas de rendimiento para el cálculo.")
        else:
            if 'weights' not in st.session_state:
                st.session_state.weights = {col: 1.0 for col in disponible_metrics}
            selected_rem_cols = st.multiselect(
                "Elige las métricas que influyen en la remuneración:",
                options=disponible_metrics,
                default=disponible_metrics[:2] if len(disponible_metrics) >= 2 else disponible_metrics,
                key='rem_cols_multiselect'
            )
            weights = {}
            if selected_rem_cols:
                st.markdown("**Asigna un peso a cada métrica:**")
                for col in selected_rem_cols:
                    weights[col] = st.slider(
                        f"Peso para '{col}'",
                        min_value=-1.0,
                        max_value=1.0,
                        value=st.session_state.weights.get(col, 0.1),
                        step=0.1,
                        key=f"slider_{col}"
                    )
                st.session_state.weights.update(weights)
                st.markdown("---")
                if 'TODOS' in selected_tecnicos:
                    if not filtered_df.empty:
                        resultados_list = []
                        for tecnico in filtered_df['Tecnico'].unique():
                            tecnico_data = filtered_df[filtered_df['Tecnico'] == tecnico][selected_rem_cols].mean()
                            remuneracion_calculada = sum(tecnico_data.get(col, 0) * weights.get(col, 0) for col in selected_rem_cols)
                            resultados_list.append({
                                'Técnico': tecnico,
                                'Remuneración Total': remuneracion_calculada
                            })
                        if resultados_list:
                            resultados_df = pd.DataFrame(resultados_list)
                            st.dataframe(resultados_df, use_container_width=True)
                        else:
                            st.info("No hay resultados para mostrar.")
                else:
                    tecnico_data = filtered_df[selected_rem_cols].mean()
                    remuneracion_calculada = sum(tecnico_data.get(col, 0) * weights.get(col, 0) for col in selected_rem_cols)
                    st.markdown(f"### **Remuneración Total Calculada para los técnicos seleccionados:**")
                    st.success(f"**$ {remuneracion_calculada:,.2f}**")
            else:
                st.info("Por favor, selecciona las métricas para el cálculo.")
st.markdown("---")

# --- 5. MAPA INTERACTIVO DE GEOREFERENCIA ---
with st.container():
    with st.expander("5. Mapa Interactivo de Georeferencia", expanded=True):
        st.subheader("Mapa de georeferencia de los lugares")
        if "longitude" in filtered_df.columns and "latitude" in filtered_df.columns and not filtered_df.empty:
            map_data = filtered_df[['longitude', 'latitude', 'Tecnico']].copy()
            map_data = map_data.dropna(subset=['longitude', 'latitude'])
            if not map_data.empty:
                view_state = pdk.ViewState(
                    latitude=map_data['latitude'].mean(),
                    longitude=map_data['longitude'].mean(),
                    zoom=10,
                    pitch=50,
                )
                layer = pdk.Layer(
                    "ScatterplotLayer",
                    data=map_data,
                    get_position="[longitude, latitude]",
                    get_color=[200, 30, 0, 160],
                    get_radius=100,
                    pickable=True,
                    tooltip={
                        "html": "<b>Técnico:</b> {Tecnico}",
                        "style": {"color": "white"}
                    }
                )
                r = pdk.Deck(
                    layers=[layer],
                    initial_view_state=view_state,
                    tooltip=True,
                    map_style='light'
                )
                st.pydeck_chart(r, use_container_width=True)
            else:
                st.info("No hay datos de coordenadas válidos para mostrar en el mapa para los filtros seleccionados.")
        else:
            st.info("El archivo no contiene las columnas 'longitude' y 'latitude' o no hay datos válidos.")
st.markdown("---")