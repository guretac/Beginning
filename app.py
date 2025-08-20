import streamlit as st
import pandas as pd
import pydeck as pdk
import numpy as np
import io

# Configurar el diseño de la página para ocupar todo el ancho
st.set_page_config(layout="wide")

# Título de la aplicación
st.title("Dashboard de Rendimiento - CLARO-VTR")
st.markdown("<p style='font-size:18px;'>Desarrollado por EconoDataAI</p>", unsafe_allow_html=True)
st.markdown("<a href='https://www.econodataai.cl' target='_blank'>www.econodataai.cl</a>", unsafe_allow_html=True)
st.markdown("---")

# Función específica para corregir coordenadas en formato chileno
def fix_chilean_coord(coord):
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
    try:
        df = pd.read_csv(file_path, sep=';')
    except Exception:
        try:
            df = pd.read_csv(file_path, sep=',')
        except Exception as e:
            st.error(f"No se pudo cargar el archivo CSV. Por favor, verifica el formato y el delimitador. Error: {e}")
            return None

    df.columns = df.columns.str.strip()

    for col in ['Coord X', 'Coord Y']:
        if col in df.columns:
            df[col] = df[col].apply(fix_chilean_coord)

    # Convertir 'KMS RECORRIDOS' a formato numérico
    if 'KMS RECORRIDOS' in df.columns:
        df['KMS RECORRIDOS'] = df['KMS RECORRIDOS'].astype(str).str.replace(',', '.', regex=False)
        df['KMS RECORRIDOS'] = pd.to_numeric(df['KMS RECORRIDOS'], errors='coerce')

    columnas_numericas = ['Duracion', 'Qact', 'Q_reit', 'Tiempo de viaje', 'PxDIa']
    for col in columnas_numericas:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace(',', '.', regex=False)
            df[col] = pd.to_numeric(df[col], errors='coerce')

    if "Coord X" in df.columns and "Coord Y" in df.columns:
        df.dropna(subset=['Coord X', 'Coord Y'], inplace=True)
    else:
        st.warning("No se encontraron las columnas 'Coord X' y/o 'Coord Y'. El mapa no estará disponible.")

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

# --- FILTROS GLOBALES (Similares a un Dashboard de BI) ---
st.sidebar.header("Filtros del Dashboard")
if 'Tecnico' in df.columns:
    tecnicos = ['TODOS'] + df['Tecnico'].unique().tolist()
    selected_tecnico_filter = st.sidebar.selectbox("Selecciona un técnico:", options=tecnicos, key='global_tecnico_select')
else:
    st.warning("No se encontró la columna 'Tecnico'. El filtro por técnico no estará disponible.")
    selected_tecnico_filter = 'TODOS'

if 'Ciudad' in df.columns:
    ciudades = ['TODAS'] + df['Ciudad'].dropna().unique().tolist()
    if 'SANTIAGO' in ciudades:
        default_ciudades = ['SANTIAGO']
    else:
        default_ciudades = ['TODAS']
    selected_ciudades = st.sidebar.multiselect(
        "Selecciona la(s) ciudad(es):",
        options=ciudades,
        default=default_ciudades
    )
else:
    st.warning("No se encontró la columna 'Ciudad'. El filtro por ciudad no estará disponible.")
    selected_ciudades = ['TODAS']

# Aplicar los filtros al DataFrame
filtered_df = df.copy()
if selected_tecnico_filter != 'TODOS':
    filtered_df = filtered_df[filtered_df['Tecnico'] == selected_tecnico_filter]

if 'TODAS' not in selected_ciudades:
    filtered_df = filtered_df[filtered_df['Ciudad'].isin(selected_ciudades)]

if filtered_df.empty:
    st.warning("No hay datos que coincidan con los filtros seleccionados.")
    st.stop()

# --- SECCIÓN DE KPIS (Tarjetas de Indicadores Clave) ---
st.markdown("### Indicadores Clave (KPIs)")
kpi_col1, kpi_col2, kpi_col3 = st.columns(3)
with kpi_col1:
    avg_qact = filtered_df['Qact'].mean() if 'Qact' in filtered_df.columns else 0
    st.metric(label="Promedio Qact", value=f"{avg_qact:.2f}")

with kpi_col2:
    total_reits = filtered_df['Q_reit'].sum() if 'Q_reit' in filtered_df.columns else 0
    st.metric(label="Total Reincidencias", value=f"{total_reits:.0f}")

with kpi_col3:
    avg_duration = filtered_df['Duracion'].mean() if 'Duracion' in filtered_df.columns else 0
    st.metric(label="Promedio Duración (min)", value=f"{avg_duration:.0f}")

st.markdown("---")

# --- DISEÑO PRINCIPAL DEL DASHBOARD ---
# Organizar los visuales en columnas
col1, col2 = st.columns(2, gap="large")

with col1:
    with st.expander("2. Mapa Interactivo de Georeferencia", expanded=True):
        st.subheader("Mapa de georeferencia de los lugares")

        if "Coord X" in filtered_df.columns and "Coord Y" in filtered_df.columns and not filtered_df.empty:
            map_data = filtered_df[['Coord X', 'Coord Y', 'Tecnico']].copy()
            map_data.rename(columns={'Coord X': 'lon', 'Coord Y': 'lat'}, inplace=True)
            map_data = map_data.dropna(subset=['lon', 'lat'])

            if not map_data.empty:
                view_state = pdk.ViewState(
                    latitude=map_data['lat'].mean(),
                    longitude=map_data['lon'].mean(),
                    zoom=10,
                    pitch=50,
                )
                layer = pdk.Layer(
                    "ScatterplotLayer",
                    data=map_data,
                    get_position="[lon, lat]",
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

                @st.cache_data
                def convert_df_to_csv(df_to_convert):
                    return df_to_convert.to_csv(index=False).encode('utf-8')

                csv_data_map = convert_df_to_csv(filtered_df)
                st.download_button(
                    label="Descargar datos del mapa en CSV",
                    data=csv_data_map,
                    file_name="datos_mapa_filtrados.csv",
                    mime="text/csv",
                    help="Descarga los datos georeferenciados visibles en el mapa."
                )

            else:
                st.info("No hay datos de coordenadas válidos para mostrar en el mapa para los filtros seleccionados.")
        else:
            st.info("El archivo no contiene las columnas 'Coord X' y 'Coord Y' o no hay datos válidos.")


with col2:
    with st.expander("1. Análisis de Datos Generales", expanded=True):
        st.subheader("Distribución de variables")

        columns_for_filter = [col for col in filtered_df.columns if col not in ["Coord X", "Coord Y", "Tecnico", "Ciudad"]]

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

# --- MÓDULO DE CÁLCULO DE REMUNERACIÓN Y KILÓMETROS ---
with st.container():
    rem_col, km_col = st.columns(2, gap="large")

    # Módulo de Remuneración
    with rem_col:
        with st.expander("3. Calculadora de Remuneración", expanded=True):
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
                    st.subheader(f"Cálculo para {selected_tecnico_filter}")
                    if selected_tecnico_filter == 'TODOS':
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
                        tecnico_data = filtered_df[filtered_df['Tecnico'] == selected_tecnico_filter][selected_rem_cols].mean()
                        remuneracion_calculada = sum(tecnico_data.get(col, 0) * weights.get(col, 0) for col in selected_rem_cols)
                        st.markdown(f"### **Remuneración Total Calculada:**")
                        st.success(f"**$ {remuneracion_calculada:,.2f}**")
                else:
                    st.info("Por favor, selecciona las métricas para el cálculo.")

    # Módulo de Kilómetros
    with km_col:
        with st.expander("4. Kilómetros Recorridos", expanded=True):
            st.subheader("Análisis de Kilómetros Recorridos")

            # Verificar si la columna existe en el DataFrame filtrado
            if 'KMS RECORRIDOS' in filtered_df.columns and 'Tecnico' in filtered_df.columns:

                st.markdown("### Resumen Gráfico de Kilómetros Recorridos")

                if selected_tecnico_filter == 'TODOS':
                    # Sumar los kilómetros por cada técnico
                    total_km_por_tecnico = filtered_df.groupby('Tecnico')['KMS RECORRIDOS'].sum().reset_index()
                    total_km_por_tecnico.rename(columns={'KMS RECORRIDOS': 'Kilómetros Recorridos'}, inplace=True)
                    
                    if not total_km_por_tecnico.empty:
                        st.bar_chart(total_km_por_tecnico, x='Tecnico', y='Kilómetros Recorridos', use_container_width=True)
                    
                    st.markdown("### Tabla de Kilómetros Totales por Técnico")
                    st.dataframe(total_km_por_tecnico, use_container_width=True)
                    
                    # Mostrar el total general
                    total_general_km = total_km_por_tecnico['Kilómetros Recorridos'].sum()
                    st.markdown(f"### **Total General de Kilómetros:**")
                    st.success(f"**{total_general_km:,.2f} km**")

                else:
                    # Sumar la distancia total del técnico seleccionado
                    total_km_tecnico = filtered_df['KMS RECORRIDOS'].sum()
                    
                    # Mostrar el gráfico de contribución de cada viaje al total
                    st.write(f"Distribución de viajes para {selected_tecnico_filter}:")
                    st.bar_chart(filtered_df, x=filtered_df.index, y='KMS RECORRIDOS', use_container_width=True)
                    
                    st.markdown(f"### **Kilómetros Totales para {selected_tecnico_filter}:**")
                    st.success(f"**{total_km_tecnico:,.2f} km**")
                    
                    st.markdown("---")
                    st.markdown("### Detalle de Viajes")
                    st.dataframe(filtered_df, use_container_width=True)
            else:
                st.warning("No se encontró la columna 'KMS RECORRIDOS' o 'Tecnico' en los datos. No se puede realizar este análisis.")