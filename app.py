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
st.markdown("<p style='font-size:18px;'>www.econodataai.cl</p>", unsafe_allow_html=True)
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
    - Intenta leer con delimitador ';' y luego con ','
    - Limpia los nombres de las columnas.
    - Convierte y valida las columnas de coordenadas y otras columnas numéricas.
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

    # Aplica la corrección solo a las columnas de coordenadas
    for col in ['Coord X', 'Coord Y']:
        if col in df.columns:
            df[col] = df[col].apply(fix_chilean_coord)

    # Otras columnas numéricas: corregir posibles comas como decimales
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
file_path = "Beginning.csv"
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

# --- Configuración del diseño de columnas ---
col1, col2 = st.columns(2, gap="large")

# --- Columna Izquierda: Mapa Interactivo ---
with col1:
    with st.expander("2. Mapa Interactivo de Georeferencia", expanded=True):
        st.subheader("Mapa de georeferencia de los lugares")

        if "Coord X" in df.columns and "Coord Y" in df.columns and not df.empty:

            if 'Ciudad' in df.columns:
                ciudades = df['Ciudad'].dropna().unique().tolist()
                if 'SANTIAGO' in ciudades:
                    default_ciudades = ['SANTIAGO']
                else:
                    default_ciudades = []
                selected_ciudades = st.multiselect(
                    "Selecciona la(s) ciudad(es) a visualizar:",
                    options=ciudades,
                    default=default_ciudades
                )
                map_df = df[df['Ciudad'].isin(selected_ciudades)]
            else:
                st.warning("No se encontró la columna 'Ciudad' para filtrar el mapa.")
                map_df = df

            map_data = map_df[['Coord X', 'Coord Y', 'Tecnico']].copy()
            map_data.rename(columns={'Coord X': 'lon', 'Coord Y': 'lat'}, inplace=True)
            map_data = map_data.dropna(subset=['lon', 'lat'])

            if not map_data.empty:
                # Inicializa el mapa en las coordenadas de Santiago si existe
                view_state = pdk.ViewState(
                    latitude=map_data['lat'].mean() if 'SANTIAGO' not in selected_ciudades else df[df['Ciudad'] == 'SANTIAGO']['Coord Y'].mean(),
                    longitude=map_data['lon'].mean() if 'SANTIAGO' not in selected_ciudades else df[df['Ciudad'] == 'SANTIAGO']['Coord X'].mean(),
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
                    map_style='light' # Agregada la propiedad 'map_style' para un fondo claro
                )
                st.pydeck_chart(r, use_container_width=True)

                # --- Nueva función para convertir DataFrame a CSV ---
                @st.cache_data
                def convert_df_to_csv(df_to_convert):
                    # IMPORTANT: Use index=False to avoid writing the DataFrame index as a column
                    return df_to_convert.to_csv(index=False).encode('utf-8')

                # --- Botón de descarga para el mapa ---
                # Usamos map_df original (no map_data) para descargar todas las columnas filtradas
                csv_data_map = convert_df_to_csv(map_df)
                st.download_button(
                    label="Descargar datos del mapa en CSV",
                    data=csv_data_map,
                    file_name="datos_mapa_filtrados.csv",
                    mime="text/csv",
                    help="Descarga los datos georeferenciados visibles en el mapa."
                )

            else:
                st.info("No hay datos de coordenadas válidos para mostrar en el mapa para la(s) ciudad(es) seleccionada(s).")
        else:
            st.info("El archivo no contiene las columnas 'Coord X' y 'Coord Y' o no hay datos válidos.")

# --- Columna Derecha: Análisis de Datos Generales ---
with col2:
    with st.expander("1. Análisis de Datos Generales", expanded=True):
        st.subheader("Análisis de datos generales según filtro")

        # Selector para elegir un técnico
        if 'Tecnico' in df.columns:
            tecnicos = ['TODOS'] + df['Tecnico'].unique().tolist()
            selected_tecnico_filter = st.selectbox("Selecciona un técnico:", options=tecnicos, key='general_tecnico_select')
        else:
            st.warning("No se encontró la columna 'Tecnico'. El filtro por técnico no estará disponible.")
            selected_tecnico_filter = 'TODOS'

        # Filtrar el DataFrame según el técnico seleccionado
        if selected_tecnico_filter == 'TODOS':
            filtered_df = df
        else:
            filtered_df = df[df['Tecnico'] == selected_tecnico_filter]

        columns_for_filter = [col for col in filtered_df.columns if col not in ["Coord X", "Coord Y", "Tecnico"]]

        if columns_for_filter:
            filter_column = st.selectbox("Selecciona la columna para filtrar:", options=columns_for_filter, key='general_filter_col_select')

            if filter_column:
                st.subheader(f"Distribución de '{filter_column}' para {selected_tecnico_filter}")

                data_counts = filtered_df[filter_column].fillna('Sin Valor').value_counts().reset_index()
                data_counts.columns = [filter_column, 'Conteo']

                if not data_counts.empty:
                    st.bar_chart(data_counts, x=filter_column, y='Conteo', use_container_width=True)
                else:
                    st.info("No hay datos para mostrar la distribución.")

                st.write(f"Datos completos para {selected_tecnico_filter}:")
                st.dataframe(filtered_df, use_container_width=True)

                @st.cache_data
                def convert_df_to_excel(df_to_convert):
                    output = io.BytesIO()
                    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                        df_to_convert.to_excel(writer, index=False, sheet_name='Datos_Filtrados')
                    processed_data = output.getvalue()
                    return processed_data

                excel_data = convert_df_to_excel(filtered_df)

                st.download_button(
                    label="Descargar datos filtrados en Excel",
                    data=excel_data,
                    file_name="datos_generales.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    help="Descarga el DataFrame visible en la tabla de abajo."
                )

            else:
                st.warning("No hay columnas disponibles para filtrar.")
                st.dataframe(filtered_df, use_container_width=True)

# --- Contenedor Abajo: Cálculo de Remuneración ---
st.markdown("---")
with st.container():
    with st.expander("3. Calculadora de Remuneración", expanded=True):
        st.subheader("Elige las columnas y sus pesos para el cálculo")

        metricas_rendimiento = [
            'Duracion',
            'Qact',
            'Q_reit',
            'Tiempo de viaje',
            'cumple_franja',
            'PxDIa'
        ]
        
        disponible_metrics = [col for col in metricas_rendimiento if col in df.columns]

        if not disponible_metrics:
            st.warning("No se encontraron columnas de métricas de rendimiento para el cálculo.")
            st.stop()

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
            st.markdown("---")
            st.markdown("**Asigna un peso a cada métrica:**")
            st.markdown("**(Un peso negativo penalizará la remuneración, por ejemplo, para `Q_reit`)**")
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

        if 'Tecnico' not in df.columns:
            st.warning("No se encontró la columna 'Tecnico' para el cálculo.")
            st.stop()

        tecnicos = ['TODOS'] + df['Tecnico'].unique().tolist()
        selected_tecnico = st.selectbox("Selecciona un técnico:", options=tecnicos, key='rem_tecnico_select')

        if selected_rem_cols and selected_tecnico:
            st.markdown("---")
            if selected_tecnico == 'TODOS':
                st.subheader("Detalle del Cálculo para todos los técnicos")
                resultados_list = []
                for tecnico in df['Tecnico'].unique():
                    tecnico_data = df[df['Tecnico'] == tecnico][selected_rem_cols].mean()
                    remuneracion_calculada = 0
                    for col in selected_rem_cols:
                        valor = tecnico_data.get(col, 0)
                        peso = weights.get(col, 0)
                        subtotal = valor * peso
                        remuneracion_calculada += subtotal
                        resultados_list.append({
                            'Técnico': tecnico,
                            'Columna': col,
                            'Valor': valor,
                            'Peso': peso,
                            'Subtotal': subtotal
                        })
                    resultados_list.append({
                        'Técnico': tecnico,
                        'Columna': 'Remuneración Total',
                        'Valor': '',
                        'Peso': '',
                        'Subtotal': remuneracion_calculada
                    })

                if resultados_list:
                    resultados_df = pd.DataFrame(resultados_list)
                    st.dataframe(resultados_df, use_container_width=True)
                    @st.cache_data
                    def convert_df_to_excel(df_to_convert):
                        output = io.BytesIO()
                        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                            df_to_convert.to_excel(writer, index=False, sheet_name='Remuneraciones')
                        processed_data = output.getvalue()
                        return processed_data
                    excel_data = convert_df_to_excel(resultados_df)
                    st.download_button(
                        label="Descargar en Excel",
                        data=excel_data,
                        file_name="remuneraciones_detalle.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                else:
                    st.info("No hay resultados para mostrar.")
            else:
                st.subheader(f"Cálculo de Remuneración para {selected_tecnico}")
                remuneracion_calculada = 0
                detail_df = pd.DataFrame(columns=['Columna', 'Valor', 'Peso', 'Subtotal'])
                tecnico_data = df[df['Tecnico'] == selected_tecnico][selected_rem_cols].mean()
                for col in selected_rem_cols:
                    valor = tecnico_data.get(col, 0)
                    peso = weights.get(col, 0)
                    subtotal = valor * peso
                    remuneracion_calculada += subtotal
                    detail_df.loc[len(detail_df)] = [col, valor, peso, subtotal]
                st.write("### Detalle del Cálculo (Promedio de las métricas del técnico):")
                st.dataframe(detail_df, use_container_width=True)
                st.markdown(f"### **Remuneración Total Calculada:**")
                st.success(f"**$ {remuneracion_calculada:,.2f}**")
        else:
            st.info("Por favor, selecciona las métricas para el cálculo antes de continuar.")
