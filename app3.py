import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import dash_bootstrap_components as dbc
from openpyxl.drawing.image import Image


# Importar Font Awesome para los iconos
FA = "https://use.fontawesome.com/releases/v5.15.3/css/all.css"

# Crear aplicación Dash
app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP, FA])

# Leer el archivo CSV
df = pd.read_csv('datosReto1.csv', sep=';',parse_dates=['fechaInicio', 'diaDelMes'], dayfirst=True)

# Obtener todas las fechas únicas de la columna 'fechaInicio'
todas_las_fechas = df['fechaInicio'].unique()

# Obtener años y meses únicos
anos_unicos = sorted(np.unique(pd.to_datetime(todas_las_fechas).year))
meses_unicos = sorted(np.unique(pd.to_datetime(todas_las_fechas).month))
dias_unicos = sorted(np.unique(pd.to_datetime(todas_las_fechas).day))

available_chart_types = ["Gráfico de Líneas", "Gráfico de Barras"]


# Filtrar DataFrame para obtener ingresos y gastos por separado
df_ingresos = df[df['nombreCuenta'].str.contains('Ingresos|Gastos', case=False, na=False)].copy()

# Separar ingresos y gastos
df_ingresos['tipo'] = np.where(df_ingresos['nombreCuenta'].str.contains('Ingresos', case=False, na=False), 'ingresos', 'gastos')

# Agrupar por fecha y tipo y sumar los valores
resultados_por_fecha = df_ingresos.groupby(['fechaInicio', 'fechaFin', 'tipo'])['Saldo'].sum().unstack(fill_value=0).reset_index()

# Calcular la utilidad neta por fecha restando los gastos de los ingresos
resultados_por_fecha['utilidad_neta'] = resultados_por_fecha['ingresos'] - resultados_por_fecha['gastos']

# Filtrar el DataFrame para obtener solo las filas relacionadas con "activos"
filtro_activos = df['nombreCuenta'].str.contains('activos', case=False, na=False)
df_activos = df[filtro_activos]

# Agrupar por fecha y calcular la suma de 'Saldo' para obtener los activos totales por fecha
activos_totales_por_fecha = df_activos.groupby(['fechaInicio', 'fechaFin'])['Saldo'].sum().reset_index()

# Calcular el promedio de activos totales por fecha
activo_promedio_por_fecha = activos_totales_por_fecha.groupby('fechaInicio')['Saldo'].mean().reset_index()

# Filtrar las filas que contienen '45' en la columna 'codContable'
gas_ope = df['codContable'] == 45
df_gastos_operativos = df[gas_ope]

# Agrupar por fecha y calcular la suma de 'Saldo' para obtener los gastos operativos totales por fecha
gastos_operativos_por_fecha = df_gastos_operativos.groupby(['fechaInicio', 'fechaFin'])['Saldo'].sum().reset_index()

# Filtrar el DataFrame para obtener solo las filas relacionadas con "PATRIMONIO" o "PATRIMONIAL"
patrimonio = df['nombreCuenta'].str.contains('PATRIMONIO|PATRIMONIAL', case=False, na=False)
df_ing = df[patrimonio]

# Agrupar por fecha y calcular la suma de 'Saldo' para obtener los ingresos por fecha
patrimonio_por_fecha = df_ing.groupby(['fechaInicio', 'fechaFin'])['Saldo'].sum().reset_index()

# Fusiona los DataFrames en base a las columnas de fecha
df_solvencia = pd.merge(patrimonio_por_fecha, activos_totales_por_fecha, on=['fechaInicio', 'fechaFin'], suffixes=('_patrimonio', '_activo_total'))

# Calcula el indicador de Solvencia Patrimonial
df_solvencia['Solvencia_Patrimonial'] = df_solvencia['Saldo_patrimonio'] / df_solvencia['Saldo_activo_total']


def generate_page(chart_type, selected_year, start_month, end_month, start_day):
    start_date = pd.Timestamp(year=int(selected_year), month=int(start_month), day=int(start_day))
    end_date = pd.Timestamp(year=int(selected_year), month=int(end_month), day=1) + pd.offsets.MonthEnd(0)
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')  # Frecuencia diaria

    if chart_type == 'ROA':
        # Filtrar los datos según el rango de fechas seleccionado
        filtered_resultados = resultados_por_fecha[resultados_por_fecha['fechaInicio'].isin(date_range)]
        filtered_activo_promedio = activo_promedio_por_fecha[activo_promedio_por_fecha['fechaInicio'].isin(date_range)]
        
        # Calcula el indicador ROA
        indicador_roa = filtered_resultados['utilidad_neta'] / filtered_activo_promedio['Saldo']
        
        # Genera la gráfica de ROA como línea
        fig = px.line(x=filtered_resultados['fechaInicio'], y=indicador_roa, labels={'x': 'Fecha de Inicio', 'y': 'ROA'}, title='ROA por Fecha de Inicio')
        fig.update_traces(line_color='#F5CA44')  # Cambia el color de la línea

        fig.update_layout(
            title={
                'text': "<b>ROA</b>",
                'y': 0.95,  # Ajusta la posición vertical del título
                'x': 0.5,   # Centra el título horizontalmente
                'xanchor': 'center',
                'yanchor': 'top',
                'font': dict(
                    family="POPPINS",  # Establece la familia de fuente
                    size=20,         # Ajusta el tamaño de la fuente
                    color="#2C68AD"  # Establece el color del título
                )
            },
            xaxis_title="Fecha de Inicio",
            yaxis_title="ROA",
            plot_bgcolor='#f8f9fa',  # Cambia el color de fondo del gráfico
        )
       
        fig_combined = go.Figure()
        # Agregar la serie de datos para la Utilidad Neta
        fig_combined.add_trace(go.Scatter(x=filtered_resultados['fechaInicio'], y=filtered_resultados['utilidad_neta'],
                                  mode='lines', name='Utilidad Neta', line=dict(color='#29609B')))

        # Agregar la serie de datos para el Activo Promedio
        fig_combined.add_trace(go.Scatter(x=filtered_activo_promedio['fechaInicio'], y=filtered_activo_promedio['Saldo'],
                                  mode='lines', name='Activo Promedio', line=dict(color='#F5CA44')))

        # Personalizar el diseño del gráfico
        fig_combined.update_layout(
            title="<b>Utilidad Neta vs Activo Promedio</b>",
            title_font_color="#2C68AD",  # Cambia el color del título
            xaxis_title="Fecha",
            title_font_size=20,              # Establece el tamaño de la fuente del título
            title_font_family='Poppins',     # Establece la familia de fuente del título
            yaxis_title="Valor",
            legend_title="Variables",
            legend=dict(x=0, y=1.1),
            plot_bgcolor='#f8f9fa',  # Cambia el color de fondo del gráfico
            xaxis=dict(gridcolor='#dddddd', gridwidth=0),  # Líneas de cuadrícula entrecortadas en el eje x
            yaxis=dict(gridcolor='#dddddd', gridwidth=0.1),   # Líneas de cuadrícula entrecortadas en el eje y
            title_x=0.5,  # Centra el título horizontalmente,
        )

        fig_list = [fig, fig_combined]
        # Texto informativo para ROA
        info_text = dcc.Markdown('''
                    **Indicador ROA:**
                    El ROA mide la rentabilidad de los activos. Un valor más alto indica mayor eficiencia en la utilización de los activos.
                ''')
        # Mostrar el valor de las variables utilizadas para calcular el indicador ROA
        variables_text = [
            f"Utilidad Neta: {filtered_resultados['utilidad_neta'].sum():,.2f}",
            f"Activo Promedio: {filtered_activo_promedio['Saldo'].sum():,.2f}"
        ]
        
        return fig_list, info_text, variables_text, filtered_resultados


    # Resto del código para otros tipos de gráficos


    elif chart_type == 'Eficiencia en Gasto Operativo':
        # Filtrar los datos según el rango de fechas seleccionado
        filtered_resultados = resultados_por_fecha[(resultados_por_fecha['fechaInicio'] >= start_date) & (resultados_por_fecha['fechaInicio'] <= end_date)]
        filtered_gastos_operativos = gastos_operativos_por_fecha[(gastos_operativos_por_fecha['fechaInicio'] >= start_date) & (gastos_operativos_por_fecha['fechaInicio'] <= end_date)]
        # Calcula el indicador de eficiencia en gasto operativo
        indicador_gasto_operativo = filtered_resultados['ingresos'] - filtered_gastos_operativos['Saldo'] / filtered_resultados['ingresos']

        # Genera la gráfica de eficiencia en gasto operativo
        fig = px.line(x=filtered_resultados['fechaInicio'], y=indicador_gasto_operativo, labels={'y': 'Eficiencia en Gasto Operativo'}, color_discrete_sequence=['#34729F'])
        fig.update_layout(
                    title={
                        'text': "<b>Eficiencia en Gasto Operativo</b>",
                        'y':0.95,  # Ajusta la posición vertical del título
                        'x':0.5,   # Centra el título horizontalmente
                        'xanchor': 'center',
                        'yanchor': 'top',
                        'font':dict(
                            family="POPPINS",  # Establece la familia de fuente
                            size=20,         # Ajusta el tamaño de la fuente
                            color="#2C68AD"  # Establece el color del título
                        )
                    },
                    xaxis_title="Fecha",
                    yaxis_title="Eficiencia",
                    plot_bgcolor='#f8f9fa',
                )
        fig_combined_gastos = go.Figure()

        # Agregar la serie de datos para el indicador de eficiencia en gasto operativo
        fig_combined_gastos.add_trace(go.Scatter(x=filtered_resultados['fechaInicio'], y=filtered_resultados['ingresos'], mode='lines', name='Eficiencia en Gasto Operativo', line=dict(color='#29609B')))
        # Agregar la serie de datos para los ingresos
        fig_combined_gastos.add_trace(go.Scatter(x=filtered_gastos_operativos['fechaInicio'], y=filtered_gastos_operativos['Saldo'], mode='lines', name='Ingresos', line=dict(color='#F5CA44')))
        # Personalizar el diseño del gráfico
        fig_combined_gastos.update_layout(
            title="<b>Utilidad Neta vs Eficiencia en Gastos Operativos</b>",
            title_font_color="#2C68AD",  # Cambia el color del título
            xaxis_title="Fecha",
            title_font_size=20,              # Establece el tamaño de la fuente del título
            title_font_family='Poppins',     # Establece la familia de fuente del título
            yaxis_title="Valor",
            legend_title="Variables",
            legend=dict(x=0, y=1.1),
            plot_bgcolor='#f8f9fa',  # Cambia el color de fondo del gráfico
            xaxis=dict(gridcolor='#dddddd', gridwidth=0),  # Líneas de cuadrícula entrecortadas en el eje x
            yaxis=dict(gridcolor='#dddddd', gridwidth=0.1),   # Líneas de cuadrícula entrecortadas en el eje y
            title_x=0.5,  # Centra el título horizontalmente,
        )
        fig_list = [fig, fig_combined_gastos]
        # Texto informativo para Eficiencia en Gasto Operativo
        info_text = dcc.Markdown('''
                    **Indicador de Eficiencia en Gasto Operativo:**
                    Este indicador muestra la eficiencia en la gestión de gastos operativos en comparación con los ingresos.
                ''')
        # Mostrar el valor de las variables utilizadas para calcular el indicador de eficiencia en gasto operativo
        variables_text = [
            f"Utilidad Neta Total: {filtered_resultados['utilidad_neta'].sum():,.2f}",
            f"Gastos Operativos Totales: {filtered_gastos_operativos['Saldo'].sum():,.2f}"
        ]

        
        return fig_list, info_text, variables_text, filtered_resultados

    elif chart_type == 'Solvencia Patrimonial':
        # Filtrar los datos según el rango de fechas seleccionado
        filtered_patrimonio = patrimonio_por_fecha[(patrimonio_por_fecha['fechaInicio'] >= start_date) & (patrimonio_por_fecha['fechaInicio'] <= end_date)]
        filtered_activos_totales = activos_totales_por_fecha[(activos_totales_por_fecha['fechaInicio'] >= start_date) & (activos_totales_por_fecha['fechaInicio'] <= end_date)]

        # Calcula la solvencia patrimonial
        filtered_patrimonio['Solvencia_Patrimonial'] = filtered_patrimonio['Saldo'] / filtered_activos_totales['Saldo']

        # Genera la gráfica de solvencia patrimonial
        fig = px.bar(filtered_patrimonio, x='fechaInicio', y='Solvencia_Patrimonial', labels={'y': 'Solvencia Patrimonial'}, color_discrete_sequence=['#34729F'])

        fig.update_layout(
            title={
                'text': "<b>Solvencia Patrimonial</b>",
                'y':0.95,  # Ajusta la posición vertical del título
                'x':0.5,   # Centra el título horizontalmente
                'xanchor': 'center',
                'yanchor': 'top',
                'font':dict(
                    family="POPPINS",  # Establece la familia de fuente
                    size=20,         # Ajusta el tamaño de la fuente
                    color="#2C68AD"  # Establece el color del título
                )
            },
            xaxis_title="Fecha",
            yaxis_title="Solvencia Patrimonial",
            plot_bgcolor='#f8f9fa',
        )

        # Otro tipo de gráfico para solvencia patrimonial que utiliza las variables para calcular el indicador
        fig_solvencia = go.Figure()

        # Agregar la serie de datos para el patrimonio total
        fig_solvencia.add_trace(go.Scatter(x=filtered_patrimonio['fechaInicio'], y=filtered_patrimonio['Saldo'],
                                            mode='lines', name='Patrimonio Total', line=dict(color='#29609B')))

        # Agregar la serie de datos para los activos totales
        fig_solvencia.add_trace(go.Scatter(x=filtered_activos_totales['fechaInicio'], y=filtered_activos_totales['Saldo'],
                                            mode='lines', name='Activo Total', line=dict(color='#F5CA44')))

        # Personalizar el diseño del gráfico
        fig_solvencia.update_layout(
            title="<b>Patrimonio vs Activo Total</b>",
            title_font_color="#2C68AD",  # Cambia el color del título
            xaxis_title="Fecha",
            title_font_size=20,              # Establece el tamaño de la fuente del título
            title_font_family='Poppins',     # Establece la familia de fuente del título
            yaxis_title="Valor",
            legend_title="Variables",
            legend=dict(x=0, y=1.1),
            plot_bgcolor='#f8f9fa',  # Cambia el color de fondo del gráfico
            xaxis=dict(gridcolor='#dddddd', gridwidth=0),  # Líneas de cuadrícula entrecortadas en el eje x
            yaxis=dict(gridcolor='#dddddd', gridwidth=0.1),   # Líneas de cuadrícula entrecortadas en el eje y
            title_x=0.5,  # Centra el título horizontalmente,
        )

        fig_list = [fig, fig_solvencia]
        # Texto informativo para Solvencia Patrimonial
        info_text = dcc.Markdown('''
                    **Indicador de Solvencia Patrimonial:**
                    La solvencia patrimonial mide la capacidad de la empresa para cumplir con sus obligaciones a largo plazo.
                ''')
        # Mostrar el valor de las variables utilizadas para calcular el indicador de solvencia patrimonial
        variables_text = [
            f"Patrimonio Total: {filtered_patrimonio['Saldo'].sum():,.2f}",
            f"Activo Total: {filtered_activos_totales['Saldo'].sum():,.2f}"
        ]

       

        return fig_list, info_text, variables_text, filtered_patrimonio

@app.callback(
    [Output("page-content", "children"),
     Output("download", "data")], 
    [Input("url", "pathname"), 
     Input("year-dropdown", "value"), 
     Input("start-month-dropdown", "value"), 
     Input("end-month-dropdown", "value"),
     Input("start-day-dropdown", "value"),
     Input("export-button", "n_clicks")])
def render_page_content(pathname, selected_year, start_month, end_month, start_day, n_clicks):
    if pathname == "/":
        
        return dcc.Location(id='redirect', pathname='/ROA', refresh=True), None
    else:
        chart_type = pathname.split('/')[-1].replace('_', ' ')
        
        fig_list, info_text, variables_text, df = generate_page(chart_type, selected_year, start_month, end_month, start_day)
        
        # Generar tarjetas con texto informativo y variables
        info_cards = generate_info_cards(info_text, variables_text)
        
        # Crear una lista para almacenar todas las partes del contenido
        content_parts = [
            html.H1(chart_type),
           
            html.Div(info_cards),  # Agregar las tarjetas al contenido
        ]
        
        # Iterar sobre las figuras y agregarlas al contenido
        for fig in fig_list:
            content_parts.append(dcc.Graph(figure=fig))

        if n_clicks:
            with pd.ExcelWriter('output.xlsx', engine='openpyxl') as excel_writer:
                sheet_name = 'All_Figures'
                for i, fig in enumerate(fig_list):
                    fig.write_image(f'figure_{i}.png')  # Guardar la figura como imagen
                # Agregar todas las imágenes al archivo de Excel en la misma hoja
                sheet = excel_writer.book.create_sheet(title=sheet_name)
                df.to_excel(excel_writer, sheet_name=sheet_name, index=False)
                for i, fig in enumerate(fig_list):
                    img = Image(f'figure_{i}.png')
                    sheet.add_image(img, f'J{(i * 20) + 1}')  # Ajustar posición de las imágenes
            return html.Div(content_parts), dcc.send_file('output.xlsx')  # Enviar el archivo Excel para descargar

        return html.Div(content_parts), None


default_year = anos_unicos[0]  # Seleccionar el primer año en la lista como predeterminado
months = ['Enero', 'Febrero', 'Marzo', 'Abril', 'Mayo', 'Junio', 'Julio', 'Agosto', 'Septiembre', 'Octubre', 'Noviembre', 'Diciembre']
# Establecer el mes inicial y final predeterminado
default_start_month = 1
default_end_month = 12

# Estilo de la barra lateral
SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "16rem",
    "padding": "2rem 1rem",
    "background-color": "#f8f9fa",
}

# Estilo del contenido principal
CONTENT_STYLE = {
    "margin-left": "18rem",
    "margin-right": "2rem",
    "padding": "2rem 1rem",
}
sidebar = html.Div(
    [
        html.Img(src="https://lorente.fin.ec/wp-content/uploads/2021/06/logo_lorente_web.png", style={"width": "100%", "margin-bottom": "20px"}),
        html.Hr(),
        html.P(
            "Indicadores a tu mano", className="lead"
        ),
        dbc.Nav(
            [
                dbc.NavLink([html.I(className="fas fa-chart-line mr-2 "), "    ROA"], href="/ROA", active="exact"),
                dbc.NavLink([html.I(className="fas fa-dollar-sign mr-2 "), "    Eficiencia en Gasto Operativo"], href="/Eficiencia_en_Gasto_Operativo", active="exact"),
                dbc.NavLink([html.I(className="fas fa-balance-scale mr-2 "), "    Solvencia Patrimonial"], href="/Solvencia_Patrimonial", active="exact"),
            ],
            vertical=True,
            pills=True,
        ),

        html.Label('Seleccione el año:'),
        dcc.Dropdown(
            id='year-dropdown',
            options=[{'label': str(year), 'value': str(year)} for year in anos_unicos],
            value=str(anos_unicos[-1])  # Último año por defecto
        ),
        html.Label('Seleccione el mes de inicio:'),
        dcc.Dropdown(
            id='start-month-dropdown',
            options=[{'label': month_name, 'value': str(month)} for month, month_name in enumerate(meses_unicos, start=1)],
            value=str(meses_unicos[0])  # Primer mes por defecto
        ),
        html.Label('Seleccione el mes de fin:'),
        dcc.Dropdown(
            id='end-month-dropdown',
            options=[{'label': month_name, 'value': str(month)} for month, month_name in enumerate(meses_unicos, start=1)],
            value=str(meses_unicos[-1])  # Último mes por defecto
        ),
        html.Label('Seleccione el día de inicio:'),
        dcc.Dropdown(
            id='start-day-dropdown',
            options=[{'label': str(day), 'value': str(day)} for day in dias_unicos],
            value=str(dias_unicos[0])  # Primer día por defecto
        ),
        html.Hr(),

    dcc.Download(id="download"),  # Componente para descargar archivos
    dbc.Button("Exportar a Excel", id="export-button", color="primary", className="mr-2"),


    ],
    style=SIDEBAR_STYLE,
)
@app.callback(
    Output('start-day-dropdown', 'options'),
    [Input('year-dropdown', 'value'),
     Input('start-month-dropdown', 'value')]
)
def update_day_options(selected_year, selected_month):
    # Convertir las fechas únicas a un DataFrame de pandas
    unique_dates_df = pd.DataFrame(pd.to_datetime(todas_las_fechas), columns=['fecha'])
    print('update_day_options',selected_year, selected_month)
    # Filtrar las fechas por año y mes seleccionados
    filtered_dates_df = unique_dates_df[(unique_dates_df['fecha'].dt.year == int(selected_year)) & (unique_dates_df['fecha'].dt.month == int(selected_month))]
    # Obtener los días únicos del DataFrame filtrado
    filtered_days = filtered_dates_df['fecha'].dt.day.unique()
    # Generar las opciones para el dropdown del día
    day_options = [{'label': str(day), 'value': str(day)} for day in filtered_days]
    
    return day_options

def generate_info_cards(info_text, variables_text):
    # Crear una lista para almacenar todas las partes del contenido
    content_parts = []
    
    # Agregar el texto informativo directamente sin tarjeta
    content_parts.append(html.P(info_text))
    
    # Crear tarjetas para cada variable y colocarlas una al lado de la otra
    variable_card_columns = []
    for variable, color in zip(variables_text, ["primary", "success"]):  # Asignar un color a cada tarjeta de variable
        variable_card = dbc.Card(
            dbc.CardBody(
                [
                    html.H5(className="card-title"),
                    html.P(variable, className="card-text"),
                ]
            ),
            color="#F5CA44",  # Usar un color diferente para cada tarjeta de variable
            outline=True,
            style={"margin-right": "10px"}  # Ajustar el margen derecho entre las tarjetas
        )
        variable_card_columns.append(dbc.Col(variable_card, width=6))  # Colocar cada tarjeta de variable en una columna de ancho 6
    
    # Agregar las columnas de tarjetas de variables en una fila
    content_parts.append(dbc.Row(variable_card_columns))

    return content_parts

content = html.Div(id="page-content", style=CONTENT_STYLE)

# Layout de la aplicación
app.layout = html.Div([dcc.Location(id="url"), sidebar, content])

if __name__ == "__main__":
    app.run_server(debug=True)
