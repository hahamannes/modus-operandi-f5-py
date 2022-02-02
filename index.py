from dash import dcc
from dash import html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc

# must add this line in order for the app to be deployed successfully on Heroku
#from app import server

from app import app

# import all pages in the app
from apps import about, home, LDA_Plot, LSA_topics, heatmap, tf_idf_avg_plot, scattertext


# building the navigation bar
# https://github.com/facultyai/dash-bootstrap-components/blob/master/examples/advanced-component-usage/Navbars.py

navbar = dbc.Navbar(
    dbc.Container(
        [
            html.A(
                # Use row and col to control vertical alignment of logo / brand
                dbc.Row(
                    [
                        dbc.Col(dbc.NavbarBrand((html.Img(src="/assets/logo-uva.png", height="30px")), href="https://www.uva.nl/en")),
                        dbc.Col(dbc.NavbarBrand((html.Img(src="/assets/politie-academie-logo.png", height="30px")), href="https://www.politieacademie.nl")),
                        dbc.Col(dbc.NavbarBrand((html.Img(src="/assets/TNO-logo.png", height="30px")), href="https://www.tno.nl/en/")),
                        dbc.Col(dbc.NavbarBrand("Home", href="/home", style={"marginLeft": "280px", "font-size":"12px"})),
                        dbc.Col(dbc.NavbarBrand("LDA", href="/LDA", style={"font-size":"12px"})),
                        dbc.Col(dbc.NavbarBrand("LSA", href="/LSA", style={"font-size":"12px"})),
                        dbc.Col(dbc.NavbarBrand("Heatmap", href="/Heatmap", style={"font-size":"12px"})),
                        dbc.Col(dbc.NavbarBrand("TF-IDF", href="/TF-IDF", style={"font-size":"12px"})),
                        dbc.Col(dbc.NavbarBrand("scattertext", href="/scattertext", style={"font-size":"12px"})),
                    ], align="center"
                    #no_gutters=True,
                ),
                href="/home",
            ),
#            dbc.NavbarToggler(id="navbar-toggler2"),
        ]
    ),
    color="#D6EAF8",
    className="mb-4",
)

#def toggle_navbar_collapse(n, is_open):
#    if n:
#        return not is_open
#    return is_open

#for i in [2]:
#    app.callback(
#        Output(f"navbar-collapse{i}", "is_open"),
#        [Input(f"navbar-toggler{i}", "n_clicks")],
#        [State(f"navbar-collapse{i}", "is_open")],
#    )(toggle_navbar_collapse)

# embedding the navigation bar
app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    navbar,
    html.Div(id='page-content')
])


@app.callback(Output('page-content', 'children'),
              [Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/LDA':
        return LDA_Plot.layout
    elif pathname == '/LSA':
        return LSA_topics.layout
    elif pathname == '/Heatmap':
        return heatmap.layout
    elif pathname == '/TF-IDF':
        return tf_idf_avg_plot.layout
    elif pathname == '/scattertext':
        return scattertext.layout
    else:
        return home.layout

if __name__ == '__main__':
    app.run_server(host='127.0.0.1', debug=True)



#    http://127.0.0.1:8050/

