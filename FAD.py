from fastapi import FastAPI, Request, Response
import requests
from fastapi import HTTPException
import os
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
import io
from io import BytesIO
import base64
import json
from fastapi.responses import JSONResponse
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import uvicorn
from pydantic import BaseModel
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from fastapi.responses import HTMLResponse
import ast
import glob
import folium
import math
from bs4 import BeautifulSoup

app = FastAPI()


@app.get('/', response_class=HTMLResponse)
def index():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Farmer Analytical Dashboard</title>
        <style>
            body {
                background-color: #f4f4f2; /* Light grey background for a softer look */
                color: #333; /* Dark grey text for better readability */
                font-family: 'Arial', sans-serif;
                text-align: center;
                padding-top: 50px;
                margin: 0;
                line-height: 1.6;
            }
            h1 {
                color: #4CAF50; /* Green color for a fresh, agricultural feel */
                font-size: 3rem; /* Larger font size for the main heading */
                margin-bottom: 20px;
            }
            img {
                max-width: 100%; /* Responsive image size */
                height: auto;
                border-radius: 8px; /* Rounded corners for the image */
                margin-bottom: 20px;
            }
            p {
                font-size: 1.2rem;
                max-width: 800px; /* Max width for paragraphs to control line length */
                margin: 20px auto; /* Center the paragraphs */
            }
            a {
                color: #4CAF50;
                text-decoration: none; /* No underline on links */
                font-weight: bold;
            }
            a:hover {
                text-decoration: underline; /* Underline on hover for better UX */
            }
            footer {
                margin-top: 40px;
                padding: 20px;
                background-color: #333;
                color: #fff;
                position: absolute;
                bottom: 0;
                width: 100%;
            }
        </style>
    </head>
    <body>
        <h1>Farmer Analytical Dashboard</h1>
        
        <!-- Correct the image source by providing the URL directly -->
        <img src="https://venturebeat.com/wp-content/uploads/2022/05/GettyImages-1318237749.jpg?fit=2119%2C1414&strip=allr" alt="Farmer Analytical Dashboard">
        
        <p>
            Analysis of Weather, Soil, Crop Cycle, Crop Price.
        </p>
        <p>
            Go to <a href="http://127.0.0.1:8000/docs">docs</a> to explore the full list of available endpoints.
        </p>
        <p>
             <a href="/map">Maps</a> 
        </p>
        <p>
             <a href="/market">Market</a> 
        </p>
        <p>
             <a href="/weather">Weather</a> 
        </p>
        <p>
             <a href="/events">Events</a> 
        </p>
        <p>
             <a href="/productionplot">Production of Crop</a> 
        </p>
        <p>
             <a href="/rankplot">Rank of US(Cropwise)</a> 
        </p>
        <p>
             <a href="/plot">Production Analysis</a> 
        </p>
        <p>
             <a href="/loan">Farmer Loan</a> 
        </p>
        <footer>
            <p>
                Built by <a href="mailto:pridhvipinninti@gmail.com">Pridhvi</a>.
            </p>
        </footer>
    </body>
    </html>
"""


df = pd.read_excel('/Users/pridhvipinninti/Downloads/Farmer_DW.xlsx')

locations = df[["Latitude", "Longitude", "Organization"]]


def haversine(lat1, lon1, lat2, lon2):
    """
        Calculate the great-circle distance between two points
        on the Earth's surface given their latitude and longitude
        using the Haversine formula.

        Args:
            lat1 (float): Latitude of point 1 in degrees.
            lon1 (float): Longitude of point 1 in degrees.
            lat2 (float): Latitude of point 2 in degrees.
            lon2 (float): Longitude of point 2 in degrees.

        Returns:
            float: Distance between the two points in miles.

        References:
        1. Calculate Geographic Distances in Python with the Haversine Method
           Link: https://louwersj.medium.com/calculate-geographic-distances-in-python-with-the-haversine-method-ed99b41ff04b

        """
    lat1 = math.radians(lat1)
    lon1 = math.radians(lon1)
    lat2 = math.radians(lat2)
    lon2 = math.radians(lon2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    R = 6371.0
    distance_km = R * c
    distance_miles = distance_km * 0.621371

    return distance_miles


my_location = [37.548271, -121.988571]


@app.get("/map", response_class=HTMLResponse)
async def market_map():
    """
       Generate and display a market map with markers for 'my_location' and other latitude and longitudes from Dummy data

       Returns:
           HTMLResponse: HTML content with the market map.
    """
    x = folium.Map(location=my_location, zoom_start=11, control_scale=True)
    folium.Marker(
        my_location,
        popup="My Location",
        icon=folium.Icon(color='red')
    ).add_to(x)
    for i, location_info in locations.iterrows():
        distance_miles = haversine(my_location[0], my_location[1], location_info["Latitude"],
                                   location_info["Longitude"])
        popup_text = f"{location_info['Organization']} - Distance: {distance_miles:.2f} miles"
        folium.Marker(
            [location_info["Latitude"], location_info["Longitude"]],
            popup=popup_text
        ).add_to(x)
    x.save("map.html")
    with open("map.html", "r") as f:
        html_content = f.read()

    return HTMLResponse(content=html_content, status_code=200)

d = pd.read_excel('/Users/pridhvipinninti/Downloads/Farmer_DW.xlsx')
columns_to_print = [
    'Organization', 'First Name', 'Last Name', 'Street Address', 'City',
    'State', 'Zip Code', 'Email', 'Phone', 'website', 'Days open',
    'year round', 'Season 1', 'Time 1', 'Season 2', 'Time 2', 'Programs Accepted'
]


data_d = d[columns_to_print].to_dict(orient='records')


@app.get("/market", response_class=HTMLResponse)
async def index(request: Request):
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Farmers Market Data</title>
        <style>
            .card {
                border: 1px solid #ccc;
                border-radius: 5px;
                padding: 15px;
                margin-bottom: 20px;
                box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
                background-color: #f9f9f9;
            }
            .card p {
                margin: 5px 0;
            }
            .card p strong {
                font-weight: bold;
                margin-right: 5px;
            }
        </style>
    </head>
    <body>
        <h1 style="text-align: center;">Farmers Data</h1>
    """

    for row in data_d:
        html_content += """
        <div class="card">
            <p><strong>Organization:</strong> {Organization}</p>
            <p><strong>First Name:</strong> {First Name}</p>
            <p><strong>Last Name:</strong> {Last Name}</p>
            <p><strong>Street Address:</strong> {Street Address}</p>
            <p><strong>City:</strong> {City}</p>
            <p><strong>State:</strong> {State}</p>
            <p><strong>Zip Code:</strong> {Zip Code}</p>
            <p><strong>Email:</strong> {Email}</p>
            <p><strong>Phone:</strong> {Phone}</p>
            <p><strong>Website:</strong> {website}</p>
            <p><strong>Days Open:</strong> {Days open}</p>
            <p><strong>Year Round:</strong> {year round}</p>
            <p><strong>Season 1:</strong> {Season 1}</p>
            <p><strong>Time 1:</strong> {Time 1}</p>
            <p><strong>Season 2:</strong> {Season 2}</p>
            <p><strong>Time 2:</strong> {Time 2}</p>
            <p><strong>Programs Accepted:</strong> {Programs Accepted}</p>
        </div>
        """.format(**row)

    html_content += """
    </body>
    </html>
    """

    return HTMLResponse(content=html_content)


def weather_plot(data):
    """
        Generate HTML content for a weather data plot using Plotly.

        Args:
            data (list): List of dictionaries containing weather data.
                         Each dictionary should contain keys: "date", "tavg", "tmin", "tmax", "prcp", "wspd".

        Returns:
            str: HTML content containing the weather data plot.

        Raises:
            HTTPException: If an error occurs during plot generation.
    """
    try:
        dates = [entry["date"] for entry in data]
        tavg = [entry["tavg"] for entry in data]
        tmin = [entry["tmin"] for entry in data]
        tmax = [entry["tmax"] for entry in data]
        prcp = [entry["prcp"] for entry in data]
        wspd = [entry["wspd"] for entry in data]

        dates_json = json.dumps(dates)
        tavg_json = json.dumps(tavg)
        tmin_json = json.dumps(tmin)
        tmax_json = json.dumps(tmax)
        prcp_json = json.dumps(prcp)
        wspd_json = json.dumps(wspd)

        html_content = f"""
        <html>
            <head>
                <title>Weather Data Plot</title>
                <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            </head>
            <body>
                <h1>Weather Data Plot</h1>
                <div id="plot"></div>
                <script>
                    var dates = {dates_json};
                    var tavg = {tavg_json};
                    var tmin = {tmin_json};
                    var tmax = {tmax_json};
                    var prcp = {prcp_json};
                    var wspd = {wspd_json};

                    var trace1 = {{
                        x: dates,
                        y: tavg,
                        mode: 'lines+markers',
                        name: 'Average Temperature'
                    }};
                    var trace2 = {{
                        x: dates,
                        y: tmin,
                        mode: 'lines+markers',
                        name: 'Minimum Temperature'
                    }};
                    var trace3 = {{
                        x: dates,
                        y: tmax,
                        mode: 'lines+markers',
                        name: 'Maximum Temperature'
                    }};
                    var trace4 = {{
                        x: dates,
                        y: prcp,
                        mode: 'lines+markers',
                        name: 'Precipitation'
                    }};
                    var trace5 = {{
                        x: dates,
                        y: wspd,
                        mode: 'lines+markers',
                        name: 'Wind Speed'
                    }};

                    var layout = {{
                        title: 'Weather Data',
                        xaxis: {{ title: '' }},
                        yaxis: {{ title: 'Measurement' }},
                        legend: {{ "orientation": "h" }},
                        margin: {{ l: 50, r: 50, b: 50, t: 20 }},
                    }};

                    var data = [trace1, trace2, trace3, trace4, trace5];

                    Plotly.newPlot('plot', data, layout);
                </script>
            </body>
        </html>
        """
        return html_content
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/weather")
def get_weather_plot():
    """
        Endpoint to fetch weather data and generate the weather data plot.

        We are plotting data of Temperature,Humidity, Precipitation for 3 days from Today
    """
    try:
        url = "https://meteostat.p.rapidapi.com/point/daily"
        querystring = {
            "lat": "37.548271",
            "lon": "-121.988571",
            "start": "2024-03-25",
            "end": "2024-03-27",
            "alt": "30"
        }
        headers = {
            "X-RapidAPI-Key": "e91801946cmshe2cd563f1afe85ep1badbfjsnbf2cf1304e18",
            "X-RapidAPI-Host": "meteostat.p.rapidapi.com"
        }
        response = requests.get(url, headers=headers, params=querystring)
        data = response.json()["data"]
        html_plot = weather_plot(data)

        return Response(content=html_plot, media_type="text/html")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/get_OHLC")
async def three():
    """
           We are extracting data of Highest Price , Lowest Price , Open Market Price & Close Market Price of the crop for a given Year
           We are Displaying Data of Crop - "COTTON" and Base Symbol(Currency)- "USD".
    """
    try:
        url = "https://commodity-rates-api.p.rapidapi.com/open-high-low-close/2022-01-10"
        querystring = {"base": "USD", "symbols": "COTTON"}
        headers = {
            "X-RapidAPI-Key": "e91801946cmshe2cd563f1afe85ep1badbfjsnbf2cf1304e18",
            "X-RapidAPI-Host": "commodity-rates-api.p.rapidapi.com"
        }
        response = requests.get(url, headers=headers, params=querystring)
        data = response.json()

        html_content = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>OHLC Data</title>
            <style>
                .card {{
                    border: 1px solid #ccc;
                    border-radius: 5px;
                    padding: 10px;
                    margin: 10px;
                    width: 300px;
                    background-color: #f9f9f9;
                    box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2);
                }}
                .card h3 {{
                    color: #333;
                }}
                .card p {{
                    margin: 5px 0;
                }}
                .card strong {{
                    font-weight: bold;
                }}
            </style>
        </head>
        <body>
            <div class="card">
                <h3>OHLC Data for {data['date']}</h3>
                <p><strong>Symbol:</strong> {data['symbol']}</p>
                <p><strong>Base:</strong> {data['base']}</p>
                <p><strong>Open:</strong> {round(data['rates']['open'], 2)}</p>
                <p><strong>High:</strong> {round(data['rates']['high'], 2)}</p>
                <p><strong>Low:</strong> {round(data['rates']['low'], 2)}</p>
                <p><strong>Close:</strong> {round(data['rates']['close'], 2)}</p>
                <p><strong>Unit:</strong> {data['unit']}</p>
            </div>
        </body>
        </html>
        """
        return HTMLResponse(content=html_content, status_code=200)
    except Exception as e:
        return f"Error: {e}"


class FolderPath(BaseModel):
    folder_path: str


grad = GradientBoostingClassifier()
rf = RandomForestClassifier()
svc = SVC(probability=True)
log_reg = make_pipeline(StandardScaler(), LogisticRegression(max_iter=10000))


async def train_models():
    """
        We are giving Input Crop, for which we are extracting data from a CSV File and performing basic data analysis
        We are using 4 Classifiers Gradient Booster, Random Forest, SVC and Logistic Regression


    """
    global grad, rf, svc, log_reg
    data = pd.read_csv("/Users/pridhvipinninti/Downloads/ANLT-221/Crop_recommendation.csv")
    X = data.drop("label", axis=1)
    y = data["label"]

    grad.fit(X, y)
    rf.fit(X, y)
    svc.fit(X, y)
    log_reg.fit(X, y)


@app.on_event("startup")
async def startup_event():
    """
        Run Models before calling GET predict
    """
    await train_models()


@app.post("/predict", response_class=HTMLResponse)
async def predict(data: FolderPath):
    """
    We are calculating probability and taking average of them as final probability for Classification.
    """
    folder_path = data.folder_path
    all_data = []

    file_pattern = os.path.join(folder_path, '*.txt')

    for file_path in glob.glob(file_pattern):
        with open(file_path, 'r') as file:
            data_str = file.read().strip()
            data_list = ast.literal_eval(data_str)
            all_data.append(data_list)

    columns = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
    test_df = pd.DataFrame(all_data, columns=columns)

    probabilities = (grad.predict_proba(test_df) +
                     rf.predict_proba(test_df) +
                     svc.predict_proba(test_df) +
                     log_reg.predict_proba(test_df)) / 4

    class_probabilities = probabilities.max(axis=1)
    test_df['Probability'] = class_probabilities
    location_labels = ["Location" + str(i) for i in range(1, len(test_df) + 1)]
    test_df['Location'] = location_labels

    test_df['Probability (%)'] = test_df['Probability'] * 100

    ranked_test = test_df.sort_values(by='Probability', ascending=False)

    ranked_test['Rank'] = range(1, len(ranked_test) + 1)

    top_locations = ranked_test.head(15)[["Location", 'Rank', 'Probability (%)']].to_dict(orient="records")

    return JSONResponse(content={"top_locations": top_locations})


@app.get("/priceplot", response_class=HTMLResponse)
async def plot_crop(request: Request, crop: str):
    """
        Plotting 4 plots of Production, Consumption, Imports & Exports
    """

    D2 = pd.read_csv("/Users/pridhvipinninti/Downloads/ANLT-221/D2.csv")
    Plot_Data = D2[['Country', 'Commodity', 'Variable', 'TIME', 'PowerCode', 'Value']]
    crop_data = Plot_Data[Plot_Data['Commodity'] == crop]

    if crop_data.empty:
        return {"message": "Crop data not found!"}
    imports_data = crop_data[crop_data['Variable'] == 'Imports']
    exports_data = crop_data[crop_data['Variable'] == 'Exports']
    production_data = crop_data[crop_data['Variable'] == 'Production']
    consumption_data = crop_data[crop_data['Variable'] == 'Consumption']
    area_harvested_data = crop_data[crop_data['Variable'] == 'Area harvested']
    producer_price_data = crop_data[crop_data['Variable'] == 'Producer price']

    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Wheat Data Analysis')

    axs[0, 0].plot(imports_data['TIME'], imports_data['Value'], label='Imports', marker='o')
    axs[0, 0].plot(exports_data['TIME'], exports_data['Value'], label='Exports', marker='x')
    axs[0, 0].set_title('Imports vs. Exports')
    axs[0, 0].set_xlabel('Year')
    axs[0, 0].set_ylabel('Quantity (in thousands of tons)')
    axs[0, 0].legend()
    axs[0, 0].grid(True, linestyle='--', linewidth=0.5)

    axs[0, 1].plot(production_data['TIME'], production_data['Value'], label='Production', marker='o', linestyle='-')
    axs[0, 1].plot(consumption_data['TIME'], consumption_data['Value'], label='Consumption', marker='x', linestyle='-')
    axs[0, 1].set_title('Production vs. Consumption')
    axs[0, 1].set_xlabel('Year')
    axs[0, 1].set_ylabel('Quantity (in thousands of tons)')
    axs[0, 1].legend()
    axs[0, 1].grid(True, linestyle='--', linewidth=0.5)

    axs[1, 0].plot(area_harvested_data['TIME'], area_harvested_data['Value'], label='Area Harvested', marker='o',
                   linestyle='-')
    axs[1, 0].set_title('Area Harvested')
    axs[1, 0].set_xlabel('Year')
    axs[1, 0].set_ylabel('Area (in thousands of hectares)')
    axs[1, 0].legend()
    axs[1, 0].grid(True, linestyle='--', linewidth=0.5)

    axs[1, 1].plot(producer_price_data['TIME'], producer_price_data['Value'], label='Producer Price', marker='o',
                   linestyle='-')
    axs[1, 1].set_title('Producer Price')
    axs[1, 1].set_xlabel('Year')
    axs[1, 1].set_ylabel('Price (USD per Ton)')
    axs[1, 1].legend()
    axs[1, 1].grid(True, linestyle='--', linewidth=0.5)

    html_bytes = BytesIO()
    plt.savefig(html_bytes, format='png')
    plt.close()
    html_bytes.seek(0)
    plot_url = base64.b64encode(html_bytes.read()).decode('utf-8')
    html_plot = f'<img src="data:image/png;base64,{plot_url}" />'

    html_content = """
    <html>
        <head>
            <title>Crop Data Analysis</title>
        </head>
        <body>
            <h1>{}</h1>
            {}
        </body>
    </html>
    """.format(f'{crop} Data Analysis', html_plot)

    return HTMLResponse(content=html_content, status_code=200)


# http://127.0.0.1:8000/plot?crop=Wheat
def extract_organization_data(html_content):
    """
        Extracts organization and event details from the given HTML content.

        Parses the HTML to find organization names and their respective event details,
        including 'What', 'Where', and 'When' information.

        Parameters:
        - html_content (str): The HTML content as a string.

        Returns:
        - list of dicts: A list where each element is a dictionary containing the organization name
          and a nested dictionary of event details ('What', 'Where', 'When').
          If an error occurs during extraction, an empty list is returned.
    """
    try:
        soup = BeautifulSoup(html_content, "html.parser")
        h4_elements = soup.find_all('h4', class_='separator-line-after')
        p_elements = [h4.find_next_sibling('p') for h4 in h4_elements]
        organizations_data = []
        for h4, p in zip(h4_elements, p_elements):
            organization = h4.get_text(strip=True).replace("Organization: ", "")

            event_info = {}
            for line in p.stripped_strings:
                if "What:" in line:
                    event_info["What"] = line.replace("What:", "").strip()
                elif "Where:" in line:
                    event_info["Where"] = line.replace("Where:", "").strip()
                elif "When:" in line:
                    event_info["When"] = line.replace("When:", "").strip()

            organizations_data.append({"organization": organization, "event_details": event_info})

        return organizations_data
    except Exception as e:
        print(f"Error occurred while extracting organization data: {e}")
        return []


@app.get("/events")
async def events_data():
    """
        Fetches and displays organization and event details from a specified URL.

        This function requests the HTML content from a given URL, extracts the data using
        `extract_organization_data`, and then formats it into an HTML table for display.

        Returns:
        - HTMLResponse: An HTML page containing a table of organizations and their event details.
          If an error occurs during fetching or processing, a 500 HTTP exception is raised.
    """
    url = "https://cafe.ucr.edu/california-sustainable-agriculture-food-events"
    try:
        response = requests.get(url)
        response.raise_for_status()
        html_content = response.text
        organizations_data = extract_organization_data(html_content)

        html_output = """
        <html>
        <head>
            <title>Organizations and Event Details</title>
            <style>
                table {
                    width: 100%;
                    border-collapse: collapse;
                }
                th, td {
                    padding: 8px;
                    text-align: left;
                    border-bottom: 1px solid #ddd;
                }
                th {
                    background-color: #f2f2f2;
                }
            </style>
        </head>
        <body>
            <h1>Organizations and Event Details</h1>
            <table>
                <thead>
                    <tr>
                        <th>Organization</th>
                        <th>What</th>
                        <th>Where</th>
                        <th>When</th>
                    </tr>
                </thead>
                <tbody>
        """

        for data in organizations_data:
            html_output += f"""
                <tr>
                    <td>{data['organization']}</td>
                    <td>{data['event_details'].get('What', 'N/A')}</td>
                    <td>{data['event_details'].get('Where', 'N/A')}</td>
                    <td>{data['event_details'].get('When', 'N/A')}</td>
                </tr>
            """

        html_output += """
                </tbody>
            </table>
        </body>
        </html>
        """

        return HTMLResponse(content=html_output, status_code=200)

    except Exception as e:
        print(f"Error occurred while fetching HTML content: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch HTML content")


@app.get("/productionplot", response_class=HTMLResponse)
async def production_plot():
    """
        Generates and displays a horizontal bar plot of the United States agriculture production in 2018.

        This function scrapes data from the specified Wikipedia page, specifically focusing on the section
        that outlines the US's agriculture production in 2018. It extracts categories of agriculture products
        and their corresponding production values. These are then plotted as a horizontal bar chart,
        which is encoded to Base64 and embedded directly into an HTML response for display.

        Returns:
        - HTML content: An HTML page containing a bar plot image of the US agriculture production in 2018.
          If an error occurs during the data fetching or processing, a 500 HTTP exception is raised with the error detail.
    """
    try:
        url = "https://en.wikipedia.org/wiki/Agriculture_in_the_United_States"
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        section = soup.find('span', id="United_States_agriculture_production_in_2018")
        list_items = section.find_next('ul').find_all('li')

        categories = []
        production = []

        for item in list_items:
            text = item.get_text()
            if 'world producer of' in text:
                category = text.split('world producer of')[1].split('(')[0].strip()
                production_value = float(
                    text.split('(')[1].split()[0].replace('million', '').replace('billion', '').strip())
                categories.append(category)
                production.append(production_value)

        plt.figure(figsize=(10, 8))
        plt.barh(categories, production, color='skyblue')
        plt.xlabel('Production (in million tons)')
        plt.title('US Agriculture Production 2018')

        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close()
        buf.seek(0)
        image_base64 = base64.b64encode(buf.read()).decode('utf-8')

        html_content = f"<html><body><h1>US Agriculture Production 2018</h1><img src='data:image/png;base64,{image_base64}' /></body></html>"
        return html_content

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/plot", response_class=HTMLResponse)
async def agriculture_plot():
    """
        Fetches agriculture data from a Wikipedia page, processes it, and generates comparative box and violin plots.

        This function scrapes a table of agriculture data from the specified URL, converts it into a pandas DataFrame,
        then cleans and processes the data for the years 2003 and 2013. It generates a box plot and a violin plot to
        compare the distribution of agriculture values between these two years. The plots are combined side by side
        in a single figure, converted to PNG format, encoded to Base64, and returned embedded in an HTML image tag.

        Returns:
        - HTML content: An HTML string containing a Base64-encoded PNG image of the generated plots. If there's an
          error in fetching the data, processing it, or generating the plots, an HTTP exception with status code 500
          is raised along with the error detail.
    """
    global df
    url = 'https://en.wikipedia.org/wiki/Agriculture_in_the_United_States'
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    table = soup.find('table', class_='wikitable sortable')
    rows_data = []

    if table:
        for row in table.find_all('tr'):
            cells = row.find_all(['td', 'th'])
            row_data = [cell.text.strip() for cell in cells]
            rows_data.append(row_data)
        df = pd.DataFrame(rows_data[1:], columns=rows_data[0])

    df['2003'] = pd.to_numeric(df['2003'], errors='coerce')
    df['2013'] = pd.to_numeric(df['2013'], errors='coerce')

    plot_data = df[['2003', '2013']].melt()
    plot_data = plot_data.dropna()

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    sns.boxplot(x='variable', y='value', data=plot_data, width=0.5, ax=axes[0])
    axes[0].set_title('Box Plot for 2003 and 2013')
    axes[0].set_ylim(0, 100)
    sns.violinplot(x='variable', y='value', data=plot_data, ax=axes[1])
    axes[1].set_title('Violin Plot for 2003 and 2013')
    axes[1].set_ylim(-100, 500)

    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    plt.close(fig)
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()

    encoded = base64.b64encode(image_png).decode('utf-8')

    html = f'<img src="data:image/png;base64,{encoded}">'
    return html


def fetch_and_process_data():
    """
        Fetches data from the specified Wikipedia page about agriculture in the United States,
        processes it to extract crop production ranks, and returns the processed data.

        Returns:
            list: A list of tuples containing crop production ranks and crop names.
                  Example: [(1, 'Wheat'), (2, 'Corn'), ...]
    """
    url = "https://en.wikipedia.org/wiki/Agriculture_in_the_United_States"

    response = requests.get(url)
    if not response.ok:
        raise HTTPException(status_code=404, detail="Failed to fetch data")

    html_content = response.text

    soup = BeautifulSoup(html_content, "html.parser")
    list_items = soup.find_all("li")
    data = []
    for item in list_items:
        text = item.get_text()
        if "largest world producer of" in text:
            parts = text.split("largest world producer of")
            rank_part = parts[0].strip()
            crop_part = parts[1].split("(")[0].strip()
            rank_number = ''.join(filter(str.isdigit, rank_part))
            rank = int(rank_number) if rank_number.isdigit() else None
            if rank is not None:
                data.append((rank, crop_part))
    return data


def generate_plot(data):
    """
        Generates a plot based on the provided data and returns the plot as a base64-encoded image.
        Using  if "largest world producer of" in text we find rank and crop name

        Args:
            data (list): A list of tuples containing crop production ranks and crop names.

        Returns:
            str: Base64-encoded image of the generated plot.
    """
    data.sort(key=lambda x: x[0])
    ranks, crops = zip(*data)
    plt.figure(figsize=(10, len(crops) * 0.5))
    plt.scatter(ranks, crops)
    plt.title('US Crop Production Rank')
    plt.xlabel('Rank')
    plt.ylabel('Crop')
    plt.grid(True)

    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


@app.get("/rankplot", response_class=HTMLResponse)
async def rank_plot():
    """
        Endpoint to generate and display a plot of US crop production ranks.

        Returns:
            HTMLResponse: HTML content with the generated plot or an error message.
    """
    try:
        data = fetch_and_process_data()
        image_base64 = generate_plot(data)
        html_content = f"""
        <html>
            <head>
                <title>US Crop Production Rank</title>
            </head>
            <body>
                <h1>US Crop Production Rank</h1>
                <img src="data:image/png;base64,{image_base64}" alt="Crop Ranks" />
            </body>
        </html>
        """
        return HTMLResponse(content=html_content)
    except Exception as e:
        return HTMLResponse(content=f"<html><body><h1>Error</h1><p>{str(e)}</p></body></html>", status_code=500)

def scrape_table_data(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    table = soup.find('table')  # Assuming you're interested in the first table

    headers = [th.text.strip() for th in table.find_all('th')]
    rows = table.find_all('tr')

    data = []
    for row in rows:
        columns = row.find_all('td')
        if columns:
            data.append([column.text.strip() for column in columns])

    return headers, data


@app.get("/loan", response_class=HTMLResponse)
async def display_loan_data():
    url = "https://www.fsa.usda.gov/programs-and-services/farm-loan-programs/index"
    headers, data = scrape_table_data(url)

    # Generate HTML content
    html_content = "<html><body><table border='1'>"

    # Add headers
    html_content += "<tr>" + "".join([f"<th>{header}</th>" for header in headers]) + "</tr>"

    # Add row data
    for row in data:
        html_content += "<tr>" + "".join([f"<td>{cell}</td>" for cell in row]) + "</tr>"

    html_content += "</table></body></html>"

    return html_content




if __name__ == "__main__":
    uvicorn.run("FAD:app", host="127.0.0.1", port=8000, log_level="info", reload=True)
