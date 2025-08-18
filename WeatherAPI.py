import requests

def get_weather(city: str) -> dict:
    """Fetch weather from Open-Meteo (free, no key needed)."""
    try:
        # Step 1: Geocode city → lat/lon
        geo_resp = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": city, "count": 1, "language": "en", "format": "json"},
            timeout=10,
        )
        geo_resp.raise_for_status()
        geo_data = geo_resp.json()

        if not geo_data.get("results"):
            return {"status": "error", "error_message": f"City '{city}' not found."}

        lat = geo_data["results"][0]["latitude"]
        lon = geo_data["results"][0]["longitude"]
        resolved_name = geo_data["results"][0]["name"]
        country = geo_data["results"][0].get("country", "")

        # Step 2: Fetch current weather
        weather_resp = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": lat,
                "longitude": lon,
                "current_weather": "true",
            },
            timeout=10,
        )
        weather_resp.raise_for_status()
        weather_data = weather_resp.json()

        current = weather_data.get("current_weather", {})
        if not current:
            return {"status": "error", "error_message": "No current weather available."}

        temp = current.get("temperature")
        wind = current.get("windspeed")

        report = (
            f"The weather in {resolved_name}, {country} is "
            f"{temp}°C with wind speed {wind} km/h."
        )
        return {"status": "success", "report": report}

    except Exception as e:
        return {"status": "error", "error_message": str(e)}
