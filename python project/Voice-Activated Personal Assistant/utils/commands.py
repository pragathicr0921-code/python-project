import wikipedia
import pyjokes
import pywhatkit
import requests

def search_wikipedia(query):
    try:
        summary = wikipedia.summary(query, sentences=2)
        return summary
    except wikipedia.exceptions.DisambiguationError as e:
        return f"Multiple results found for {query}. Please be more specific."
    except wikipedia.exceptions.PageError:
        return f"Sorry, I could not find any page for {query}."
    except Exception as e:
        return f"An error occurred: {str(e)}"

def tell_joke():
    return pyjokes.get_joke()

def play_youtube(video):
    try:
        pywhatkit.playonyt(video)
        return f"Playing {video} on YouTube."
    except Exception as e:
        return f"Could not play video: {str(e)}"

def get_weather(city):
    try:
        api_key = "YOUR_OPENWEATHERMAP_API_KEY"  
        url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
        response = requests.get(url).json()
        if response.get("cod") != 200:
            return f"Could not get weather for {city}."
        weather = response["weather"][0]["description"]
        temp = response["main"]["temp"]
        return f"The weather in {city} is {weather} with a temperature of {temp}°C."
    except Exception as e:
        return f"Error fetching weather: {str(e)}"