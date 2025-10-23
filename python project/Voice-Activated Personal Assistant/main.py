import warnings
warnings.filterwarnings("ignore")  # Hide unnecessary warnings

import speech_recognition as sr
import pyttsx4 as pyttsx3
import datetime
import pywhatkit
import wikipedia
import pyjokes
import time

# Initialize the text-to-speech engine
engine = pyttsx3.init()

def talk(text):
    """Speak the text and print it in terminal"""
    print("Assistant:", text)
    engine.say(text)
    engine.runAndWait()

def take_command():
    """Listen to the user and return the recognized command"""
    listener = sr.Recognizer()
    try:
        with sr.Microphone() as source:
            print("Listening...")
            voice = listener.listen(source)
            command = listener.recognize_google(voice)
            command = command.lower()
            if 'assistant' in command:
                command = command.replace('assistant', '')
            print("You said:", command)
            return command
    except:
        talk("Sorry, I could not hear you properly.")
        return ""

def run_assistant():
    """Process commands and execute actions"""
    command = take_command()
    if not command:
        return

    # Time
    if 'time' in command:
        time_now = datetime.datetime.now().strftime('%I:%M %p')
        talk('Current time is ' + time_now)

    # Date
    elif 'date' in command or 'today' in command:
        today = datetime.datetime.now().strftime('%A, %B %d, %Y')
        talk('Today is ' + today)

    # Wikipedia search
    elif 'who' in command or 'wikipedia' in command or 'tell me about' in command:
        try:
            query = command.replace('who is', '').replace('tell me about', '').replace('wikipedia', '').strip()
            info = wikipedia.summary(query, 1)
            talk(info)
        except:
            talk("Sorry, I could not find information on " + query)

    # Play on YouTube
    elif 'play' in command or 'youtube' in command:
        try:
            song = command.replace('play', '').replace('on youtube', '').strip()
            talk('Playing ' + song)
            pywhatkit.playonyt(song)
        except:
            talk("Sorry, I could not play " + song)

    # Tell a joke
    elif 'joke' in command:
        talk(pyjokes.get_joke())

    else:
        talk('Please say the command again.')

# --- Continuous listening loop ---
talk("Hello, I am your voice assistant. How can I help you?")

while True:
    run_assistant()
    print("Waiting for next command...")
    time.sleep(1)  # Small delay to prevent overlapping
