import speech_recognition as sr
import pyttsx3

# Create a recognizer instance
r = sr.Recognizer()

# Use the microphone as the audio source
with sr.Microphone() as source:
    print("Listening...")

    # Adjust for ambient noise
    r.adjust_for_ambient_noise(source)

    # Record the audio
    audio = r.listen(source)

    print("Recognition complete.")

    try:
        # Recognize speech using Google Web Speech API
        text = r.recognize_google(audio)
        print("Recognized text: " + text)

        # Initialize the text-to-speech engine
        engine = pyttsx3.init()

        # Set the speech rate (optional)
        engine.setProperty("rate", 150)

        # Generate the spoken response
        engine.say("You said: " + text)
        engine.runAndWait()

    except sr.UnknownValueError:
        print("Unable to recognize speech")
    except sr.RequestError as e:
        print("Error: {0}".format(e))
