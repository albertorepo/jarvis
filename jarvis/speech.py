import speech_recognition as sr
import os
import subprocess

# obtain audio from the microphone
r = sr.Recognizer()
while True:
    with sr.Microphone() as source:
        r.adjust_for_ambient_noise(source)
        print "Habla"
        audio = r.listen(source)


    try:
        command = r.recognize_google(audio, language="es-Es")
        print command
        if command.lower() == "abre chrome":
            CHROME = os.path.join('C:\\', 'Users', 'usuario', 'AppData', 'Local', 'Google', 'Chrome', 'Application', 'chrome.exe')
            subprocess.call([CHROME])

    except sr.UnknownValueError:
        print("Google Speech Recognition could not understand audio")
    except sr.RequestError as e:
        print("Could not request results from Google Speech Recognition service; {0}".format(e))

