import os.path
from logging import exception

import speech_recognition as sr
import win32com.client
import webbrowser
import openai
from openai import OpenAI
import random

chatStr = ""
def chat(query):

    global chatStr
    print(chatStr)
    client = OpenAI(
        api_key="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"
    )
    chatStr += f"darshil : {query}\n Rudra:"
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        store=True,

        messages=[
            {"role": "user", "content": chatStr}
        ]
    )

    response_content = completion.choices[0].message.content
    say(response_content)
    chatStr += f"{response_content}\n"
    return response_content



    


def ai(prompt):


    text = f"Open ai response :{prompt} \n ***********************\n\n"
    client = OpenAI(
        api_key="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"
    )

    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        store=True,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    response_content = completion.choices[0].message.content
    print(response_content)
    text += response_content
    if not os.path.exists("Openai"):
        os.mkdir("Openai")

    with open(f"Openai/{prompt[0:30]}.txt", "w") as f:
        f.write(text)


def say(text):
    speaker = win32com.client.Dispatch("SAPI.SpVoice")
    speaker.Speak(text)

def takeCommand():
    r = sr.Recognizer()
    with sr.Microphone() as source:
       # r.pause_threshold = 1
        audio = r.listen(source)
        try:
            query = r.recognize_google(audio, language="en-IN")
            print(f"user said: {query}")
            return query
        except Exception as e:
            return "some error occured sorry"


say("darshil")
while True:
    print("listning...")
    query = takeCommand()
    sites = [["youtube", "youtube.com"],["wikipedia", "wikipedia.com"],["google", "google.com"]]
    for site in sites:
        if f"open {site[0]}".lower() in query.lower():
            webbrowser.open(site[1])
            say(f"openning {site[0]} sir")


    if "using artificial intelligence".lower() in query.lower():
        ai(prompt=query)

    else:
        chat(query)
