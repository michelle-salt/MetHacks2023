import cohere
import openai
import json
co = cohere.Client('') # insert your api key here

from cohere.responses.classify import Example
def cohereAnalysis():


    examples = [
        Example("The order came 5 days early", "happy"),
        Example("The item exceeded my expectations", "happy"),
        Example("I ordered more for my friends", "happy"),
        Example("I would buy this again", "happy"),
        Example("I would recommend this to others", "happy"),
        Example("The package was damaged", "sad"),
        Example("The order is 5 days late", "sad"),
        Example("The order was incorrect", "sad"),
        Example("I want to return my item", "sad"),
        Example("The item\'s material feels low quality", "sad"),
        Example("The product was okay", "neutral"),
        Example("I received five items in total", "neutral"),
        Example("I bought it from the website", "neutral"),
        Example("I used the product this morning", "neutral"),
        Example("The product arrived yesterday", "neutral"),
        Example("I'm so proud of you", "happy"),
        Example("What a great time to be alive", "happy"),
        Example("That's awesome work", "happy"),
        Example("The service was amazing", "happy"),
        Example("I love my family", "happy"),
        Example("They don't care about me", "sad"),
        Example("I hate this place", "sad"),
        Example("The most ridiculous thing I've ever heard", "sad"),
        Example("I am really frustrated", "sad"),
        Example("This is so unfair", "sad"),
        Example("This made me think", "neutral"),
        Example("The good old days", "neutral"),
        Example("What's the difference", "neutral"),
        Example("You can't ignore this", "neutral"),
        Example("That's how I see it", "neutral"),
        Example("My dog died yesterday", "sad"),
        Example("I am sad because my dog died", "sad"),
        Example("I ate ice cream today", "happy"),
        Example("I went to the park for a walk today", "happy"),
        Example("I went to my favourite restaurant", "happy")
    ]

    inputs = ["Hello, world! What a beautiful day",
              "It was a great time with great people",
              "Great place to work",
              "That was a wonderful evening",
              "Maybe this is why",
              "Let's start again",
              "That's how I see it",
              "These are all facts",
              "This is the worst thing",
              "I cannot stand this any longer",
              "This is really annoying",
              "I am just plain fed up"
              ]

    response = co.classify(
        model='large',
        inputs=voice_transcription(),
        examples=examples,
    )

    print(response.classifications)
    string = str(response.classifications)
    string_array = string.split(",")

    happyCount = 0
    sadCount = 0
    neutralCount = 0

    for string in string_array:
         happyCount += string.count("happy")

    for string in string_array:
         sadCount += string.count("sad")

    for string in string_array:
         neutralCount += string.count("neutral")

    totalCount = happyCount + sadCount + neutralCount
    print()
    print()
    print("Voice Analysis Stats:")
    print()
    print("Happy Prompts: " + str(happyCount))
    print("Proportion: " + str(int(((happyCount/totalCount)*100))) + " %" )
    print()
    print("Sad Prompts: " + str(sadCount))
    print("Proportion: " + str(int(((sadCount/totalCount)*100))) + " %" )
    print()
    print("Neutral Prompts: " + str(neutralCount))
    print("Proportion: " + str(int(((neutralCount/totalCount)*100))) + " %" )

def voice_transcription():
    API_KEY = '' # insert your api key here
    model_id = 'whisper-1'

    media_file_path = 'president-obamas-greatest-speech.mp3'

    media_file = open(media_file_path, 'rb')

    response = openai.Audio.transcribe(

        api_key=API_KEY,
        model=model_id,
        file=media_file,
        response_format='json'  # text. json, srt, vtt
    )

    # Extract the text field from the JSON response
    text = response['text']

    # Split the string into individual sentences
    sentences = text.split('. ')

    # Add a period to the end of each sentence
    for i in range(len(sentences)):
        sentences[i] += '.'

    # Print the resulting string array
    print(sentences)

    return sentences

cohereAnalysis()
# voice_transcription()