from gtts import gTTS
import os

def speak_result(result, languages=['hi','en']):
    """
    result: dict with keys 'animal', 'breed',' disease', 'confidence'
    languages: list of language codes in order, e.g., ['hi', 'en']
    """

    for lang in languages:
        if lang == 'hi':
            text = f"पशु का प्रकार {result['animal']} है, नस्ल {result['breed']}। रोग स्थिति: {result['disease']}।"
        else:
            text = f"The animal is a {result['animal']}, breed {result['breed']}. Disease status: {result['disease']}."

        print(f"Text ({lang}): {text}")

        tts = gTTS(text=text, lang=lang)
        filename = f"result_{lang}.mp3"
        tts.save(filename)
        print(f"Audio saved as {filename}")


        if os.name == 'nt':  # For Windows
            os.system(f"start {filename}")
        elif os.name == 'posix':  # For macOS and Linux
            os.system(f"open {filename}")

if __name__ == "__main__":

    result_hi = {
    "animal": "गाय",
    "breed": "गिर",
    "disease": "कोई रोग नहीं"
    }

# For English audio
    result_en = {
        "animal": "Cow",
        "breed": "Gir",
        "disease": "None"
    }

    speak_result(result_hi, languages=['hi'])
    #speak_result(result_en, languages=['en'])
