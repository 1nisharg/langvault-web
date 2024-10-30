from flask import Flask, render_template, request, send_file, Response
import pytesseract
import io
import base64
import PyPDF2
from textblob import TextBlob
from langchain_groq import ChatGroq
from gtts import gTTS
from dotenv import load_dotenv
import os

app = Flask(__name__)
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

def text_speech(text):
    tts = gTTS(text=text, lang='hi')
    speech_bytes = io.BytesIO()
    tts.write_to_fp(speech_bytes)
    speech_bytes.seek(0)
    return speech_bytes.read()

def convert_pdf_to_images(pdf_bytes):
    pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))
    for page in pdf_reader.pages:
        text = page.extract_text()
        yield text

def create_download_link(text):
    text_bytes = text.encode()
    b64 = base64.b64encode(text_bytes).decode()
    return b64

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', error="No file uploaded")
        
        file = request.files['file']
        lang = request.form.get('language')
        
        if file.filename == '':
            return render_template('index.html', error="No file selected")
        
        if not lang:
            return render_template('index.html', error="Please specify a language")

        llm = ChatGroq(
            temperature=0,
            model="llama3-70b-8192",
            api_key=GROQ_API_KEY
        )

        translated_text = ""
        page_count = 0
        
        try:
            for text in convert_pdf_to_images(file.read()):
                page_count += 1
                result = llm.predict(
                    f"Translate this text separated by triple backticks delimiter(```) \n Text: \n ```\n {text} \n ``` \n in {lang} without changing its meaning"
                )
                clean_result = result.replace("```", " ")
                translated_text += f"\n----- Page {page_count} -----\n{clean_result}\n"
                
            audio_data = text_speech(translated_text)
            audio_b64 = base64.b64encode(audio_data).decode()
            download_b64 = create_download_link(translated_text)
            
            return render_template('index.html', 
                                 translated_text=translated_text,
                                 audio_data=audio_b64,
                                 download_data=download_b64,
                                 page_count=page_count)
        
        except Exception as e:
            return render_template('index.html', error=f"An error occurred: {str(e)}")

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)