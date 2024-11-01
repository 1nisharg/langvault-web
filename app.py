from flask import Flask, render_template, request
import io
import base64
import PyPDF2
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

app = Flask(__name__)
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

def convert_pdf_to_text(pdf_bytes):
    # Yield each page's text one by one to manage memory
    pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))
    for page in pdf_reader.pages:
        text = page.extract_text()
        if text:
            yield text

def create_download_link(text):
    # Convert text to base64 to create a downloadable link
    text_bytes = text.encode()
    b64 = base64.b64encode(text_bytes).decode()
    return b64

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files.get('file')
        lang = request.form.get('language')

        if not file or file.filename == '' or not lang:
            return render_template('index.html', error="Please upload a file and specify a language")

        # Initialize LLM only when needed to save memory
        llm = ChatGroq(
            temperature=0,
            model="gemma2-9b-it",
            api_key=GROQ_API_KEY
        )

        translated_text = ""
        page_count = 0

        try:
            # Process each page text in chunks to reduce memory usage
            for text in convert_pdf_to_text(file.read()):
                page_count += 1
                result = llm.predict(
                    f"Translate this text separated by triple backticks delimiter(```) \n Text: \n ```\n {text} \n ``` \n in {lang} without changing its meaning"
                )
                clean_result = result.replace("```", " ")
                translated_text += f"\n----- Page {page_count} -----\n{clean_result}\n"

            download_b64 = create_download_link(translated_text)

            return render_template('index.html',
                                   translated_text=translated_text,
                                   download_data=download_b64,
                                   page_count=page_count)

        except Exception as e:
            return render_template('index.html', error=f"An error occurred: {str(e)}")

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
