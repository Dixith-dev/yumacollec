import os
import logging
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from dotenv import load_dotenv
from salesgpt.agents import SalesGPT
from langchain_community.chat_models import ChatLiteLLM
import sys
import io

#__import__('pysqlite3')
#import sys

#sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# Load environment variables
load_dotenv()

# Suppress logging output
logging.getLogger().setLevel(logging.ERROR)

# Set your OpenAI API key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize the SalesGPT agent
llm = ChatLiteLLM(temperature=0.4, model='gpt-3.5-turbo-16k')

instruction = """

As the YUUMA AI, developed by Tab Robotics, your main responsibility is to provide users with assistance regarding the products offered by YUUMA. When addressing users, please use the pronoun 'we' to refer to the company. Your integration on their website aims to ensure prompt and relevant support in a professional manner, enabling users to navigate through available options efficiently. As a user, my questions will solely pertain to YUUMA, and not about the AI itself. Therefore, please only provide information based on the data provided. For instance, if I ask 'What products do you provide?', your response should be concise and informative, stating the specific services offered by YUUMA. Additionally, when the user greets you at the start of the conversation, please respond with a good response according to the question and provide the info user asks and please continue the conversation like a human. Please look at what you are saying and make sure it makes sense if the user asks a question instead of greeting please answer the question and please continue the conversation like a human. Please try to answer in very short format but also provide the most important information in short. Please maintain some space between the sentences. Try to be happy and positive. When user asks to recommend a product also give him the image link with the product link and price, this is very very very important. If you cant find the link in the dataset please do not make it up because the link will be processed as an image to show to the user. Do not recommend or display any links that are not in the dataset.

Please provide the image links too that is the most important feature

"""

from data import kb

sales_agent = SalesGPT.from_llm(
    llm,
    use_tools=False,
    verbose=False,
    salesperson_name="YUUMA AI",
    salesperson_role=instruction + kb,
    company_name="YUMMA",
    company_business="Ecommerce")

# Initialize Flask app
app = Flask(__name__)
CORS(app)


# Generate response
def generate_response(user_input):
  # Redirect stdout to capture the response
  captured_output = io.StringIO()
  sys.stdout = captured_output

  # Process the user input and generate a response
  sales_agent.human_step(user_input)
  sales_agent.step()

  # Restore stdout and get the captured output
  sys.stdout = sys.__stdout__
  full_response = captured_output.getvalue()

  # Extract the response part after "Heyford AI:"
  response = full_response.split("YUUMA AI:", 1)[-1].strip()
  import re
  response = re.sub(r'\[Image Link\]\((.*?)\)', r'\1', response)
  return response


# Flask routes
@app.route('/')
def home():
  return render_template('bannner.html')


@app.route('/get-response', methods=['POST'])
def get_response():
  user_input = request.json['message'].lower().strip()
  user_id = request.json.get(
      'user_id')  # Get user/session identifier from the request
  if not user_id:
    return jsonify({'response': "Error: User ID is missing or invalid."})

  response = generate_response(user_input)  # Generate response from the AI
  return jsonify({'response': response})


if __name__ == '__main__':
  app.run(host='0.0.0.0', port=5000)
