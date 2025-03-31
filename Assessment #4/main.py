from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import random
from skeletonCodeAssessment4 import patients
from langchain_groq import ChatGroq
import os
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import json

# the api Key will not work once the submission is made, to make the code work, use your own api key
os.environ['GROQ_API_KEY']='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'
# as the confirmation reply may not be in the format as we expect it to be, thus using an llm to get confirmation results is much more viable and efficient
llm = ChatGroq(model='llama-3.3-70b-versatile', temperature=0.9, api_key=os.environ['GROQ_API_KEY'])
template='''You are analyzing a patient's response regarding their appointment confirmation. 

Patient's response: "{user_answer}"

Determine if the patient has confirmed their appointment. 
- If the response clearly indicates confirmation (e.g., "Yes", "Confirmed", "I will come"), return **only** the number `1`.
- If the response indicates cancellation, uncertainty, or no confirmation (e.g., "No", "Not sure", "Maybe", "Reschedule"), return **only** the number `0`.
- Return only a single integer (`1` or `0`) without any extra text.'''
app = FastAPI()
prompt = PromptTemplate(template=template, input_variables=['user_answer'])

# Sample patient database
patients = patients

def translate_message(message: str, language: str) -> str:
    translations = {
        "Tamil": "உங்கள் நேரம் உறுதிசெய்யப்பட்டது. தயவுசெய்து வருக!",
        "Telugu": "మీ నియామకం నిర్ధారించబడింది. దయచేసి రండి!",
        "Malayalam": "നിങ്ങളുടെ അപോയിന്റ്മെന്റ് സ്ഥിരീകരിച്ചിരിക്കുന്നു. ദയവായി വരൂ!",
        "Hindi": "आपका अपॉइंटमेंट कन्फर्म हो गया है। कृपया आएं!",
        "English": "Your appointment is confirmed. Please visit!"
    }
    return translations.get(language, message)  # Use AI in real implementation

class MessageRequest(BaseModel):
    patient_id: int

responses = []  # Simulated confirmation tracking

@app.post("/send-message")
def send_message(request: MessageRequest):
    patient = next((p for p in patients if p["id"] == request.patient_id), None)
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    
    message = translate_message("Your appointment is confirmed. Please visit!", patient["language"])
    channel = patient["channel"]
    status = f"Sent via {channel} to {patient['name']} ({patient['language']}): {message}"
    return {"status": status}

@app.post("/log-response")
def log_response(patient_id: int, response: str):
    responses.append({"patient_id": patient_id, "response": response})
    return {"message": "Response logged successfully"}

@app.get("/analytics")
def get_analytics():
    total_responses = len(responses)
    confirmed_responses = 0

    for response in responses:
        chain = LLMChain(llm=llm, prompt=prompt)
        result = chain.run(user_answer=response["response"])
        try:
            confirmation_number = int(result.strip())  # Directly parse the result as an integer
            if confirmation_number == 1:
                confirmed_responses += 1
        except (ValueError, TypeError):
            continue

    confirmation_rate = (confirmed_responses / total_responses) * 100 if total_responses > 0 else 0
    return {"total_responses": total_responses, "confirmation_rate":f"{confirmation_rate:.2f}%"}

@app.get("/patients")
def get_patients():
    return patients
