from flask import Flask, render_template, request
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import re

app = Flask(__name__)

# Load DialoGPT
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

chat_history_ids = None

@app.route("/")
def index():
    return render_template("chat.html")

@app.route("/get", methods=["POST"])
def chat():
    user_input = request.form["msg"]
    return hospital_bot_response(user_input)

# ---------------- RULE + AI LOGIC ----------------

def hospital_bot_response(text):
    text_lower = text.lower()

    # 🚨 Emergency rule
    if any(word in text_lower for word in ["emergency", "accident", "severe pain", "bleeding"]):
        return "🚨 This looks like an emergency. Please visit the nearest emergency room immediately."

    # 🏥 Appointment rule
    if any(word in text_lower for word in ["appointment", "book", "schedule"]):
        return "Sure! Please tell me which department you want to book an appointment for (Dental, Cardiology, Orthopedic)."

    # 🦷 Department rule
    if any(dept in text_lower for dept in ["dental", "cardiology", "orthopedic", "neurology"]):
        dept = re.findall(r"dental|cardiology|orthopedic|neurology", text_lower)[0]
        return f"Great! {dept.capitalize()} department is available. Would you like a morning or evening slot?"

    # ⏰ Timing rule
    if "timing" in text_lower or "open" in text_lower:
        return "Our hospital operates from 9 AM to 8 PM, Monday to Saturday."

    # 📍 Location rule
    if "location" in text_lower or "address" in text_lower:
        return "We are located at City Center Road, near Central Mall."

    # 🤖 AI fallback (DialoGPT)
    return dialoGPT_response(text)

# ---------------- DialoGPT Response ----------------

def dialoGPT_response(text):
    global chat_history_ids

    new_input_ids = tokenizer.encode(text + tokenizer.eos_token, return_tensors='pt')

    bot_input_ids = (
        torch.cat([chat_history_ids, new_input_ids], dim=-1)
        if chat_history_ids is not None
        else new_input_ids
    )

    chat_history_ids = model.generate(
        bot_input_ids,
        max_length=1000,
        pad_token_id=tokenizer.eos_token_id
    )

    reply = tokenizer.decode(
        chat_history_ids[:, bot_input_ids.shape[-1]:][0],
        skip_special_tokens=True
    )

    return reply


if __name__ == "__main__":
    app.run(debug=True)
