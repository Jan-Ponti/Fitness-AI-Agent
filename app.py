# import os, json, re
# from flask import Flask, request, jsonify, render_template
# from flask_cors import CORS
# from dotenv import load_dotenv
# import google.generativeai as genai

import os
from typing import List, Dict, Any
from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request
from flask_cors import CORS
import google.generativeai as genai

# ---------- Setup ----------
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise RuntimeError("GEMINI_API_KEY is missing. Put it in .env as GEMINI_API_KEY=...")

genai.configure(api_key=API_KEY)

# Choose a fast, capable model. You can switch to "gemini-1.5-pro" later.
MODEL_NAME = "gemini-2.0-flash"
model = genai.GenerativeModel(MODEL_NAME)

app = Flask(__name__, static_folder="static", template_folder="templates")
CORS(app)


# ---------- Helpers ----------
def build_system_preamble(profile: Dict[str, Any]) -> str:
    """Build a brief profile context block for personalization."""
    if not profile:
        return (
            "You are a friendly, factual fitness & diet assistant. "
            "Answer clearly and helpfully for general questions."
        )

    # Normalize expected keys; tolerate missing fields
    age = profile.get("age", "")
    gender = profile.get("gender", "")
    height = profile.get("height", "")
    weight = profile.get("weight", "")
    goal = profile.get("goal", "")
    diet = profile.get("diet", "")
    activity = profile.get("activity", "")
    allergies = profile.get("allergies", "")

    return (
        "You are a friendly, factual fitness & diet assistant. "
        "Personalize suggestions using this user profile when relevant.\n\n"
        f"User Profile:\n"
        f"- Age: {age}\n"
        f"- Gender: {gender}\n"
        f"- Height: {height} cm\n"
        f"- Weight: {weight} kg\n"
        f"- Goal: {goal}\n"
        f"- Diet Preference: {diet}\n"
        f"- Activity Level: {activity}\n"
        f"- Allergies: {allergies or 'none'}\n"
    )


def build_prompt(history: List[Dict[str, str]], user_message: str, system_preamble: str, intent_hint: str = "") -> str:
    """
    Convert chat history + current message into a single prompt for Gemini.
    We keep it simple & stateless (history comes from the client).
    """
    lines = [system_preamble, "\nConversation so far:"]
    for turn in history[-12:]:  # keep last ~12 turns to limit prompt length
        role = turn.get("role", "user")
        content = turn.get("content", "")
        lines.append(f"{role.capitalize()}: {content}")

    lines.append(f"User: {user_message}")

    if intent_hint == "variation":
        lines.append(
            "\nInstruction: Provide a different variation from the last plan or suggestion. "
            "Keep it consistent with the user's profile & preferences."
        )

    # Helpful style instruction
    lines.append(
        "\nAssistant: Respond concisely. Use bullet points for plans/steps. "
        "If giving a day plan, include approximate calories/macros when helpful."
    )

    return "\n".join(lines)


# ---------- Routes ----------
@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")


@app.route("/api/chat", methods=["POST"])
def api_chat():
    """
    Expected JSON body:
    {
      "message": "string",
      "history": [{"role":"user"|"assistant", "content":"..."}],
      "profile": {...},
      "intentHint": "variation" | ""    # optional
    }
    """
    data = request.get_json(force=True) or {}
    message: str = data.get("message", "").strip()
    history: List[Dict[str, str]] = data.get("history", [])
    profile: Dict[str, Any] = data.get("profile", {})
    intent_hint: str = data.get("intentHint", "")

    if not message:
        return jsonify({"reply": "Please type a message.", "error": None})

    system_preamble = build_system_preamble(profile)
    prompt = build_prompt(history, message, system_preamble, intent_hint)

    try:
        resp = model.generate_content(prompt)
        text = (resp.text or "").strip() if resp else ""
        if not text:
            text = "Sorry, I couldn’t generate a response."
        return jsonify({"reply": text})
    except Exception as e:
        print(f"Model error: {e}")  # Print error to console for debugging
        return jsonify({"reply": "Something went wrong calling the model.", "error": str(e)}), 500


if __name__ == "__main__":
    # Flask dev server
    app.run(host="127.0.0.1", port=5000, debug=True)
