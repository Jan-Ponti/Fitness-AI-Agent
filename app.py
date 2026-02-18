# import os, json, re
# from flask import Flask, request, jsonify, render_template
# from flask_cors import CORS
# from dotenv import load_dotenv
# import google.generativeai as genai

# # --- Load API key ---
# load_dotenv()
# GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
# if not GEMINI_API_KEY:
#     raise RuntimeError("GEMINI_API_KEY is missing. Put it in .env")

# genai.configure(api_key=GEMINI_API_KEY)
# # Use a fast, inexpensive model to start. You can switch to "gemini-1.5-pro".
# MODEL_NAME = "gemini-1.5-flash"
# model = genai.GenerativeModel(MODEL_NAME)

# app = Flask(__name__, static_folder="static", template_folder="templates")
# CORS(app)

# # --- Simple calorie helpers (Mifflin-St Jeor) ---
# def bmr_mifflin(weight_kg, height_cm, age, gender):
#     # gender: "male" or "female"
#     s = 5 if gender.lower().startswith("m") else -161
#     return 10*weight_kg + 6.25*height_cm - 5*age + s

# def activity_factor(level):
#     m = {
#         "sedentary": 1.2,
#         "light": 1.375,   # 1-3 days/wk
#         "moderate": 1.55, # 3-5 days/wk
#         "active": 1.725,  # 6-7 days/wk
#         "very_active": 1.9
#     }
#     return m.get(level, 1.2)

# def target_calories(bmr, level, goal):
#     tdee = bmr * activity_factor(level)
#     # basic goal adjustment
#     if goal == "lose":
#         return max(1200, round(tdee - 500))
#     if goal == "gain":
#         return round(tdee + 300)
#     return round(tdee)  # maintain

# # --- JSON extraction (in case model wraps in code fences) ---
# def extract_json(text):
#     # pull from ```json ... ``` if present
#     fenced = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
#     if fenced:
#         text = fenced.group(1).strip()
#     # else assume raw JSON
#     return json.loads(text)

# @app.route("/", methods=["GET"])
# def home():
#     return render_template("index.html")

# @app.route("/api/plan", methods=["POST"])
# def plan():
#     data = request.get_json(force=True)

#     # 1) Validate & sanitize input
#     try:
#         age = int(data.get("age"))
#         gender = data.get("gender", "male").lower()
#         weight = float(data.get("weight_kg"))
#         height = float(data.get("height_cm"))
#         activity = data.get("activity_level", "sedentary")  # sedentary/light/moderate/active/very_active
#         goal = data.get("goal", "maintain")                 # lose/gain/maintain
#         diet = data.get("diet", "vegetarian")               # veg/vegan/non-veg/keto/etc.
#         allergies = data.get("allergies", "")               # comma-separated
#         cuisine = data.get("cuisine", "any")
#         meals_per_day = int(data.get("meals_per_day", 3))
#     except Exception as e:
#         return jsonify({"error": f"Invalid input: {e}"}), 400

#     # 2) Compute a reasonable calorie target locally
#     bmr = bmr_mifflin(weight, height, age, gender)
#     kcal_target = target_calories(bmr, activity, goal)

#     # 3) Ask Gemini for a structured plan (STRICT JSON)
#     schema_hint = {
#         "type": "object",
#         "properties": {
#             "calorie_target": {"type": "number"},
#             "macro_split": {
#                 "type": "object",
#                 "properties": {
#                     "protein_g": {"type": "number"},
#                     "carbs_g": {"type": "number"},
#                     "fat_g": {"type": "number"}
#                 },
#                 "required": ["protein_g", "carbs_g", "fat_g"]
#             },
#             "meals": {
#                 "type": "array",
#                 "items": {
#                     "type": "object",
#                     "properties": {
#                         "name": {"type": "string"},
#                         "total_calories": {"type": "number"},
#                         "items": {
#                             "type": "array",
#                             "items": {
#                                 "type": "object",
#                                 "properties": {
#                                     "title": {"type": "string"},
#                                     "calories": {"type": "number"},
#                                     "ingredients": {"type": "array", "items": {"type": "string"}},
#                                     "instructions": {"type": "string"}
#                                 },
#                                 "required": ["title", "calories"]
#                             }
#                         }
#                     },
#                     "required": ["name", "total_calories", "items"]
#                 }
#             },
#             "exercises": {
#                 "type": "array",
#                 "items": {
#                     "type": "object",
#                     "properties": {
#                         "name": {"type": "string"},
#                         "duration_min": {"type": "number"},
#                         "instructions": {"type": "string"}
#                     },
#                     "required": ["name", "duration_min"]
#                 }
#             },
#             "shopping_list": {"type": "array", "items": {"type": "string"}},
#             "tips": {"type": "array", "items": {"type": "string"}}
#         },
#         "required": ["calorie_target", "meals", "exercises"]
#     }

#     prompt = f"""
# You are a certified fitness & nutrition coach. Create a one-day diet+fitness plan.

# USER PROFILE
# - Age: {age}
# - Gender: {gender}
# - Weight: {weight} kg
# - Height: {height} cm
# - Activity level: {activity}
# - Goal: {goal}
# - Diet preference: {diet}
# - Allergies/intolerances: {allergies or "none"}
# - Preferred cuisine: {cuisine}
# - Meals per day: {meals_per_day}

# CALORIE TARGET
# Use this as the plan's target calories: {kcal_target} kcal/day.

# RESPONSE FORMAT (IMPORTANT)
# Return ONLY valid JSON (no prose, no markdown). Follow this schema idea:
# {json.dumps(schema_hint)}

# Constraints:
# - Respect diet preference and allergies.
# - Provide {meals_per_day} meals (e.g., Breakfast/Lunch/Dinner/...).
# - Balanced macro split; keep total close to {kcal_target} kcal.
# - Keep recipes simple (home-cook friendly).
# """

#     try:
#         result = model.generate_content(prompt)
#         text = result.text.strip()
#         plan_json = extract_json(text)
#     except Exception as e:
#         # If parsing fails, return raw text for debugging
#         return jsonify({"error": "Model response parsing failed", "raw": text if "text" in locals() else "", "details": str(e)}), 500

#     # Ensure calorie_target is set (fallback to our local calc)
#     plan_json.setdefault("calorie_target", kcal_target)
#     return jsonify(plan_json), 200

# if __name__ == "__main__":
#     # Run local dev server
#     app.run(host="127.0.0.1", port=5000, debug=True)


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
