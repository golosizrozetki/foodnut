import os
import sys
import random
import pandas as pd
from flask import Flask, request, jsonify, render_template

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(os.path.dirname(CURRENT_DIR), "src")
sys.path.insert(0, SRC_DIR)

from recipes import NutritionFacts, SimilarRecipes, RatingPredictor

app = Flask(__name__, template_folder=os.path.join(CURRENT_DIR, "templates"))

# Pre-load ML model and static databases
print("Initializing ML models and nutritional databases...")
predictor = RatingPredictor()
nutrition_facts = NutritionFacts()
similar_recipes = SimilarRecipes()

# Fast in-memory MenuGenerator for instant web response
MENU_CSV = os.path.join(SRC_DIR, "menu_recipes.csv")
menu_df = pd.read_csv(MENU_CSV).fillna(0.0)
menu_records = menu_df.to_dict("records")
meals_by_type = {
    "breakfast": [r for r in menu_records if r.get("meal_type") == "breakfast"],
    "lunch": [r for r in menu_records if r.get("meal_type") == "lunch"],
    "dinner": [r for r in menu_records if r.get("meal_type") == "dinner"],
}

ALL_KNOWN_INGREDIENTS = sorted(list(set(predictor.features) | set(nutrition_facts.df.index)))

VERDICT_MESSAGES = {
    "great": "Great choice! These ingredients combine harmoniously and are predicted to produce a delicious, highly-rated dish.",
    "so-so": "A decent combination, but nothing extraordinary. The predicted flavor profile is moderate or average.",
    "bad": "Questionable pairing. Based on culinary data and our trained ML model, this ingredient combination is likely to disappoint."
}


def fast_generate_menu(n_samples=5000, top_k=20):
    nutrients = ["calories_pct", "fat_pct", "protein_pct", "sodium_pct"]
    valid = []
    b_list = meals_by_type["breakfast"]
    l_list = meals_by_type["lunch"]
    d_list = meals_by_type["dinner"]

    for _ in range(n_samples):
        b = random.choice(b_list)
        l = random.choice(l_list)
        d = random.choice(d_list)

        ok = True
        totals = {}
        for n in nutrients:
            tot = b.get(n, 0.0) + l.get(n, 0.0) + d.get(n, 0.0)
            totals[n] = round(tot, 1)
            if not (50.0 <= tot <= 100.0):
                ok = False
                break
        if ok:
            total_rating = round(b.get("rating", 0.0) + l.get("rating", 0.0) + d.get("rating", 0.0), 3)
            valid.append((total_rating, b, l, d, totals))

    if not valid:
        return None

    valid.sort(key=lambda x: x[0], reverse=True)
    picked = random.choice(valid[:top_k])
    score, b, l, d, totals = picked

    def format_meal(m):
        raw_ings = str(m.get("ingredients", "")).split("|")
        return {
            "title": m.get("title", "Untitled Recipe"),
            "rating": round(float(m.get("rating", 0.0)), 2),
            "url": m.get("url", ""),
            "ingredients": [i.strip() for i in raw_ings if i.strip()],
            "nutrients": {
                "calories": round(float(m.get("calories_pct", 0.0))),
                "protein": round(float(m.get("protein_pct", 0.0))),
                "fat": round(float(m.get("fat_pct", 0.0))),
                "sodium": round(float(m.get("sodium_pct", 0.0))),
            }
        }

    return {
        "score": score,
        "breakfast": format_meal(b),
        "lunch": format_meal(l),
        "dinner": format_meal(d),
        "totals": {
            "calories": totals.get("calories_pct", 0.0),
            "protein": totals.get("protein_pct", 0.0),
            "fat": totals.get("fat_pct", 0.0),
            "sodium": totals.get("sodium_pct", 0.0),
        }
    }


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/ingredients", methods=["GET"])
def get_ingredients():
    q = request.args.get("q", "").lower().strip()
    if not q:
        return jsonify({"ingredients": ALL_KNOWN_INGREDIENTS[:60]})
    matches = [ing for ing in ALL_KNOWN_INGREDIENTS if q in ing]
    return jsonify({"ingredients": matches[:40]})


@app.route("/api/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True) or {}
    raw_ingredients = data.get("ingredients", [])
    if isinstance(raw_ingredients, str):
        ingredients = [i.strip() for i in raw_ingredients.split(",") if i.strip()]
    elif isinstance(raw_ingredients, list):
        ingredients = [str(i).strip() for i in raw_ingredients if str(i).strip()]
    else:
        ingredients = []

    if not ingredients:
        return jsonify({"error": "Ingredients list is empty."}), 400

    # 1. Rating Prediction + Probabilities
    ing_set = {i.lower() for i in ingredients}
    x_df = pd.DataFrame(
        [[1 if f in ing_set else 0 for f in predictor.features]],
        columns=predictor.features,
    )
    predicted_class = str(predictor.model.predict(x_df)[0])
    
    probabilities = {}
    if hasattr(predictor.model, "predict_proba"):
        probs = predictor.model.predict_proba(x_df)[0]
        for cls_name, prob_val in zip(predictor.model.classes_, probs):
            probabilities[str(cls_name)] = round(float(prob_val) * 100, 1)

    # 2. Nutrition Facts
    missing = nutrition_facts.missing(ingredients)
    facts = nutrition_facts.get_facts(ingredients)

    agg_nutrients = {}
    cleaned_facts = {}
    for ing, n_dict in facts.items():
        cleaned_facts[ing] = {}
        for nut_name, val in n_dict.items():
            try:
                num = round(float(val), 1)
                cleaned_facts[ing][nut_name] = num
                agg_nutrients[nut_name] = round(agg_nutrients.get(nut_name, 0.0) + num, 1)
            except (ValueError, TypeError):
                continue

    # 3. Similar Recipes (adaptive similarity threshold)
    matches = similar_recipes.find_similar(ingredients, top_n=3, min_score=0.20)
    if not matches:
        matches = similar_recipes.find_similar(ingredients, top_n=3, min_score=0.01)

    formatted_matches = []
    for r in matches:
        score_val = r.get("score", 0.0)
        formatted_matches.append({
            "title": r.get("title", "Unknown"),
            "rating": round(float(r.get("rating", 0.0)), 2),
            "url": r.get("url", ""),
            "match_pct": round(float(score_val) * 100, 1)
        })

    return jsonify({
        "verdict": predicted_class,
        "message": VERDICT_MESSAGES.get(predicted_class, f"Predicted class: {predicted_class}"),
        "probabilities": probabilities,
        "ingredients": ingredients,
        "missing_ingredients": missing,
        "nutrition_by_ingredient": cleaned_facts,
        "aggregate_nutrition": agg_nutrients,
        "similar_recipes": formatted_matches
    })


@app.route("/api/menu", methods=["POST", "GET"])
def generate_menu_endpoint():
    menu = fast_generate_menu()
    if not menu:
        return jsonify({"error": "Unable to assemble a menu meeting strict 50-100% DV limits. Please try again."}), 422
    return jsonify(menu)


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "features_count": len(predictor.features)})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    print(f"Starting NutriTaste ML Service on http://0.0.0.0:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)
