import os
import random
import pandas as pd
import joblib
import gradio as gr

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def get_path(filename):
    local_path = os.path.join(BASE_DIR, filename)
    if os.path.exists(local_path):
        return local_path
    parent_path = os.path.join(os.path.dirname(BASE_DIR), "src", filename)
    if os.path.exists(parent_path):
        return parent_path
    return local_path

# Load Model
model_payload = joblib.load(get_path("recipe_model.joblib"))
clf_model = model_payload["model"]
features = model_payload["features"]

# Load Nutrition Facts
nutrition_df = pd.read_csv(get_path("nutrition_facts.csv"), index_col=0).fillna(0.0)
nutrition_df.columns = [str(c).replace(" (%DV)", "").strip() for c in nutrition_df.columns]

# Load Similar Recipes
similar_df = pd.read_csv(get_path("similar_recipes.csv"))
similar_df["norm_set"] = similar_df["norm_set"].apply(
    lambda s: set(s.split("|")) if isinstance(s, str) and s else set()
)

# Load Menu Recipes for Fast In-Memory Permutations
menu_df = pd.read_csv(get_path("menu_recipes.csv")).fillna(0.0)
menu_records = menu_df.to_dict("records")
meals_by_type = {
    "breakfast": [r for r in menu_records if r.get("meal_type") == "breakfast"],
    "lunch": [r for r in menu_records if r.get("meal_type") == "lunch"],
    "dinner": [r for r in menu_records if r.get("meal_type") == "dinner"],
}

ALL_KNOWN_INGREDIENTS = sorted(list(set(features) | set(nutrition_df.index)))

VERDICTS = {
    "great": ("Great Pairing ⭐⭐⭐", "#15803d", "Great choice! These ingredients combine harmoniously and are predicted to produce a delicious, highly-rated dish."),
    "so-so": ("So-so / Mediocre ⭐⭐", "#b45309", "A decent combination, but nothing extraordinary. The predicted flavor profile is moderate or average."),
    "bad":   ("Bad Combination ⚠️", "#b91c1c", "Questionable pairing. Based on culinary data and our trained ML model, this ingredient combination is likely to disappoint.")
}

def jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)

def evaluate_ingredients(selected_items):
    if not selected_items:
        return (
            "<div style='padding:20px; text-align:center; color:#666;'>Please select at least one ingredient.</div>",
            {},
            pd.DataFrame(columns=["Title", "Rating", "Similarity", "Link"]),
            pd.DataFrame(columns=["Nutrient", "% Daily Value (%DV)"])
        )

    ing_set = {str(i).lower().strip() for i in selected_items}
    x_df = pd.DataFrame([[1 if f in ing_set else 0 for f in features]], columns=features)
    pred_class = str(clf_model.predict(x_df)[0])

    # Probabilities
    probs = {}
    if hasattr(clf_model, "predict_proba"):
        for cls_name, p in zip(clf_model.classes_, clf_model.predict_proba(x_df)[0]):
            probs[cls_name.capitalize()] = round(float(p), 3)

    v_title, v_color, v_desc = VERDICTS.get(pred_class, (pred_class, "#333", ""))

    verdict_html = f"""
    <div style='background:{v_color}10; border:1.5px solid {v_color}; border-radius:12px; padding:18px 22px;'>
      <div style='font-size:18px; font-weight:800; color:{v_color}; margin-bottom:6px;'>{v_title}</div>
      <div style='font-size:14px; color:#222; line-height:1.5;'>{v_desc}</div>
    </div>
    """

    # Similar Recipes
    scores = similar_df["norm_set"].apply(lambda s: jaccard(ing_set, s))
    best = scores.sort_values(ascending=False).head(3)
    rec_rows = []
    if not best.empty:
        for idx, score_val in best.items():
            r = similar_df.loc[idx]
            url = r.get("url", "")
            title = r.get("title", "Untitled")
            rating = round(float(r.get("rating", 0.0)), 2)
            match_pct = f"{round(score_val * 100, 1)}%"
            link_html = f"<a href='{url}' target='_blank'>Open Recipe</a>" if url else "-"
            rec_rows.append({"Title": title, "Rating": f"★ {rating}", "Similarity": match_pct, "Link": link_html})
    recipes_table = pd.DataFrame(rec_rows)

    # Nutrition Facts (%DV)
    agg_nutr = {}
    for ing in ing_set:
        if ing in nutrition_df.index:
            row = nutrition_df.loc[ing]
            for col, val in row.items():
                agg_nutr[col] = agg_nutr.get(col, 0.0) + float(val)

    primary_cols = [
        "Protein", "Total Fat", "Total Carbohydrate", "Dietary Fiber", "Sodium", "Added Sugars",
        "Calcium", "Iron", "Potassium", "Vitamin C", "Vitamin A", "Vitamin B12"
    ]
    nutr_rows = []
    for col in primary_cols:
        val = round(agg_nutr.get(col, 0.0), 1)
        nutr_rows.append({"Nutrient": col, "% Daily Value (%DV)": f"{val}%"})
    nutrition_table = pd.DataFrame(nutr_rows)

    return verdict_html, probs, recipes_table, nutrition_table


def generate_daily_diet():
    nutrients = ["calories_pct", "fat_pct", "protein_pct", "sodium_pct"]
    b_list = meals_by_type["breakfast"]
    l_list = meals_by_type["lunch"]
    d_list = meals_by_type["dinner"]

    valid = []
    for _ in range(5000):
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
            score = round(b.get("rating", 0.0) + l.get("rating", 0.0) + d.get("rating", 0.0), 2)
            valid.append((score, b, l, d, totals))

    if not valid:
        return "<p>Could not find triplet satisfying limits. Please try again.</p>", pd.DataFrame()

    valid.sort(key=lambda x: x[0], reverse=True)
    score, b, l, d, totals = random.choice(valid[:20])

    summary_html = f"""
    <div style='background:#f0fdf4; border:1px solid #bbf7d0; border-radius:12px; padding:16px 20px; margin-bottom:16px;'>
      <div style='font-size:16px; font-weight:700; color:#15803d; margin-bottom:4px;'>
        ✓ Balanced Triplet Found — Cumulative Score: {score} / 15.0
      </div>
      <div style='font-size:13px; color:#374151;'>
        Calories: <strong>{totals['calories_pct']}% DV</strong> | 
        Protein: <strong>{totals['protein_pct']}% DV</strong> | 
        Healthy Fat: <strong>{totals['fat_pct']}% DV</strong> | 
        Sodium: <strong>{totals['sodium_pct']}% DV</strong>
      </div>
    </div>
    """

    plan_rows = [
        {
            "Meal": "BREAKFAST",
            "Recipe": b.get("title", ""),
            "Rating": f"★ {b.get('rating', '')}",
            "Key Nutrients": f"Cal: {int(b.get('calories_pct',0))}% | Prot: {int(b.get('protein_pct',0))}% | Fat: {int(b.get('fat_pct',0))}%",
            "Link": f"<a href='{b.get('url','')}' target='_blank'>View Recipe</a>"
        },
        {
            "Meal": "LUNCH",
            "Recipe": l.get("title", ""),
            "Rating": f"★ {l.get('rating', '')}",
            "Key Nutrients": f"Cal: {int(l.get('calories_pct',0))}% | Prot: {int(l.get('protein_pct',0))}% | Fat: {int(l.get('fat_pct',0))}%",
            "Link": f"<a href='{l.get('url','')}' target='_blank'>View Recipe</a>"
        },
        {
            "Meal": "DINNER",
            "Recipe": d.get("title", ""),
            "Rating": f"★ {d.get('rating', '')}",
            "Key Nutrients": f"Cal: {int(d.get('calories_pct',0))}% | Prot: {int(d.get('protein_pct',0))}% | Fat: {int(d.get('fat_pct',0))}%",
            "Link": f"<a href='{d.get('url','')}' target='_blank'>View Recipe</a>"
        },
    ]
    return summary_html, pd.DataFrame(plan_rows)


# BUILD GRADIO BLOCKS
with gr.Blocks(title="NutriTaste AI — Culinary ML Studio") as demo:
    gr.Markdown("# 🌿 NutriTaste AI: Culinary Intelligence & Dietary Optimization")
    gr.Markdown("Predict recipe rating suitability, calculate official USDA % Daily Value metrics, and optimize daily meal plans.")

    with gr.Tab("🍳 Recipe Evaluator"):
        with gr.Row():
            with gr.Column(scale=1):
                ingredients_input = gr.Dropdown(
                    choices=ALL_KNOWN_INGREDIENTS,
                    value=["milk", "honey", "jam"],
                    multiselect=True,
                    label="Pantry Ingredients",
                    info="Select or type ingredients from our 497-item dictionary"
                )
                with gr.Row():
                    btn_panna = gr.Button("🍯 Panna Cotta", size="sm")
                    btn_roast = gr.Button("🍗 Roast Poultry", size="sm")
                    btn_salad = gr.Button("🥗 Salad", size="sm")
                    btn_clash = gr.Button("🧪 Experimental", size="sm")

                eval_btn = gr.Button("Evaluate Ingredients", variant="primary")

            with gr.Column(scale=1):
                verdict_output = gr.HTML()
                proba_output = gr.Label(label="Confidence Distribution")

        with gr.Row():
            with gr.Column():
                gr.Markdown("### 📖 Top Matching Recipes (Epicurious)")
                recipes_out = gr.DataFrame(datatype=["str", "str", "str", "html"])
            with gr.Column():
                gr.Markdown("### 📊 Nutritional Breakdown (% Daily Value)")
                nutrition_out = gr.DataFrame()

        eval_btn.click(
            evaluate_ingredients,
            inputs=[ingredients_input],
            outputs=[verdict_output, proba_output, recipes_out, nutrition_out]
        )
        btn_panna.click(lambda: ["milk", "honey", "jam"], outputs=[ingredients_input])
        btn_roast.click(lambda: ["chicken", "garlic", "rosemary", "thyme"], outputs=[ingredients_input])
        btn_salad.click(lambda: ["spinach", "egg", "tomato", "arugula", "feta"], outputs=[ingredients_input])
        btn_clash.click(lambda: ["saffron", "gin", "chile pepper"], outputs=[ingredients_input])

    with gr.Tab("🥗 Daily Diet Optimizer"):
        gr.Markdown("### Stochastic 3-Meal Optimizer (50% – 100% Daily Value Constraint)")
        generate_btn = gr.Button("Generate Balanced Menu", variant="primary")
        diet_summary = gr.HTML()
        diet_table = gr.DataFrame(datatype=["str", "str", "str", "str", "html"])

        generate_btn.click(
            generate_daily_diet,
            outputs=[diet_summary, diet_table]
        )

if __name__ == "__main__":
    demo.launch()
