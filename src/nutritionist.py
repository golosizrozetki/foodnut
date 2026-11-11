#!/usr/bin/env python3
"""Главный скрипт: прогноз + нутриенты + похожие рецепты."""
import sys
from recipes import NutritionFacts, SimilarRecipes, RatingPredictor

MESSAGES = {
    "bad":    "You might find it tasty, but in our opinion, it is a bad idea\n"
              "to have a dish with that list of ingredients.",
    "so-so":  "It sounds okay, but not spectacular — a so-so dish at best.",
    "great":  "Great choice! These ingredients should make a delicious dish.",
}


# Бонус

def _print_recipe_block(r):
    print(f"{r['title']} (rating: {r['rating']})")
    print("Ingredients:")
    for ing in str(r["ingredients"]).split("|"):
        print(f"- {ing}")
    print("Nutrients:")
    for label, col in [
        ("calories", "calories_pct"),
        ("protein",  "protein_pct"),
        ("fat",      "fat_pct"),
        ("sodium",   "sodium_pct"),
    ]:
        print(f"- {label}: {r[col]:.0f}%")
    print(f"URL: {r['url']}")


def print_menu():
    from recipes import MenuGenerator
    menu = MenuGenerator().generate()
    if menu is None:
        print("Не удалось собрать меню, удовлетворяющее ограничениям. "
              "Ослабь MIN_TOTAL/MAX_TOTAL в MenuGenerator.")
        return
    for meal in ("breakfast", "lunch", "dinner"):
        print(meal.upper())
        print("-" * 21)
        _print_recipe_block(menu[meal])
        print()


def main():
    if "--menu" in sys.argv:
        print_menu()
        return

    if len(sys.argv) < 2:
        print("Usage: ./nutritionist.py milk, honey, jam")
        sys.exit(1)

    raw = " ".join(sys.argv[1:])
    ingredients = [i.strip() for i in raw.split(",") if i.strip()]
    if not ingredients:
        print("No ingredients provided.")
        sys.exit(1)

    # ---------- I. FORECAST ----------
    print("I. OUR FORECAST")
    try:
        predictor = RatingPredictor()
        pred = predictor.predict(ingredients)
        print(MESSAGES.get(pred, f"Predicted class: {pred}"))
    except FileNotFoundError:
        print("recipe_model.joblib не найден — сначала запусти recipes.ipynb")

    # ---------- II. NUTRITION FACTS ----------
    print("\nII. NUTRITION FACTS")
    try:
        nf = NutritionFacts()
        facts = nf.get_facts(ingredients)
    except FileNotFoundError:
        print("nutrition_facts.csv не найден")
        facts = {}

    for ing in ingredients:
        if ing not in facts:
            print(f"\n{ing.title()} — not found in our database")
            continue
        print(f"\n{ing.title()}")
        for nutrient, value in facts[ing].items():
            if value and value > 0:
                print(f"{nutrient} - {value:.0f}% of Daily Value")

    # ---------- III. SIMILAR RECIPES ----------
    print("\nIII. TOP-3 SIMILAR RECIPES:")
    try:
        sr = SimilarRecipes()
        recipes = sr.find_similar(ingredients)
    except FileNotFoundError:
        print("similar_recipes.csv не найден")
        recipes = []

    if not recipes:
        print("No similar recipes found.")
    else:
        for r in recipes:
            print(f"- {r['title']}, rating: {r['rating']}, URL: {r['url']}")


if __name__ == "__main__":
    main()