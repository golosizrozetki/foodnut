import os, re, json
import pandas as pd
import joblib
import random

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_GZ_PATH = os.path.join(BASE_DIR, "recipe_model.joblib.gz")
MODEL_PATH = os.path.join(BASE_DIR, "recipe_model.joblib")
NUTRITION_CSV = os.path.join(BASE_DIR, "nutrition_facts.csv")
SIMILAR_CSV   = os.path.join(BASE_DIR, "similar_recipes.csv")
MENU_CSV = os.path.join(BASE_DIR, "menu_recipes.csv")


class NutritionFacts:
    def __init__(self, csv_path=NUTRITION_CSV):
        self.df = pd.read_csv(csv_path, index_col=0).fillna(0.0)
        self.df.columns = [
            str(c).replace(" (%DV)", "").strip() for c in self.df.columns
        ]

    def has(self, ingredient: str) -> bool:
        return ingredient.lower() in self.df.index

    def get_facts(self, ingredients: list) -> dict:
        out = {}
        for ing in ingredients:
            key = ing.lower().strip()
            if key not in self.df.index:
                continue
            out[ing] = self.df.loc[key].to_dict()
        return out

    def missing(self, ingredients: list) -> list:
        return [i for i in ingredients if i.lower().strip() not in self.df.index]


class SimilarRecipes:
    def __init__(self, csv_path=SIMILAR_CSV):
        self.df = pd.read_csv(csv_path)
        self.df["norm_set"] = self.df["norm_set"].apply(
            lambda s: set(s.split("|")) if isinstance(s, str) and s else set()
        )

    @staticmethod
    def _jaccard(a: set, b: set) -> float:
        if not a or not b:
            return 0.0
        return len(a & b) / len(a | b)

    def find_similar(self, ingredients: list, top_n=3, min_score=0.4):
        q = {i.lower().strip() for i in ingredients}
        if not q:
            return []
        scores = self.df["norm_set"].apply(lambda s: self._jaccard(q, s))
        best = scores.sort_values(ascending=False).head(top_n)
        if best.empty or best.iloc[0] < min_score:
            return []
        return self.df.loc[best.index, ["title", "rating", "url"]].assign(
            score=best.values
        ).to_dict("records")


class RatingPredictor:
    """Загружает recipe_model.joblib (или сжатый .joblib.gz) и предсказывает класс блюда."""
    def __init__(self, model_path=None):
        if model_path is None:
            model_path = MODEL_GZ_PATH if os.path.exists(MODEL_GZ_PATH) else MODEL_PATH
        payload = joblib.load(model_path)
        self.model    = payload["model"]
        self.features = payload["features"]
        self.classes  = payload["classes"]

    def predict(self, ingredients):
        ing_set = {i.lower().strip() for i in ingredients}
        x = pd.DataFrame(
            [[1 if f in ing_set else 0 for f in self.features]],
            columns=self.features,
        )
        return str(self.model.predict(x)[0])


class MenuGenerator:
    """Генерирует дневное меню: 1 рецепт на завтрак, обед, ужин.

    Ограничение: суммарный %DV по каждому нутриенту из NUTRIENTS
    лежит в [MIN_TOTAL, MAX_TOTAL] (покрываем «основную часть», не превышая).
    Выбор — случайный из TOP_K лучших валидных троек по сумме рейтингов.
    """
    NUTRIENTS = ["calories_pct", "fat_pct", "protein_pct", "sodium_pct"]
    MIN_TOTAL = 50.0
    MAX_TOTAL = 100.0

    def __init__(self, csv_path=MENU_CSV):
        self.df = pd.read_csv(csv_path).fillna(0.0)
        self.by_meal = {
            m: self.df[self.df["meal_type"] == m].reset_index(drop=True)
            for m in ("breakfast", "lunch", "dinner")
        }
        for m, d in self.by_meal.items():
            if d.empty:
                raise ValueError(f"В menu_recipes.csv нет рецептов для {m}")

    def _valid(self, b, l, d):
        for n in self.NUTRIENTS:
            total = b[n] + l[n] + d[n]
            if not (self.MIN_TOTAL <= total <= self.MAX_TOTAL):
                return False
        return True

    def generate(self, n_samples=5000, top_k=20, seed=None):
        if seed is not None:
            random.seed(seed)

        valid = []
        for _ in range(n_samples):
            b = self.by_meal["breakfast"].sample(1).iloc[0]
            l = self.by_meal["lunch"].sample(1).iloc[0]
            d = self.by_meal["dinner"].sample(1).iloc[0]
            if self._valid(b, l, d):
                score = b["rating"] + l["rating"] + d["rating"]
                valid.append((score, b, l, d))

        if not valid:
            return None

        valid.sort(key=lambda x: x[0], reverse=True)
        _, b, l, d = random.choice(valid[:top_k])
        return {"breakfast": b, "lunch": l, "dinner": d}

    