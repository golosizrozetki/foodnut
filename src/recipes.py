import os, re, json
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
NUTRITION_CSV = os.path.join(BASE_DIR, "nutrition_facts.csv")
SIMILAR_CSV   = os.path.join(BASE_DIR, "similar_recipes.csv")


class NutritionFacts:
    def __init__(self, csv_path=NUTRITION_CSV):
        self.df = pd.read_csv(csv_path, index_col=0).fillna(0.0)

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