FROM python:3.11-slim

# Prevent Python from writing .pyc files and buffer stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=7860

WORKDIR /app

# Install dependencies
COPY web/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend logic and data files from src
COPY src/recipes.py /app/src/recipes.py
COPY src/recipe_model.joblib /app/src/recipe_model.joblib
COPY src/nutrition_facts.csv /app/src/nutrition_facts.csv
COPY src/similar_recipes.csv /app/src/similar_recipes.csv
COPY src/menu_recipes.csv /app/src/menu_recipes.csv

# Copy web service and templates
COPY web/app.py /app/web/app.py
COPY web/templates /app/web/templates

EXPOSE 7860

# Run with Gunicorn on dynamic PORT
CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:${PORT:-7860} --workers 1 --threads 4 --timeout 120 web.app:app"]

