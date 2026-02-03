# 1. Use Python 3.11 to support Scikit-Learn 1.7+
FROM python:3.11-slim

WORKDIR /app

# 2. Install uv
RUN pip install uv

# 3. Copy requirements (containing scikit-learn>=1.7.0)
COPY requirements.txt .

# 4. Install dependencies
RUN uv pip install -r requirements.txt --system

# 5. Copy application code and model
COPY predict.py .
COPY model.pkl .

# 6. Expose port
EXPOSE 5000

# 7. Run with Uvicorn
CMD ["uvicorn", "predict:app", "--host", "0.0.0.0", "--port", "5000"]