FROM python:3.11-slim

# Create a non-root user with UID 1000 (required by Hugging Face)
RUN useradd -m -u 1000 user

WORKDIR /app

# Copy files and set ownership
COPY --chown=user:user . .

# Install requirements
RUN pip install --no-cache-dir -r requirements.txt

# Switch to non-root user
USER user

# Set environmental default port (Hugging Face expects 7860)
ENV PORT=7860

# Run database seeder on boot, then start the Flask application
CMD ["sh", "-c", "python preload_2026_db.py && python app.py"]
