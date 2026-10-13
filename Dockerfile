# Slim Python base — small image, fast Cloud Build.
FROM python:3.12-slim

WORKDIR /app

# Install dependencies first (separate layer) so Docker can cache this
# step and skip reinstalling on every code-only change.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Now copy the rest of the app.
COPY . .

# Cloud Run injects PORT at runtime (defaults to 8080) and expects the
# container to listen on it — 0.0.0.0, not 127.0.0.1, so traffic from
# outside the container can reach it.
ENV PORT=8080
EXPOSE 8080

# Shell form (not exec-array form) so ${PORT} is expanded by the shell
# at container start, not baked in at build time.
CMD exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT}
