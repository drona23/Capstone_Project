# ─── React Frontend Dockerfile (Multi-Stage) ──────────────────────────────────

# ── Stage 1: Build ────────────────────────────────────────────────────────────
# Use Node to compile TypeScript and bundle with Vite
FROM node:20-slim AS builder

WORKDIR /app

# Install dependencies first (layer-cached unless package-lock.json changes)
COPY package.json package-lock.json ./
RUN npm ci

# Copy source and build
COPY . .
RUN npm run build
# Output is in /app/dist/

# ── Stage 2: Serve ────────────────────────────────────────────────────────────
# Use tiny nginx image to serve the static files
FROM nginx:alpine

# Copy only the compiled output from Stage 1
COPY --from=builder /app/dist /usr/share/nginx/html

# Replace default nginx config with our proxy config
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Nginx serves on port 80 by default
EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
