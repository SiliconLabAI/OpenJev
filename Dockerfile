# OpenJev server — production container.
# Build:  docker build -t openjev .
# Run:    docker run -p 3001:3001 --env-file .env openjev
FROM node:22-alpine AS build

WORKDIR /app
COPY package.json package-lock.json* ./
RUN npm ci || npm install

COPY tsconfig.json vite.config.ts index.html ./
COPY server ./server
COPY src ./src
RUN npm run build

# ── runtime ─────────────────────────────────────────────────────────────────
FROM node:22-alpine

WORKDIR /app
ENV NODE_ENV=production

COPY package.json package-lock.json* ./
RUN npm ci --omit=dev || npm install --omit=dev

COPY --from=build /app/dist ./dist
COPY server ./server
COPY tsconfig.json ./

EXPOSE 3001
# Graceful-shutdown window must fit inside the orchestrator's stop timeout.
ENV PORT=3001
CMD ["npx", "tsx", "server/index.ts"]
