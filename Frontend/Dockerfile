# ---- Build Stage ----
FROM node:18 AS build

WORKDIR /app

# Only copy package.json/package-lock.json first (better caching)
COPY package*.json ./
RUN npm ci

# Copy entire frontend source
COPY . .

# Build Vite project → output goes to /app/dist
RUN npm run build


# ---- Production Stage (NGINX) ----
FROM nginx:stable-alpine

# Copy custom nginx configuration
COPY nginx.front.conf /etc/nginx/conf.d/default.conf

# Copy built frontend assets
COPY --from=build /app/dist /usr/share/nginx/html

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
