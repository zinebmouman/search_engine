# Build
FROM node:20-alpine AS build
WORKDIR /app

COPY package*.json ./
RUN npm ci

COPY . .
RUN npm run build

# Serve (Nginx)
FROM nginx:alpine
COPY --from=build /app/dist /usr/share/nginx/html
# Nginx conf sera gérée par le container reverse-proxy principal (voir plus bas)
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
