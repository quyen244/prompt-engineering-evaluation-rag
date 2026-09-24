FROM node:24.18.1-bookworm-slim

WORKDIR /app

COPY package*.json ./
RUN npm install

COPY . .

EXPOSE 5173

ENV VITE_BACKEND_URL=http://rag-api:8000

CMD ["npx", "vite", "--host", "0.0.0.0", "--port", "5173"]
