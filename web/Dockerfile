FROM node:22-alpine
WORKDIR /web
COPY package*.json ./
RUN npm ci
COPY . .
ARG NEXT_PUBLIC_API=http://localhost:8000
RUN npm run build
CMD ["npm", "start"]
