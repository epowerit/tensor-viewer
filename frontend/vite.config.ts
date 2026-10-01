import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

// A second checkout can run beside the default one by overriding these.
const port = Number(process.env.PORT) || 5173;
const api = process.env.TENSORVIEWER_API_URL || "http://127.0.0.1:8000";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    host: "127.0.0.1",
    port,
    strictPort: true,
    proxy: { "/api": api },
  },
});
