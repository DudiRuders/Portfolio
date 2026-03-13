import { defineConfig } from 'astro/config';
import tailwindcss from '@tailwindcss/vite';

export default defineConfig({
  site: 'https://DudiRuders.github.io',
  base: '/Portfolio',
  vite: {
    plugins: [tailwindcss()],
  },
});