# tshirt-studio – installation guide (EasyPanel / Hostinger VPS)

Workflow n8n: **AI T-Shirt Design Studio (3dstardesignShop)** (ID NEvpkjSiBYyHBrOz, déjà créé sur votre n8n).

## 1. Déployer le service `tshirt-studio` (Python: Pillow, rembg, Tesseract)
1. EasyPanel → projet de votre n8n → **+ Service → App**. Nom: `tshirt-studio`.
2. Source: GitHub (poussez ce dossier) ou Dockerfile. Build = `Dockerfile` de ce dossier. Port **8000**. Pas de domaine public nécessaire.
3. **Mounts** → Volume, mount path `/data/tshirt-generator` (les PNG/ZIP persistent ici).
4. Environnement (seulement ce que vous utilisez):
   - `API_KEY` (optionnel)
   - ComfyUI: `COMFYUI_URL=http://comfyui:8188`, `COMFY_CKPT=flux1-schnell-fp8.safetensors`
   - Cloudflare Workers AI (FLUX.1-schnell, quota gratuit quotidien): `CF_ACCOUNT_ID`, `CF_API_TOKEN`
   - Pollinations: rien (anonyme, ~1 requête/15 s, filigrane possible). `POLLINATIONS_TOKEN` optionnel.
   - AI Horde: rien (clé anonyme 0000000000), lent.
5. Test: depuis la console n8n/EasyPanel `curl http://tshirt-studio:8000/health` → tesseract: true.

## 2. Dans n8n
1. Ouvrir le workflow. Dans **Config** et **Gallery Settings**: vérifier `sidecarUrl` (par défaut `http://tshirt-studio:8000`, même réseau EasyPanel). Dans **Config**: `geminiModel` (mettez le modèle gratuit actuel de votre compte AI Studio), ou `ollamaUrl`/`ollamaModel`.
2. Le nœud *Gemini Concepts* utilise votre credential « Google Gemini(PaLM) Api account ».
3. **Publish** le workflow (les URLs /form et /webhook ne marchent qu'en production).
4. Formulaire: `https://VOTRE-N8N/form/tshirt-studio` → Galerie live: `https://VOTRE-N8N/webhook/tshirt-gallery`.
5. Réglage conseillé n8n: pas de timeout d'exécution (`EXECUTIONS_TIMEOUT=-1`), la génération de 10 images peut durer longtemps.

## 3. Test rapide
1. Lancer avec 2 designs, image provider `pollinations`.
2. Regarder la galerie: cartes « Generating… » puis images. Badge **QC passed** seulement si OCR + alpha + résolution + recadrage + espace vide validés.

## Limites honnêtes
- Score d'opportunité = estimation IA. Seuls les « measured signals » viennent de Google Trends RSS / Google Autocomplete (aucun volume ni concurrence inventés).
- Gemini image n'a pas de quota API gratuit (vérifié sept. 2026) → non utilisé.
- Les fournisseurs gratuits sortent ~1024 px: upscale Lanczos vers 4500×5400 (avertissement si ×3,2+). Pour une netteté d'impression, utilisez ComfyUI (GPU) + upscaler ESRGAN.
- FLUX sur un VPS sans GPU: très lent / RAM élevée. Utilisez Cloudflare ou Pollinations en attendant.
- Le ZIP est relayé par n8n (mémoire): 10 PNG ≈ plusieurs dizaines de Mo.
- Testé ici: traitement d'image, QC, OCR, API, ZIP/CSV avec images synthétiques, et logique du workflow avec données simulées. NON testé: les fournisseurs d'images réels, ComfyUI, rembg.
