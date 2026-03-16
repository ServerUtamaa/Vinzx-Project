
# 🚀 PANDUAN DEPLOYMENT: ONLINE & KENDALI PENUH

Agar aplikasi ini bisa diakses orang lain (Public) dan Anda punya kendali penuh (Admin), ikuti 3 Tahap ini. Jangan ada yang terlewat.

---

## ✅ TAHAP 1: DATABASE (Tempat Simpan User & Token)
Kita pakai **Neon.tech** (Gratis, Cepat, Online 24 Jam).

1. Buka [Neon.tech](https://neon.tech) dan Daftar/Login.
2. Klik **"New Project"**. Beri nama `nvr-db`.
3. Setelah jadi, Anda akan melihat **Connection String** (Contoh: `postgresql://nvr:password123@ep-xyz.neon.tech/neondb...`).
   👉 **COPY & SIMPAN KODE ITU.**
4. Klik menu **"SQL Editor"** di Neon.
5. Buka file `database/schema.sql` di project ini, copy semua isinya, paste ke SQL Editor Neon, lalu klik **Run**.
   *(Ini akan membuat tabel User, History, dll)*.

---

## ✅ TAHAP 2: BACKEND (Otak Server & Admin)
Kita pakai **Render.com** (Gratis). Ini agar fitur Login & Admin Panel berfungsi.

1. Upload seluruh folder project ini ke **GitHub** Anda.
2. Buka [Render.com](https://render.com), Login.
3. Klik **New +** -> **Web Service**.
4. Pilih repo GitHub Anda.
5. **Setting Render (PENTING):**
   - **Root Directory:** `backend` (Wajib diisi ini!).
   - **Build Command:** `npm install`
   - **Start Command:** `node server.js`
   - **Instance Type:** Free.
6. Scroll ke bawah, cari **Environment Variables**. Klik "Add Environment Variable":
   - Key: `DATABASE_URL`
   - Value: (Paste Connection String dari Neon tadi)
   - Key: `JWT_SECRET`
   - Value: `RAHASIA_NEGARA_V99` (Bebas isi apa aja)
   - Key: `NODE_ENV`
   - Value: `production`
7. Klik **Deploy Web Service**.
8. Tunggu sampai selesai. Render akan memberi Anda URL (Contoh: `https://nvr-api.onrender.com`).
   👉 **COPY & SIMPAN URL INI.**

---

## ✅ TAHAP 3: FRONTEND (Tampilan Website)
Kita pakai **Vercel** (Tercepat untuk React).

1. Buka [Vercel.com](https://vercel.com), Login.
2. Klik **Add New...** -> **Project**.
3. Import repo GitHub yang sama.
4. **Setting Vercel (PENTING):**
   - **Framework Preset:** Vite (Biasanya otomatis).
   - **Root Directory:** `./` (Biarkan kosong/default).
   - **Environment Variables:** (Klik tab Environment Variables)
     - Nama: `VITE_API_URL`
     - Isi: `https://nvr-api.onrender.com` (URL dari Render tadi).
     - Nama: `VITE_GEMINI_API_KEY`
     - Isi: `AIzaSy...` (API Key Gemini Google Anda).
5. Klik **Deploy**.
6. **SELESAI!** Vercel akan memberi Anda link website (Contoh: `https://nvr-ai.vercel.app`).

---

## 👑 CARA KENDALI PENUH (ADMIN CONTROL)

Sekarang website sudah online. Bagaimana cara Anda mengaturnya?

1. Buka website Vercel Anda di HP/Laptop.
2. Klik tombol User di pojok kanan atas -> Klik "Login".
3. **Login Sakti (Backdoor Developer):**
   - Username: `System Admin`
   - Password: `admin_secure_key`
4. Anda akan masuk mode **DEV**.
   - Klik **System Menu** (Icon Gear/Menu).
   - Klik **User Database**.
   - Di sini Anda bisa melihat siapa saja yang daftar.
   - Anda bisa **Tambah Token**, **Banned User**, atau **Aktifkan Membership** mereka secara Realtime.

**Selamat! Anda sekarang adalah Owner Aplikasi Trading AI.** 🚀
