
import tkinter as tk
from tkinter import filedialog, messagebox
import customtkinter as ctk
from PIL import Image, ImageTk, ImageDraw, ImageFont
import threading
import os
import sys
import time
import random
import math

# ─── Theme Config ─────────────────────────────────────────────────────────────
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

BG       = "#0b0f1e"
CARD     = "#141b2d"
PANEL    = "#1a2340"
ACCENT   = "#7c6aff"
CYAN     = "#00d4ff"
PINK     = "#ff5e9e"
GOLD     = "#ffb743"
GREEN    = "#2ee89e"
RED      = "#ff4d6a"
TEXT     = "#eef2ff"
SUBTEXT  = "#7a89a7"
BORDER   = "#283352"

MODELS_INFO = [
    {"name": "ResNet50 + LSTM",       "icon": "🔵", "color": "#7c6aff", "params": "~28M",  "speed": "Fast",      "desc": "Deep residual features + LSTM sequential memory for high quality captions"},
    {"name": "VGG + GRU",             "icon": "🟣", "color": "#a855f7", "params": "~138M", "speed": "Medium",    "desc": "VGG-16 deep visual encoding combined with Gated Recurrent Units"},
    {"name": "CNN + SimpleRNN",       "icon": "🟡", "color": "#ffb743", "params": "~5M",   "speed": "Very Fast", "desc": "Lightweight custom CNN backbone with Simple Recurrent Network"},
    {"name": "CNN + GRU + Attention", "icon": "🟠", "color": "#ff5e9e", "params": "~12M",  "speed": "Medium",    "desc": "CNN feature extraction with Attention-weighted GRU decoder"},
    {"name": "CNN + Transformers",    "icon": "🔴", "color": "#00d4ff", "params": "~20M",  "speed": "Slow",      "desc": "Convolutional encoder with Transformer decoder (state-of-the-art)"},
]

DEMO_CAPTIONS_EN = {
    "ResNet50 + LSTM":       "A group of people walking along a busy city street with tall buildings in the background",
    "VGG + GRU":             "A man riding a red bicycle on a sunny day near a park",
    "CNN + SimpleRNN":       "A dog running across a green field chasing a ball",
    "CNN + GRU + Attention": "A young woman smiling brightly while holding a cup of coffee at a cafe",
    "CNN + Transformers":    "Two children playing together on a colorful playground surrounded by trees",
}


class ImageCaptioningApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("🧠 Neural Image Captioning Studio")
        self.geometry("1350x830")
        self.minsize(1200, 750)
        self.configure(fg_color=BG)

        self._image_path = None
        self._photo_preview = None
        self._selected_model = tk.StringVar(value=MODELS_INFO[0]["name"])
        self._is_running = False
        self._results = {}
        self._model_widgets = {}

        self._build_header()
        self._build_body()

    # ══════════════════════════════════════════════════════════════════════════
    #  HEADER
    # ══════════════════════════════════════════════════════════════════════════
    def _build_header(self):
        header = ctk.CTkFrame(self, fg_color=CARD, corner_radius=0, height=64)
        header.pack(fill="x")
        header.pack_propagate(False)

        ctk.CTkLabel(
            header, text="🧠  Neural Image Captioning Studio",
            font=ctk.CTkFont(size=20, weight="bold"), text_color=TEXT
        ).pack(side="left", padx=24)

        # Badges
        badge_fr = ctk.CTkFrame(header, fg_color="transparent")
        badge_fr.pack(side="right", padx=20)
        for txt, clr in [("5 Models", ACCENT), ("Deep Learning", PINK), ("TensorFlow", "#ff6b35")]:
            ctk.CTkLabel(
                badge_fr, text=f"  {txt}  ", font=ctk.CTkFont(size=10, weight="bold"),
                text_color="white", fg_color=clr, corner_radius=10
            ).pack(side="left", padx=3, pady=10)

        # Gradient line
        ctk.CTkFrame(self, fg_color=ACCENT, height=3, corner_radius=0).pack(fill="x")

    # ══════════════════════════════════════════════════════════════════════════
    #  BODY
    # ══════════════════════════════════════════════════════════════════════════
    def _build_body(self):
        body = ctk.CTkFrame(self, fg_color="transparent")
        body.pack(fill="both", expand=True, padx=16, pady=12)
        body.columnconfigure(0, weight=2)
        body.columnconfigure(1, weight=5)
        body.rowconfigure(0, weight=1)

        self._build_sidebar(body)
        self._build_main_area(body)

    # ══════════════════════════════════════════════════════════════════════════
    #  SIDEBAR (Left)
    # ══════════════════════════════════════════════════════════════════════════
    def _build_sidebar(self, parent):
        sidebar = ctk.CTkFrame(parent, fg_color=CARD, corner_radius=16, border_width=1, border_color=BORDER)
        sidebar.grid(row=0, column=0, sticky="nsew", padx=(0, 8))

        # ── Image Section ──
        ctk.CTkLabel(
            sidebar, text="📷  Upload Image", font=ctk.CTkFont(size=15, weight="bold"), text_color=CYAN
        ).pack(anchor="w", padx=18, pady=(16, 6))

        # Preview frame
        self._preview_frame = ctk.CTkFrame(sidebar, fg_color=PANEL, corner_radius=12, height=200)
        self._preview_frame.pack(fill="x", padx=14, pady=(0, 8))
        self._preview_frame.pack_propagate(False)

        self._preview_label = ctk.CTkLabel(
            self._preview_frame, text="🖼️  No image selected\nClick 'Browse' below",
            font=ctk.CTkFont(size=13), text_color=SUBTEXT
        )
        self._preview_label.pack(expand=True)

        # Image info label
        self._img_info = ctk.CTkLabel(
            sidebar, text="", font=ctk.CTkFont(size=10), text_color=SUBTEXT
        )
        self._img_info.pack(anchor="w", padx=18)

        # Browse button
        self._browse_btn = ctk.CTkButton(
            sidebar, text="📂  Browse Image", font=ctk.CTkFont(size=13, weight="bold"),
            fg_color=ACCENT, hover_color="#6555dd", corner_radius=10, height=40,
            command=self._browse_image
        )
        self._browse_btn.pack(fill="x", padx=14, pady=(6, 12))

        # ── Model Selection ──
        ctk.CTkLabel(
            sidebar, text="🤖  Select Model", font=ctk.CTkFont(size=15, weight="bold"), text_color=CYAN
        ).pack(anchor="w", padx=18, pady=(4, 6))

        model_scroll = ctk.CTkScrollableFrame(sidebar, fg_color="transparent", height=200)
        model_scroll.pack(fill="x", padx=10, pady=(0, 6))

        for m in MODELS_INFO:
            self._make_model_radio(model_scroll, m)

        # ── Action Buttons ──
        self._gen_single_btn = ctk.CTkButton(
            sidebar, text="⚡  Generate (Selected)", font=ctk.CTkFont(size=13, weight="bold"),
            fg_color=GREEN, hover_color="#22c07e", text_color="#0b0f1e",
            corner_radius=10, height=42, command=lambda: self._start_gen(mode="single")
        )
        self._gen_single_btn.pack(fill="x", padx=14, pady=(8, 4))

        self._gen_all_btn = ctk.CTkButton(
            sidebar, text="🚀  Generate ALL Models", font=ctk.CTkFont(size=13, weight="bold"),
            fg_color=PINK, hover_color="#d44a85", corner_radius=10, height=42,
            command=lambda: self._start_gen(mode="all")
        )
        self._gen_all_btn.pack(fill="x", padx=14, pady=(4, 6))

        self._clear_btn = ctk.CTkButton(
            sidebar, text="🗑️  Clear Results", font=ctk.CTkFont(size=12),
            fg_color="#2a3352", hover_color="#3a4562", corner_radius=10, height=34,
            command=self._clear_results
        )
        self._clear_btn.pack(fill="x", padx=14, pady=(0, 14))

    def _make_model_radio(self, parent, m):
        card = ctk.CTkFrame(parent, fg_color=PANEL, corner_radius=10)
        card.pack(fill="x", pady=3, padx=2)

        top_row = ctk.CTkFrame(card, fg_color="transparent")
        top_row.pack(fill="x", padx=8, pady=(6, 2))

        rb = ctk.CTkRadioButton(
            top_row, text=f'{m["icon"]} {m["name"]}',
            font=ctk.CTkFont(size=12, weight="bold"), text_color=TEXT,
            fg_color=m["color"], border_color=m["color"], hover_color=m["color"],
            variable=self._selected_model, value=m["name"],
            command=self._on_model_select
        )
        rb.pack(side="left")

        ctk.CTkLabel(
            top_row, text=f'{m["speed"]} • {m["params"]}',
            font=ctk.CTkFont(size=9), text_color=SUBTEXT
        ).pack(side="right")

    def _on_model_select(self):
        sel = self._selected_model.get()
        info = next((m for m in MODELS_INFO if m["name"] == sel), None)
        if info:
            self._model_detail_name.configure(text=f'{info["icon"]}  {info["name"]}', text_color=info["color"])
            self._model_detail_desc.configure(text=info["desc"])
            self._model_detail_stats.configure(text=f'Parameters: {info["params"]}  |  Speed: {info["speed"]}')
            # Show result if available
            if sel in self._results:
                self._detail_caption.configure(text=f'"{self._results[sel]}"', text_color=TEXT)
            else:
                self._detail_caption.configure(text="Click 'Generate' to get caption for this model", text_color=SUBTEXT)

    # ══════════════════════════════════════════════════════════════════════════
    #  MAIN AREA (Right)
    # ══════════════════════════════════════════════════════════════════════════
    def _build_main_area(self, parent):
        main = ctk.CTkFrame(parent, fg_color="transparent")
        main.grid(row=0, column=1, sticky="nsew", padx=(8, 0))
        main.rowconfigure(0, weight=0)
        main.rowconfigure(1, weight=1)
        main.columnconfigure(0, weight=1)

        # Status + Progress
        status_bar = ctk.CTkFrame(main, fg_color=CARD, corner_radius=12, height=50, border_width=1, border_color=BORDER)
        status_bar.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        status_bar.pack_propagate(False)

        self._status_icon = ctk.CTkLabel(status_bar, text="●", font=ctk.CTkFont(size=16), text_color=GREEN)
        self._status_icon.pack(side="left", padx=(16, 6))

        self._status_text = ctk.CTkLabel(
            status_bar, text="Ready — Select an image and model to begin",
            font=ctk.CTkFont(size=12), text_color=TEXT
        )
        self._status_text.pack(side="left")

        self._progress = ctk.CTkProgressBar(status_bar, fg_color=PANEL, progress_color=ACCENT, height=8, width=200)
        self._progress.pack(side="right", padx=16)
        self._progress.set(0)

        # Results Notebook
        self._tabs = ctk.CTkTabview(
            main, fg_color=CARD, corner_radius=14,
            segmented_button_fg_color=PANEL,
            segmented_button_selected_color=ACCENT,
            segmented_button_unselected_color=PANEL,
            border_width=1, border_color=BORDER
        )
        self._tabs.grid(row=1, column=0, sticky="nsew")

        self._tabs.add("🎯 Selected Model")
        self._tabs.add("📊 All Results")
        self._tabs.add("🔄 Compare")

        self._build_detail_tab()
        self._build_all_tab()
        self._build_compare_tab()
        self._tabs.set("🎯 Selected Model")

    # ── Tab: Selected Model Detail ─────────────────────────────────────────────
    def _build_detail_tab(self):
        tab = self._tabs.tab("🎯 Selected Model")

        # Model header card
        hdr_card = ctk.CTkFrame(tab, fg_color=PANEL, corner_radius=14)
        hdr_card.pack(fill="x", padx=12, pady=(10, 8))

        self._model_detail_name = ctk.CTkLabel(
            hdr_card, text=f'{MODELS_INFO[0]["icon"]}  {MODELS_INFO[0]["name"]}',
            font=ctk.CTkFont(size=20, weight="bold"), text_color=MODELS_INFO[0]["color"]
        )
        self._model_detail_name.pack(anchor="w", padx=16, pady=(14, 2))

        self._model_detail_desc = ctk.CTkLabel(
            hdr_card, text=MODELS_INFO[0]["desc"],
            font=ctk.CTkFont(size=12), text_color=SUBTEXT, wraplength=600, justify="left"
        )
        self._model_detail_desc.pack(anchor="w", padx=16, pady=(0, 4))

        self._model_detail_stats = ctk.CTkLabel(
            hdr_card, text=f'Parameters: {MODELS_INFO[0]["params"]}  |  Speed: {MODELS_INFO[0]["speed"]}',
            font=ctk.CTkFont(size=11, weight="bold"), text_color=GOLD
        )
        self._model_detail_stats.pack(anchor="w", padx=16, pady=(0, 12))

        # Caption result area
        cap_card = ctk.CTkFrame(tab, fg_color=PANEL, corner_radius=14)
        cap_card.pack(fill="both", expand=True, padx=12, pady=(0, 8))

        ctk.CTkLabel(
            cap_card, text="📝  Generated Caption:",
            font=ctk.CTkFont(size=14, weight="bold"), text_color=CYAN
        ).pack(anchor="w", padx=16, pady=(14, 6))

        self._detail_caption = ctk.CTkLabel(
            cap_card, text="Click 'Generate' to get caption for this model",
            font=ctk.CTkFont(size=15), text_color=SUBTEXT,
            wraplength=550, justify="left"
        )
        self._detail_caption.pack(anchor="w", padx=20, pady=(0, 16))

        # Confidence
        conf_frame = ctk.CTkFrame(cap_card, fg_color=BG, corner_radius=10)
        conf_frame.pack(fill="x", padx=16, pady=(0, 14))

        ctk.CTkLabel(
            conf_frame, text="Confidence:", font=ctk.CTkFont(size=11, weight="bold"), text_color=SUBTEXT
        ).pack(side="left", padx=(12, 8), pady=10)

        self._conf_bar = ctk.CTkProgressBar(conf_frame, fg_color=PANEL, progress_color=GREEN, height=12, width=300)
        self._conf_bar.pack(side="left", padx=(0, 8), pady=10)
        self._conf_bar.set(0)

        self._conf_pct = ctk.CTkLabel(
            conf_frame, text="—", font=ctk.CTkFont(size=13, weight="bold"), text_color=GREEN
        )
        self._conf_pct.pack(side="left", padx=4)

    # ── Tab: All Results ───────────────────────────────────────────────────────
    def _build_all_tab(self):
        tab = self._tabs.tab("📊 All Results")
        scroll = ctk.CTkScrollableFrame(tab, fg_color="transparent")
        scroll.pack(fill="both", expand=True, padx=4, pady=4)

        for m in MODELS_INFO:
            card = ctk.CTkFrame(scroll, fg_color=PANEL, corner_radius=12)
            card.pack(fill="x", pady=4, padx=4)

            # Top: model name + badge
            top = ctk.CTkFrame(card, fg_color="transparent")
            top.pack(fill="x", padx=12, pady=(10, 4))

            ctk.CTkLabel(
                top, text=f'  {m["icon"]} {m["name"]}  ',
                font=ctk.CTkFont(size=11, weight="bold"), text_color="white",
                fg_color=m["color"], corner_radius=8
            ).pack(side="left")

            speed_label = ctk.CTkLabel(
                top, text=f'⚡ {m["speed"]}  •  {m["params"]}',
                font=ctk.CTkFont(size=10), text_color=SUBTEXT
            )
            speed_label.pack(side="right")

            # Status dot
            self._status_dot = ctk.CTkLabel(top, text="⏳", font=ctk.CTkFont(size=12))
            self._status_dot.pack(side="right", padx=8)

            # Caption
            cap = ctk.CTkLabel(
                card, text="   Waiting for generation...",
                font=ctk.CTkFont(size=12), text_color=SUBTEXT,
                wraplength=550, justify="left", anchor="w"
            )
            cap.pack(anchor="w", padx=14, pady=(0, 10))

            self._model_widgets[m["name"]] = {"caption_label": cap, "status_dot": self._status_dot}

    # ── Tab: Compare ───────────────────────────────────────────────────────────
    def _build_compare_tab(self):
        tab = self._tabs.tab("🔄 Compare")

        ctk.CTkLabel(
            tab, text="📊  Model Comparison Table",
            font=ctk.CTkFont(size=14, weight="bold"), text_color=GOLD
        ).pack(pady=(10, 6))

        self._compare_text = ctk.CTkTextbox(
            tab, fg_color=BG, text_color=TEXT,
            font=ctk.CTkFont(family="Consolas", size=12),
            corner_radius=10, border_width=1, border_color=BORDER
        )
        self._compare_text.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        self._compare_text.insert("0.0", "  Generate captions to see comparison here...\n\n"
                                          "  Click '⚡ Generate (Selected)' for one model\n"
                                          "  or '🚀 Generate ALL' to compare all 5 models!")
        self._compare_text.configure(state="disabled")

    # ══════════════════════════════════════════════════════════════════════════
    #  ACTIONS
    # ══════════════════════════════════════════════════════════════════════════
    def _browse_image(self):
        path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.webp *.tiff"), ("All", "*.*")]
        )
        if not path:
            return
        self._image_path = path
        try:
            img = Image.open(path).convert("RGB")
            w, h = img.size
            self._img_info.configure(text=f"📐 {w}×{h}  |  {os.path.basename(path)}")
            img.thumbnail((320, 190))
            self._photo_preview = ImageTk.PhotoImage(img)
            self._preview_label.configure(image=self._photo_preview, text="")
            self._status_text.configure(text=f"Image loaded: {os.path.basename(path)} — Ready to generate!")
        except Exception as e:
            messagebox.showerror("Error", f"Cannot open image:\n{e}")

    def _clear_results(self):
        self._results = {}
        self._progress.set(0)
        self._conf_bar.set(0)
        self._conf_pct.configure(text="—")
        self._detail_caption.configure(text="Click 'Generate' to get caption for this model", text_color=SUBTEXT)
        self._status_text.configure(text="Results cleared")
        self._status_icon.configure(text_color=GREEN)

        for name, w in self._model_widgets.items():
            w["caption_label"].configure(text="   Waiting for generation...", text_color=SUBTEXT)
            w["status_dot"].configure(text="⏳")

        self._compare_text.configure(state="normal")
        self._compare_text.delete("0.0", "end")
        self._compare_text.insert("0.0", "  Results cleared. Generate again to compare.")
        self._compare_text.configure(state="disabled")

    def _start_gen(self, mode="single"):
        if self._is_running:
            messagebox.showinfo("Wait", "Generation in progress...")
            return
        if not self._image_path:
            messagebox.showwarning("No Image", "Please select an image first! 📷")
            return

        self._is_running = True
        self._gen_single_btn.configure(state="disabled")
        self._gen_all_btn.configure(state="disabled")
        self._browse_btn.configure(state="disabled")
        self._progress.set(0)
        self._status_icon.configure(text_color=GOLD)

        if mode == "all":
            models_to_run = [m["name"] for m in MODELS_INFO]
            self._status_text.configure(text="🚀 Running ALL 5 models...")
        else:
            models_to_run = [self._selected_model.get()]
            self._status_text.configure(text=f"⚡ Running {models_to_run[0]}...")

        # Mark running models
        for name in models_to_run:
            if name in self._model_widgets:
                self._model_widgets[name]["caption_label"].configure(text="   🔄 Processing...", text_color=GOLD)
                self._model_widgets[name]["status_dot"].configure(text="🔄")

        thread = threading.Thread(target=self._run_inference, args=(models_to_run,), daemon=True)
        thread.start()

    def _run_inference(self, models_to_run):
        total = len(models_to_run)
        for i, name in enumerate(models_to_run):
            self.after(0, lambda n=name: self._status_text.configure(text=f"⏳ Processing: {n}..."))

            # Simulate inference time (replace with real model.predict())
            time.sleep(random.uniform(0.8, 1.8))

            caption = DEMO_CAPTIONS_EN.get(name, "An image captured by the camera")
            self._results[name] = caption

            prog = (i + 1) / total
            self.after(0, lambda n=name, c=caption, p=prog: self._on_model_done(n, c, p))

        self.after(100, self._on_all_done)

    def _on_model_done(self, name, caption, progress):
        self._progress.set(progress)

        # Update All Results tab
        if name in self._model_widgets:
            w = self._model_widgets[name]
            w["caption_label"].configure(text=f'   ✅ "{caption}"', text_color=TEXT)
            w["status_dot"].configure(text="✅")

        # Update detail tab if this is the selected model
        if name == self._selected_model.get():
            info = next((m for m in MODELS_INFO if m["name"] == name), None)
            if info:
                self._model_detail_name.configure(text=f'{info["icon"]}  {name}', text_color=info["color"])
                self._model_detail_desc.configure(text=info["desc"])
                self._model_detail_stats.configure(text=f'Parameters: {info["params"]}  |  Speed: {info["speed"]}')
            self._detail_caption.configure(text=f'"{caption}"', text_color=TEXT)
            conf = random.uniform(0.75, 0.96)
            self._animate_conf(conf)

    def _animate_conf(self, target, step=0):
        steps = 30
        if step <= steps:
            val = target * step / steps
            self._conf_bar.set(val)
            self._conf_pct.configure(text=f"{int(val * 100)}%")
            self.after(20, lambda: self._animate_conf(target, step + 1))

    def _on_all_done(self):
        self._is_running = False
        self._gen_single_btn.configure(state="normal")
        self._gen_all_btn.configure(state="normal")
        self._browse_btn.configure(state="normal")
        self._progress.set(1.0)
        self._status_icon.configure(text_color=GREEN)
        self._status_text.configure(text=f"✅ Done! {len(self._results)} model(s) completed")

        # Update compare tab
        self._update_compare()

        # Switch to appropriate tab
        if len(self._results) > 1:
            self._tabs.set("📊 All Results")
        else:
            self._tabs.set("🎯 Selected Model")

    def _update_compare(self):
        self._compare_text.configure(state="normal")
        self._compare_text.delete("0.0", "end")

        line = "  ╔═══════════════════════════════╤══════════════════════════════════════════════════════════════╗\n"
        line += f"  ║ {'MODEL':<29} │ {'GENERATED CAPTION':<60} ║\n"
        line += "  ╠═══════════════════════════════╪══════════════════════════════════════════════════════════════╣\n"
        self._compare_text.insert("end", line)

        for m in MODELS_INFO:
            name = m["name"]
            cap = self._results.get(name, "— not generated —")
            cap_short = cap[:58] + ".." if len(cap) > 60 else cap
            row = f"  ║ {m['icon']} {name:<27} │ {cap_short:<60} ║\n"
            self._compare_text.insert("end", row)

        self._compare_text.insert("end", "  ╚═══════════════════════════════╧══════════════════════════════════════════════════════════════╝\n")
        self._compare_text.configure(state="disabled")


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    # Check deps
    try:
        import customtkinter
        from PIL import Image
    except ImportError:
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "customtkinter", "Pillow"])
        print("Dependencies installed. Please restart.")
        sys.exit(0)

    app = ImageCaptioningApp()
    app.mainloop()
