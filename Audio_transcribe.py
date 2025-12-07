import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import ttkbootstrap as tb
from ttkbootstrap.constants import *
from ttkbootstrap.scrolled import ScrolledText
import whisper
import threading
from pathlib import Path
import os
import sqlite3
import datetime
import json
import re
import requests
import traceback
import subprocess
import platform
from openai import OpenAI
import torch

# FIX: Enable MPS Fallback just in case
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# -------------------------------------------------------------------------
# 1. Configuration & Persistence
# -------------------------------------------------------------------------
CONFIG_FILE = "vna_config.json"

class ConfigManager:
    @staticmethod
    def load():
        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE, 'r') as f:
                    return json.load(f)
            except:
                return {}
        return {}

    @staticmethod
    def save(data):
        try:
            existing = ConfigManager.load()
            existing.update(data)
            with open(CONFIG_FILE, 'w') as f:
                json.dump(existing, f, indent=4)
        except Exception as e:
            print(f"Failed to save config: {e}")

# -------------------------------------------------------------------------
# 2. Database Handler
# -------------------------------------------------------------------------
class DatabaseHandler:
    def __init__(self, db_name="voice_notes_data.db"):
        self.db_name = db_name
        self.init_db()

    def init_db(self):
        conn = sqlite3.connect(self.db_name)
        c = conn.cursor()
        c.execute('''
            CREATE TABLE IF NOT EXISTS voice_notes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT,
                file_path TEXT,
                date_recorded TIMESTAMP,
                date_processed TIMESTAMP,
                transcription TEXT,
                summary TEXT,
                calls_to_action TEXT,
                tone TEXT,
                people_mentioned TEXT,
                tags TEXT,
                subject TEXT,
                ai_provider TEXT,
                ai_model TEXT
            )
        ''')
        conn.commit()
        conn.close()

    def clean_list(self, val):
        if isinstance(val, list):
            result = []
            for item in val:
                if isinstance(item, list):
                    result.extend(item)
                else:
                    result.append(item)
            return ", ".join(str(x) for x in result)
        return str(val)

    def insert_record(self, data):
        conn = sqlite3.connect(self.db_name)
        c = conn.cursor()
        c.execute('''
            INSERT INTO voice_notes 
            (filename, file_path, date_recorded, date_processed, transcription, 
             summary, calls_to_action, tone, people_mentioned, tags, subject, ai_provider, ai_model)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            data['filename'],
            data['file_path'],
            data['date_recorded'],
            datetime.datetime.now().isoformat(),
            data['transcription'],
            data.get('summary', 'N/A'),
            self.clean_list(data.get('calls_to_action', [])),
            data.get('tone', 'N/A'),
            self.clean_list(data.get('people_mentioned', [])),
            self.clean_list(data.get('tags', [])),
            data.get('subject', 'General'),
            data.get('ai_provider', 'unknown'),
            data.get('ai_model', 'unknown')
        ))
        conn.commit()
        conn.close()

    def get_all_notes(self, search_query=""):
        conn = sqlite3.connect(self.db_name)
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        
        sql = "SELECT * FROM voice_notes"
        params = []
        
        if search_query:
            sql += " WHERE transcription LIKE ? OR summary LIKE ? OR tags LIKE ? OR subject LIKE ?"
            wildcard = f"%{search_query}%"
            params = [wildcard, wildcard, wildcard, wildcard]
            
        sql += " ORDER BY date_processed DESC"
        
        c.execute(sql, params)
        rows = [dict(row) for row in c.fetchall()]
        conn.close()
        return rows

    def delete_note(self, note_id):
        conn = sqlite3.connect(self.db_name)
        c = conn.cursor()
        c.execute("DELETE FROM voice_notes WHERE id = ?", (note_id,))
        conn.commit()
        conn.close()


# -------------------------------------------------------------------------
# SQL helper utilities (shared with Explore tab)
# -------------------------------------------------------------------------
def fetch_tables(db_path: Path):
    if not Path(db_path).exists():
        raise FileNotFoundError(f"SQLite database not found: {db_path}")
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name")
        tables = [row[0] for row in cur.fetchall()]
    finally:
        conn.close()
    if not tables:
        raise RuntimeError(f"No user tables found in database: {db_path}")
    return tables


def fetch_columns(db_path: Path, table_name: str):
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.execute(f'PRAGMA table_info("{table_name}")')
        columns = [(row[1], row[2] or "TEXT") for row in cur.fetchall()]
    finally:
        conn.close()
    if not columns:
        raise RuntimeError(f"No columns found for table '{table_name}'.")
    return columns


def build_query_system_prompt(columns, table_name):
    column_lines = "\n".join(f"- {name} ({ctype})" for name, ctype in columns)
    return f"""You are an assistant that writes SQLite SELECT queries for the '{table_name}' table.
Columns:
{column_lines}

Rules:
- Return only the SQL query text; no prose or code fences.
- Use only SELECT (or CTE + SELECT). Do not use INSERT, UPDATE, DELETE, DROP, ALTER, PRAGMA, or ATTACH.
- Default to LIMIT 200 rows unless the user explicitly asks for aggregated results such as COUNT or summary statistics.
- Use double quotes around column names when helpful.
- For text matching, use LIKE with wildcards and LOWER(...) to make comparisons case-insensitive.
- date_recorded and date_processed are timestamp strings; use date() or substr if needed for date filtering.
"""


def extract_sql(text):
    text = (text or "").strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]
        if lines and lines[0].lower().startswith("sql"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


def validate_sql(sql):
    lowered = sql.strip().lower()
    if not lowered:
        raise ValueError("Empty SQL returned.")
    if not (lowered.startswith("select") or lowered.startswith("with")):
        raise ValueError("Only SELECT statements are allowed.")
    if re.search(r"\b(insert|update|delete|drop|alter|pragma|attach|vacuum)\b", lowered):
        raise ValueError("Disallowed SQL keyword detected.")
    return sql


def ensure_limit(sql, default_limit=200):
    sql_no_semicolon = sql.rstrip().rstrip(";")
    lowered = sql_no_semicolon.lower()
    if re.search(r"\blimit\s+\d+(\s*,\s*\d+)?\s*$", lowered):
        return sql_no_semicolon + ";"
    return f"{sql_no_semicolon} LIMIT {default_limit};"


def execute_sql_query(sql, db_path: Path):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        cursor = conn.execute(sql)
        rows = cursor.fetchall()
        headers = [col[0] for col in cursor.description] if cursor.description else []
    finally:
        conn.close()
    return headers, rows

# -------------------------------------------------------------------------
# 3. AI Backends
# -------------------------------------------------------------------------
SYSTEM_PROMPT = """
You are an expert analyst. Analyze the provided transcription.
Output strictly valid JSON with these keys:
- "summary": Concise summary.
- "calls_to_action": List of tasks/requests.
- "tone": Emotional tone.
- "people_mentioned": List of names.
- "tags": A list of exactly 5 single words.
- "subject": Short title (3-5 words).
Do not include markdown. Just the raw JSON.
"""

class OpenAIBackend:
    def __init__(self, api_key, model_name):
        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name

    def analyze(self, text):
        response = self.client.chat.completions.create(
            model=self.model_name,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": text}
            ]
        )
        return json.loads(response.choices[0].message.content)

class OllamaBackend:
    def __init__(self, base_url, model_name):
        self.base_url = base_url
        self.model_name = model_name

    def analyze(self, text):
        url = f"{self.base_url}/api/chat"
        payload = {
            "model": self.model_name,
            "messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": text}],
            "format": "json",
            "stream": False
        }
        try:
            response = requests.post(url, json=payload, timeout=120)
            if response.status_code != 200: raise Exception(f"HTTP {response.status_code}")
            return json.loads(response.json()['message']['content'])
        except Exception as e:
            raise Exception(f"Ollama Error: {str(e)}")

# -------------------------------------------------------------------------
# 4. GUI Application
# -------------------------------------------------------------------------
class VoiceNoteApp(tb.Window):
    def __init__(self):
        super().__init__(themename="superhero")
        self.title("Voice Note Analyst Pro 2.0")
        self.geometry("1000x800")
        
        self.db = DatabaseHandler()
        self.config = ConfigManager.load()
        self.selected_files = []
        
        self.setup_ui()

    def setup_ui(self):
        # Tabs
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill="both", expand=True, padx=10, pady=10)

        # Tab 1: Process
        self.tab_process = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_process, text=" 🎙️ Transcribe & Analyze ")
        self.build_process_tab()

        # Tab 2: Library
        self.tab_library = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_library, text=" 📚 Library & History ")
        self.build_library_tab()
        # Tab 3: Explore (SQL assistant)
        self.tab_explore = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_explore, text=" Explore ")
        self.build_explore_tab()

    # ---------------------------------------------------------------------
    # Tab 1: Processing Logic
    # ---------------------------------------------------------------------
    def build_process_tab(self):
        main_container = ttk.Frame(self.tab_process, padding=15)
        main_container.pack(fill="both", expand=True)

        # Top Section: Settings
        settings_frame = tb.Labelframe(main_container, text="Configuration", padding=10)
        settings_frame.pack(fill="x", pady=(0, 10))

        # Provider Select
        self.provider_var = tk.StringVar(value=self.config.get("provider", "OpenAI"))
        tb.Label(settings_frame, text="AI Provider:").grid(row=0, column=0, sticky="w", padx=5)
        tb.Radiobutton(settings_frame, text="OpenAI", variable=self.provider_var, 
                        value="OpenAI", command=self.toggle_provider).grid(row=0, column=1, sticky="w")
        tb.Radiobutton(settings_frame, text="Ollama (Local)", variable=self.provider_var, 
                        value="Ollama", command=self.toggle_provider).grid(row=0, column=2, sticky="w")

        # Whisper Model
        tb.Label(settings_frame, text="Whisper Model:").grid(row=0, column=3, sticky="e", padx=(20, 5))
        self.whisper_model_var = tk.StringVar(value=self.config.get("whisper_model", "base"))
        whisper_combo = tb.Combobox(settings_frame, textvariable=self.whisper_model_var, 
                                     values=["tiny", "base", "small", "medium", "large", "turbo"], width=10, state="readonly")
        whisper_combo.grid(row=0, column=4, sticky="w")

        # OpenAI Inputs
        self.f_openai = ttk.Frame(settings_frame)
        self.f_openai.grid(row=1, column=0, columnspan=5, sticky="ew", pady=10)
        
        tb.Label(self.f_openai, text="OpenAI Key:").pack(side="left", padx=5)
        self.api_key_var = tk.StringVar(value=self.config.get("openai_key", ""))
        tb.Entry(self.f_openai, textvariable=self.api_key_var, width=40, show="*").pack(side="left", padx=5)
        
        tb.Label(self.f_openai, text="Model:").pack(side="left", padx=(15, 5))
        self.openai_model_var = tk.StringVar(value=self.config.get("openai_model", "gpt-5-nano"))
        tb.Entry(self.f_openai, textvariable=self.openai_model_var, width=15).pack(side="left", padx=5)

        # Ollama Inputs
        self.f_ollama = ttk.Frame(settings_frame)
        self.f_ollama.grid(row=1, column=0, columnspan=5, sticky="ew", pady=10)
        
        tb.Label(self.f_ollama, text="Ollama URL:").pack(side="left", padx=5)
        self.ollama_url_var = tk.StringVar(value=self.config.get("ollama_url", "http://localhost:11434"))
        tb.Entry(self.f_ollama, textvariable=self.ollama_url_var, width=30).pack(side="left", padx=5)
        
        tb.Label(self.f_ollama, text="Model:").pack(side="left", padx=(15, 5))
        self.ollama_model_var = tk.StringVar(value=self.config.get("ollama_model", "llama3.2"))
        tb.Entry(self.f_ollama, textvariable=self.ollama_model_var, width=15).pack(side="left", padx=5)

        self.toggle_provider()

        # Middle Section: Files
        file_frame = tb.Labelframe(main_container, text="Files to Process", padding=10)
        file_frame.pack(fill="both", expand=True, pady=5)

        btn_bar = ttk.Frame(file_frame)
        btn_bar.pack(fill="x", pady=5)
        
        tb.Button(btn_bar, text="➕ Add Audio Files", bootstyle="success", command=self.add_files).pack(side="left", padx=5)
        tb.Button(btn_bar, text="🗑️ Clear List", bootstyle="danger-outline", command=self.clear_files).pack(side="left", padx=5)
        self.lbl_count = tb.Label(btn_bar, text="0 files ready", font=("Helvetica", 10, "bold"))
        self.lbl_count.pack(side="right", padx=5)

        self.file_list = tk.Listbox(file_frame, height=6, bg="#2b2b2b", fg="white", borderwidth=0, highlightthickness=0)
        self.file_list.pack(fill="both", expand=True, padx=5, pady=5)

        # Bottom Section: Action & Log
        action_frame = ttk.Frame(main_container)
        action_frame.pack(fill="x", pady=10)
        
        self.btn_run = tb.Button(action_frame, text="🚀 START PROCESSING", bootstyle="primary", command=self.start_processing)
        self.btn_run.pack(fill="x", ipady=8)
        
        self.progress = tb.Progressbar(action_frame, mode="determinate", bootstyle="striped")
        self.progress.pack(fill="x", pady=(10,0))

        log_frame = tb.Labelframe(main_container, text="Live Logs", padding=5)
        log_frame.pack(fill="both", expand=True)
        self.txt_log = ScrolledText(log_frame, height=8, state="disabled", font=("Consolas", 9))
        self.txt_log.pack(fill="both", expand=True)

    # ---------------------------------------------------------------------
    # Tab 2: Library Logic
    # ---------------------------------------------------------------------
    def build_library_tab(self):
        main_container = ttk.Frame(self.tab_library, padding=10)
        main_container.pack(fill="both", expand=True)

        # Toolbar
        tool_bar = ttk.Frame(main_container)
        tool_bar.pack(fill="x", pady=5)

        self.entry_search = tb.Entry(tool_bar, width=30)
        self.entry_search.pack(side="left", padx=5)
        self.entry_search.bind("<Return>", lambda e: self.refresh_library())
        
        tb.Button(tool_bar, text="🔍 Search", command=self.refresh_library, bootstyle="info-outline").pack(side="left", padx=5)
        tb.Button(tool_bar, text="🔄 Refresh", command=self.refresh_library, bootstyle="secondary-outline").pack(side="left", padx=5)
        
        tb.Button(tool_bar, text="❌ Delete", command=self.delete_selected_note, bootstyle="danger").pack(side="right", padx=5)
        tb.Button(tool_bar, text="📤 Export MD", command=self.export_selected_note, bootstyle="success").pack(side="right", padx=5)
        tb.Button(tool_bar, text="▶ Play Audio", command=self.play_audio, bootstyle="warning").pack(side="right", padx=5)

        # Split View
        split = ttk.PanedWindow(main_container, orient="horizontal")
        split.pack(fill="both", expand=True, pady=5)

        # Left: List of notes
        tree_container = ttk.Frame(split)
        split.add(tree_container, weight=1)

        self.tree = ttk.Treeview(tree_container, columns=("id", "date", "subject", "tags"), show="headings", selectmode="browse")
        self.tree.heading("id", text="ID")
        self.tree.column("id", width=40, stretch=False)
        self.tree.heading("date", text="Date")
        self.tree.column("date", width=120, stretch=False)
        self.tree.heading("subject", text="Subject")
        self.tree.column("subject", width=200)
        self.tree.heading("tags", text="Tags")
        self.tree.bind("<<TreeviewSelect>>", self.on_note_select)
        self.tree.pack(side="left", fill="both", expand=True)
        
        tree_scroll = tb.Scrollbar(tree_container, orient="vertical", command=self.tree.yview)
        tree_scroll.pack(side="right", fill="y")
        self.tree.configure(yscrollcommand=tree_scroll.set)

        # Right: Details View
        self.details_panel = ScrolledText(split, height=20, font=("Segoe UI", 10), state="disabled")
        split.add(self.details_panel, weight=2)

        self.refresh_library()

    # ---------------------------------------------------------------------
    # Logic: Processing
    # ---------------------------------------------------------------------
    def toggle_provider(self):
        if self.provider_var.get() == "OpenAI":
            self.f_ollama.grid_remove()
            self.f_openai.grid()
        else:
            self.f_openai.grid_remove()
            self.f_ollama.grid()

    def add_files(self):
        files = filedialog.askopenfilenames(filetypes=[("Audio Files", "*.mp3 *.wav *.m4a *.ogg")])
        for f in files:
            if f not in self.selected_files:
                self.selected_files.append(f)
                self.file_list.insert("end", Path(f).name)
        self.lbl_count.config(text=f"{len(self.selected_files)} files ready")

    def clear_files(self):
        self.selected_files = []
        self.file_list.delete(0, "end")
        self.lbl_count.config(text="0 files ready")

    def log(self, msg):
        self.txt_log.text.configure(state="normal")
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        self.txt_log.text.insert("end", f"[{ts}] {msg}\n")
        self.txt_log.text.see("end")
        self.txt_log.text.configure(state="disabled")

    def save_current_config(self):
        cfg = {
            "provider": self.provider_var.get(),
            "openai_key": self.api_key_var.get(),
            "openai_model": self.openai_model_var.get(),
            "ollama_url": self.ollama_url_var.get(),
            "ollama_model": self.ollama_model_var.get(),
            "whisper_model": self.whisper_model_var.get()
        }
        ConfigManager.save(cfg)

    def start_processing(self):
        if not self.selected_files:
            messagebox.showwarning("No Files", "Please add audio files first.")
            return

        self.save_current_config()
        
        # Setup Backend
        provider = self.provider_var.get()
        try:
            if provider == "OpenAI":
                if not self.api_key_var.get(): raise ValueError("OpenAI Key Missing")
                backend = OpenAIBackend(self.api_key_var.get(), self.openai_model_var.get())
            else:
                backend = OllamaBackend(self.ollama_url_var.get(), self.ollama_model_var.get())
        except Exception as e:
            messagebox.showerror("Config Error", str(e))
            return

        self.btn_run.config(state="disabled", text="Processing...")
        self.progress["value"] = 0
        
        threading.Thread(target=self.run_thread, args=(backend,), daemon=True).start()

    def parse_date(self, filepath):
        try:
            return datetime.datetime.strptime(Path(filepath).stem[:11], "%y%m%d_%H%M").isoformat()
        except:
            return datetime.datetime.fromtimestamp(os.path.getctime(filepath)).isoformat()

    def run_thread(self, backend):
        try:
            self.log(f"Loading Whisper ({self.whisper_model_var.get()})...")
            
            device = "cpu"
            if torch.cuda.is_available(): device = "cuda"
            elif torch.backends.mps.is_available(): device = "mps"
            self.log(f"Using Compute Device: {device.upper()}")

            model = whisper.load_model(self.whisper_model_var.get(), device=device)
            
            total = len(self.selected_files)
            
            for i, fpath in enumerate(self.selected_files):
                p = Path(fpath)
                self.log(f"Processing ({i+1}/{total}): {p.name}")
                
                # Transcribe
                # CRITICAL FIX: Disable FP16 on MPS to avoid NaN errors
                fp16_mode = False if device == "mps" else True
                self.log(f"  (Transcribing with fp16={fp16_mode})")
                
                res = model.transcribe(str(p), fp16=fp16_mode)
                text = res['text'].strip()
                
                # Analyze
                self.log("Analyzing text...")
                analysis = {}
                try:
                    analysis = backend.analyze(text)
                except Exception as e:
                    self.log(f"Analysis Error: {e}")
                    analysis = {"summary": "Analysis Failed", "tags": ["error"]}

                record = {
                    'filename': p.name,
                    'file_path': str(p),
                    'date_recorded': self.parse_date(p),
                    'transcription': text,
                    'ai_provider': self.provider_var.get(),
                    'ai_model': backend.model_name,
                    **analysis
                }
                self.db.insert_record(record)
                
                progress_pct = ((i + 1) / total) * 100
                self.after(0, lambda: self.progress.configure(value=progress_pct))
            
            self.log("Batch Complete!")
            self.after(0, lambda: messagebox.showinfo("Success", "All files processed successfully!"))
            self.after(0, self.refresh_library)

        except Exception as e:
            self.log(f"CRITICAL ERROR: {e}")
            traceback.print_exc()
            self.after(0, lambda: messagebox.showerror("Error", str(e)))
        finally:
            self.after(0, lambda: self.btn_run.config(state="normal", text="🚀 START PROCESSING"))

    # ---------------------------------------------------------------------
    # Logic: Library
    # ---------------------------------------------------------------------
    def refresh_library(self):
        for item in self.tree.get_children():
            self.tree.delete(item)
        
        query = self.entry_search.get()
        records = self.db.get_all_notes(query)
        
        for r in records:
            try:
                dt = datetime.datetime.fromisoformat(r['date_recorded'])
                d_str = dt.strftime("%Y-%m-%d %H:%M")
            except:
                d_str = r['date_recorded']
                
            self.tree.insert("", "end", iid=r['id'], values=(r['id'], d_str, r['subject'], r['tags']))

    def on_note_select(self, event):
        selected = self.tree.selection()
        if not selected: return
        
        note_id = selected[0]
        conn = sqlite3.connect(self.db.db_name)
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        c.execute("SELECT * FROM voice_notes WHERE id=?", (note_id,))
        row = c.fetchone()
        conn.close()
        
        if row:
            self.display_details(dict(row))

    def display_details(self, data):
        self.details_panel.text.configure(state="normal")
        self.details_panel.text.delete("1.0", "end")
        
        content = f"""
# {data['subject']}
**Date:** {data['date_recorded']}
**File:** {data['filename']}
**Tone:** {data['tone']}
**Tags:** {data['tags']}

### Summary
{data['summary']}

### Calls to Action
{data['calls_to_action']}

### People Mentioned
{data['people_mentioned']}

---
### Full Transcription
{data['transcription']}
"""
        self.details_panel.text.insert("1.0", content)
        self.details_panel.text.configure(state="disabled")

    def delete_selected_note(self):
        selected = self.tree.selection()
        if not selected: return
        if messagebox.askyesno("Confirm", "Delete this record?"):
            self.db.delete_note(selected[0])
            self.details_panel.text.configure(state="normal")
            self.details_panel.text.delete("1.0", "end")
            self.details_panel.text.configure(state="disabled")
            self.refresh_library()

    def play_audio(self):
        selected = self.tree.selection()
        if not selected: return
        
        conn = sqlite3.connect(self.db.db_name)
        c = conn.cursor()
        c.execute("SELECT file_path FROM voice_notes WHERE id=?", (selected[0],))
        row = c.fetchone()
        conn.close()
        
        if row and os.path.exists(row[0]):
            fpath = row[0]
            if platform.system() == 'Darwin':
                subprocess.call(('open', fpath))
            elif platform.system() == 'Windows':
                os.startfile(fpath)
            else:
                subprocess.call(('xdg-open', fpath))
        else:
            messagebox.showerror("Error", "Audio file not found on disk.")

    def export_selected_note(self):
        selected = self.tree.selection()
        if not selected: return
        
        note_id = selected[0]
        conn = sqlite3.connect(self.db.db_name)
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        c.execute("SELECT * FROM voice_notes WHERE id=?", (note_id,))
        data = dict(c.fetchone())
        conn.close()

        save_path = filedialog.asksaveasfilename(
            defaultextension=".md",
            initialfile=f"{data['subject'].replace(' ', '_')}.md",
            filetypes=[("Markdown", "*.md"), ("Text", "*.txt")]
        )
        
        if save_path:
            content = self.details_panel.text.get("1.0", "end")
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(content)
            messagebox.showinfo("Exported", f"Saved to {save_path}")

    # ---------------------------------------------------------------------
    # Tab 3: Explore (SQL assistant)
    # ---------------------------------------------------------------------
    def build_explore_tab(self):
        self.explore_db_path = Path(self.db.db_name)
        self.explore_tables = []
        self.explore_columns = []
        self.explore_table_var = tk.StringVar()
        self.explore_status_var = tk.StringVar(value="Ready")

        main = ttk.Frame(self.tab_explore, padding=12)
        main.pack(fill="both", expand=True)

        db_row = ttk.Frame(main)
        db_row.pack(fill="x", pady=(0, 8))
        tb.Label(db_row, text="Database:").pack(side="left")
        self.lbl_explore_db = tb.Label(db_row, text=str(self.explore_db_path))
        self.lbl_explore_db.pack(side="left", padx=(6, 8))
        tb.Button(db_row, text="Choose DB...", bootstyle="secondary", command=self.handle_choose_db).pack(side="left")

        fields_frame = tb.Labelframe(main, text="Fields", padding=8)
        fields_frame.pack(side="left", fill="y", padx=(0, 10))
        self.fields_list_explore = tk.Listbox(fields_frame, height=12)
        self.fields_list_explore.pack(fill="both", expand=True, padx=4, pady=4)
        self.fields_frame_explore = fields_frame

        right_frame = ttk.Frame(main)
        right_frame.pack(side="left", fill="both", expand=True)

        table_row = ttk.Frame(right_frame)
        table_row.pack(fill="x", pady=(0, 6))
        tb.Label(table_row, text="Table:").pack(side="left")
        self.table_combo_explore = ttk.Combobox(table_row, textvariable=self.explore_table_var, state="readonly")
        self.table_combo_explore.pack(side="left", fill="x", expand=True, padx=(6, 0))
        self.table_combo_explore.bind("<<ComboboxSelected>>", self.handle_explore_table_change)

        tb.Label(right_frame, text="Ask in plain English").pack(anchor="w")
        self.prompt_text_explore = ScrolledText(right_frame, height=5)
        self.prompt_text_explore.pack(fill="x", pady=(2, 8))

        btn_row = ttk.Frame(right_frame)
        btn_row.pack(fill="x")
        self.btn_generate_sql_explore = tb.Button(btn_row, text="Generate SQL", bootstyle="primary", command=self.handle_generate_sql_explore)
        self.btn_generate_sql_explore.pack(side="left")
        self.btn_run_sql_explore = tb.Button(btn_row, text="Run Query", bootstyle="success", command=self.handle_run_sql_explore)
        self.btn_run_sql_explore.pack(side="left", padx=(6, 0))

        tb.Label(right_frame, text="Generated SQL (editable)").pack(anchor="w", pady=(10, 0))
        self.sql_text_explore = ScrolledText(right_frame, height=8)
        self.sql_text_explore.pack(fill="both", expand=True, pady=(2, 8))

        self.lbl_explore_status = ttk.Label(self.tab_explore, textvariable=self.explore_status_var, anchor="w", relief="sunken")
        self.lbl_explore_status.pack(fill="x", padx=12, pady=(0, 8))

        self.load_explore_db(self.explore_db_path)

    def load_explore_db(self, db_path: Path):
        try:
            tables = fetch_tables(db_path)
        except Exception as exc:
            self.explore_status_var.set("Database load failed")
            messagebox.showerror("Database error", str(exc))
            return

        self.explore_db_path = Path(db_path)
        self.lbl_explore_db.config(text=str(self.explore_db_path))
        self.explore_tables = tables
        preferred = "voice_notes" if "voice_notes" in tables else tables[0]
        self.table_combo_explore["values"] = tables
        self.explore_table_var.set(preferred)
        self.load_explore_table(preferred)

    def load_explore_table(self, table_name: str):
        try:
            self.explore_columns = fetch_columns(self.explore_db_path, table_name)
        except Exception as exc:
            self.explore_status_var.set("Table load failed")
            messagebox.showerror("Schema error", str(exc))
            return
        self.refresh_explore_fields()
        self.explore_status_var.set(f"Using table '{table_name}' from {self.explore_db_path.name}")

    def refresh_explore_fields(self):
        self.fields_list_explore.delete(0, tk.END)
        for name, ctype in self.explore_columns:
            self.fields_list_explore.insert(tk.END, f"{name} ({ctype})")
        self.fields_frame_explore.config(text=f"Fields in {self.explore_table_var.get()}")
        self.fields_list_explore.configure(height=max(len(self.explore_columns), 8))

    def set_explore_busy(self, busy: bool = True, status: str | None = None):
        state = "disabled" if busy else "normal"
        if hasattr(self, "btn_generate_sql_explore"):
            self.btn_generate_sql_explore.config(state=state)
        if hasattr(self, "btn_run_sql_explore"):
            self.btn_run_sql_explore.config(state=state)
        if status is not None:
            self.explore_status_var.set(status)

    def handle_explore_table_change(self, _event=None):
        selection = self.explore_table_var.get()
        if selection:
            self.load_explore_table(selection)

    def handle_choose_db(self):
        file_path = filedialog.askopenfilename(
            title="Select SQLite database",
            filetypes=[("SQLite DB", "*.db *.sqlite *.sqlite3"), ("All files", "*.*")],
        )
        if not file_path:
            return
        self.load_explore_db(Path(file_path))

    def handle_generate_sql_explore(self):
        user_prompt = self.prompt_text_explore.text.get("1.0", "end").strip()
        if not user_prompt:
            messagebox.showinfo("Missing input", "Please enter what you want to ask about the data.")
            return
        if not self.explore_columns:
            messagebox.showerror("Schema error", "No columns available; select a database/table first.")
            return

        key = self.api_key_var.get().strip()
        model = self.openai_model_var.get().strip() or "gpt-5"
        if not key:
            messagebox.showerror("Missing key", "Enter an OpenAI API key in the Configuration section.")
            return

        table_name = self.explore_table_var.get()
        self.set_explore_busy(True, "Contacting OpenAI...")

        def worker():
            try:
                client = OpenAI(api_key=key)
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": build_query_system_prompt(self.explore_columns, table_name)},
                        {"role": "user", "content": user_prompt},
                    ],
                )
                content = response.choices[0].message.content
                sql = extract_sql(content)
                sql = ensure_limit(validate_sql(sql))
            except Exception as exc:
                def fail():
                    self.set_explore_busy(False, "Failed to generate SQL")
                    messagebox.showerror("OpenAI error", str(exc))
                self.after(0, fail)
                return

            def finish():
                self.sql_text_explore.text.delete("1.0", "end")
                self.sql_text_explore.text.insert("1.0", sql)
                self.set_explore_busy(False, "SQL generated; review or run it.")
            self.after(0, finish)

        threading.Thread(target=worker, daemon=True).start()

    def handle_run_sql_explore(self):
        sql = self.sql_text_explore.text.get("1.0", "end").strip()
        if not sql:
            messagebox.showinfo("Missing SQL", "Generate or type a SQL query first.")
            return
        self.set_explore_busy(True, "Running query...")

        def worker():
            try:
                sql_clean = ensure_limit(validate_sql(sql))
                headers, rows = execute_sql_query(sql_clean, self.explore_db_path)
            except Exception as exc:
                def fail():
                    self.set_explore_busy(False, "Query error")
                    messagebox.showerror("Query error", str(exc))
                self.after(0, fail)
                return

            def finish():
                self.set_explore_busy(False, f"Query ran successfully ({len(rows)} rows).")
                self.show_results_explore(headers, rows)
            self.after(0, finish)

        threading.Thread(target=worker, daemon=True).start()

    def show_results_explore(self, headers, rows):
        window = tk.Toplevel(self)
        window.title("Query Results")
        window.geometry("1000x600")

        frame = ttk.Frame(window, padding=10)
        frame.pack(fill="both", expand=True)

        tree = ttk.Treeview(frame, columns=headers, show="headings")
        for col in headers:
            tree.heading(col, text=col)
            tree.column(col, anchor="w", width=150)

        vsb = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
        hsb = ttk.Scrollbar(frame, orient="horizontal", command=tree.xview)
        tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)

        tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")

        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)

        max_chars = 300
        clean_rows = []
        for row in rows:
            vals = []
            for col in headers:
                val = row[col]
                if val is None:
                    val = ""
                val = str(val).replace("\n", " ")
                if len(val) > max_chars:
                    val = val[:max_chars] + "…"
                vals.append(val)
            clean_rows.append(vals)

        batch_size = 100
        total = len(clean_rows)

        def insert_batch(start=0):
            end = min(start + batch_size, total)
            for vals in clean_rows[start:end]:
                tree.insert("", tk.END, values=vals)
            if end < total:
                window.after(1, insert_batch, end)

        insert_batch(0)

        footer = ttk.Label(frame, text=f"{len(rows)} rows returned.")
        footer.grid(row=2, column=0, columnspan=2, sticky="w", pady=(6, 0))

if __name__ == "__main__":
    try:
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(1)
    except: pass
    
    app = VoiceNoteApp()
    app.mainloop()
