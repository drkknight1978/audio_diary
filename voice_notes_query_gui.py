import json
import re
import sqlite3
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

from openai import OpenAI

CONFIG_PATH = Path(__file__).resolve().parent / "vna_config.json"
DEFAULT_DB = Path(__file__).resolve().parent / "voice_notes_data.db"


def read_config():
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Missing config file: {CONFIG_PATH}")
    data = json.loads(CONFIG_PATH.read_text())
    key = data.get("openai_key")
    model = data.get("openai_model", "gpt-5")
    if not key:
        raise ValueError("openai_key not found in vna_config.json")
    return key, model


def find_default_db():
    if DEFAULT_DB.exists():
        return DEFAULT_DB
    alt = DEFAULT_DB.with_suffix(".db.sql")
    if alt.exists():
        return alt
    raise FileNotFoundError("No default SQLite database found. Use 'Choose DB...' to select one.")


def fetch_tables(db_path: Path):
    if not db_path.exists():
        raise FileNotFoundError(f"SQLite database not found: {db_path}")
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
        )
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


def build_system_prompt(columns, table_name):
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


def execute_query(sql, db_path: Path):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        cursor = conn.execute(sql)
        rows = cursor.fetchall()
        headers = [col[0] for col in cursor.description] if cursor.description else []
    finally:
        conn.close()
    return headers, rows


class QueryApp:
    def __init__(self, root, client, model, initial_db: Path):
        self.root = root
        self.client = client
        self.model = model
        self.db_path = Path(initial_db)
        self.tables = []
        self.columns = []
        self.table_var = tk.StringVar()
        self.status_var = tk.StringVar(value="Ready")
        self._build_ui()
        self.load_db(self.db_path)

    def _build_ui(self):
        self.root.title("Voice Notes SQL Assistant")
        self.root.geometry("1000x650")

        main = ttk.Frame(self.root, padding=12)
        main.pack(fill="both", expand=True)

        db_row = ttk.Frame(main)
        db_row.pack(fill="x", pady=(0, 10))
        ttk.Label(db_row, text="Database:").pack(side="left")
        self.db_label = ttk.Label(db_row, text=str(self.db_path))
        self.db_label.pack(side="left", padx=(4, 8))
        ttk.Button(db_row, text="Choose DB...", command=self.handle_choose_db).pack(side="left")

        fields_frame = ttk.LabelFrame(main, text="Fields")
        fields_frame.pack(side="left", fill="y", padx=(0, 10))
        self.fields_frame = fields_frame

        self.fields_list = tk.Listbox(fields_frame, height=12)
        self.fields_list.pack(fill="both", padx=8, pady=8)

        right_frame = ttk.Frame(main)
        right_frame.pack(side="left", fill="both", expand=True)

        table_row = ttk.Frame(right_frame)
        table_row.pack(fill="x", pady=(0, 6))
        ttk.Label(table_row, text="Table:").pack(side="left")
        self.table_combo = ttk.Combobox(table_row, textvariable=self.table_var, state="readonly")
        self.table_combo.pack(side="left", fill="x", expand=True, padx=(4, 0))
        self.table_combo.bind("<<ComboboxSelected>>", self.handle_table_change)

        prompt_label = ttk.Label(right_frame, text="Ask in plain English")
        prompt_label.pack(anchor="w")
        self.prompt_text = tk.Text(right_frame, height=6)
        self.prompt_text.pack(fill="x", pady=(2, 8))

        buttons_frame = ttk.Frame(right_frame)
        buttons_frame.pack(fill="x")
        ttk.Button(buttons_frame, text="Generate SQL", command=self.handle_generate_sql).pack(side="left")
        ttk.Button(buttons_frame, text="Run Query", command=self.handle_run_query).pack(side="left", padx=(6, 0))

        sql_label = ttk.Label(right_frame, text="Generated SQL (editable)")
        sql_label.pack(anchor="w", pady=(10, 0))
        self.sql_text = tk.Text(right_frame, height=10)
        self.sql_text.pack(fill="both", expand=True, pady=(2, 8))

        status = ttk.Label(self.root, textvariable=self.status_var, relief="sunken", anchor="w", padding=6)
        status.pack(fill="x")

    def load_db(self, db_path: Path):
        try:
            tables = fetch_tables(db_path)
        except Exception as exc:
            self.status_var.set("Database load failed")
            messagebox.showerror("Database error", str(exc))
            return

        self.db_path = Path(db_path)
        self.db_label.config(text=str(self.db_path))
        self.tables = tables
        preferred = "voice_notes" if "voice_notes" in tables else tables[0]
        self.table_combo["values"] = tables
        self.table_var.set(preferred)
        self.load_table(preferred)

    def load_table(self, table_name: str):
        try:
            self.columns = fetch_columns(self.db_path, table_name)
        except Exception as exc:
            self.status_var.set("Table load failed")
            messagebox.showerror("Schema error", str(exc))
            return
        self.refresh_fields_list()
        self.status_var.set(f"Using table '{table_name}' from {self.db_path.name}")

    def refresh_fields_list(self):
        self.fields_list.delete(0, tk.END)
        for name, ctype in self.columns:
            self.fields_list.insert(tk.END, f"{name} ({ctype})")
        self.fields_frame.config(text=f"Fields in {self.table_var.get()}")
        self.fields_list.configure(height=max(len(self.columns), 8))

    def handle_table_change(self, _event=None):
        selection = self.table_var.get()
        if selection:
            self.load_table(selection)

    def handle_choose_db(self):
        file_path = filedialog.askopenfilename(
            title="Select SQLite database",
            filetypes=[("SQLite DB", "*.db *.sqlite *.sqlite3"), ("All files", "*.*")],
        )
        if not file_path:
            return
        self.load_db(Path(file_path))

    def handle_generate_sql(self):
        user_prompt = self.prompt_text.get("1.0", tk.END).strip()
        if not user_prompt:
            messagebox.showinfo("Missing input", "Please enter what you want to ask about the data.")
            return

        if not self.columns:
            messagebox.showerror("Schema error", "No columns available; select a database/table first.")
            return

        table_name = self.table_var.get()
        self.status_var.set("Contacting OpenAI...")
        self.root.update_idletasks()

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": build_system_prompt(self.columns, table_name)},
                    {"role": "user", "content": user_prompt},
                ],
            )
            content = response.choices[0].message.content
            sql = extract_sql(content)
            sql = ensure_limit(validate_sql(sql))
        except Exception as exc:
            self.status_var.set("Failed to generate SQL")
            messagebox.showerror("OpenAI error", str(exc))
            return

        self.sql_text.delete("1.0", tk.END)
        self.sql_text.insert(tk.END, sql)
        self.status_var.set("SQL generated; review or run it.")

    def handle_run_query(self):
        sql = self.sql_text.get("1.0", tk.END).strip()
        if not sql:
            messagebox.showinfo("Missing SQL", "Generate or type a SQL query first.")
            return

        try:
            sql = ensure_limit(validate_sql(sql))
            headers, rows = execute_query(sql, self.db_path)
        except Exception as exc:
            self.status_var.set("Query error")
            messagebox.showerror("Query error", str(exc))
            return

        self.status_var.set(f"Query ran successfully ({len(rows)} rows).")
        self.show_results(headers, rows)

    def show_results(self, headers, rows):
        window = tk.Toplevel(self.root)
        window.title("Query Results")
        window.geometry("900x500")

        frame = ttk.Frame(window, padding=10)
        frame.pack(fill="both", expand=True)

        tree = ttk.Treeview(frame, columns=headers, show="headings")
        for col in headers:
            tree.heading(col, text=col)
            tree.column(col, anchor="w", width=120)

        vsb = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
        hsb = ttk.Scrollbar(frame, orient="horizontal", command=tree.xview)
        tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)

        tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")

        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)

        for row in rows:
            values = [row[col] for col in headers]
            tree.insert("", tk.END, values=values)

        footer = ttk.Label(frame, text=f"{len(rows)} rows returned.")
        footer.grid(row=2, column=0, columnspan=2, sticky="w", pady=(6, 0))


def main():
    root = tk.Tk()
    try:
        key, model = read_config()
        db_path = find_default_db()
        client = OpenAI(api_key=key)
    except Exception as exc:
        messagebox.showerror("Startup error", str(exc))
        root.destroy()
        return

    QueryApp(root, client, model, db_path)
    root.mainloop()


if __name__ == "__main__":
    main()
