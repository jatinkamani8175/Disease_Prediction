"""
auth.py — Authentication module for the Disease Prediction App.
Uses SQLite to store user credentials securely (password hashing via hashlib).
"""

import sqlite3
import hashlib
import os

# ── Path to SQLite database file ──────────────────────────────────────────────
DB_PATH = os.path.join(os.path.dirname(__file__), "users.db")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _hash_password(password: str) -> str:
    """Return a SHA-256 hex digest of the given password."""
    return hashlib.sha256(password.encode()).hexdigest()


def _get_connection() -> sqlite3.Connection:
    """Open (or create) the SQLite database and return a connection."""
    conn = sqlite3.connect(DB_PATH)
    return conn


def init_db() -> None:
    """
    Create the `users` table if it doesn't already exist.
    Call this once at app start-up.
    """
    conn = _get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id       INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT    UNIQUE NOT NULL,
            password TEXT    NOT NULL
        )
        """
    )
    conn.commit()
    conn.close()


# ── Public API ────────────────────────────────────────────────────────────────

def register_user(username: str, password: str) -> tuple[bool, str]:
    """
    Register a new user.

    Returns:
        (True, "success message")  on success.
        (False, "error message")   if the username already exists or input is invalid.
    """
    username = username.strip()
    if not username or not password:
        return False, "Username and password cannot be empty."

    if len(username) < 3:
        return False, "Username must be at least 3 characters long."

    if len(password) < 6:
        return False, "Password must be at least 6 characters long."

    hashed = _hash_password(password)
    try:
        conn = _get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO users (username, password) VALUES (?, ?)",
            (username, hashed),
        )
        conn.commit()
        conn.close()
        return True, f"Account created for '{username}'. You can now log in."
    except sqlite3.IntegrityError:
        return False, f"Username '{username}' is already taken. Choose another."


def login_user(username: str, password: str) -> tuple[bool, str]:
    """
    Validate login credentials.

    Returns:
        (True, "welcome message")  on success.
        (False, "error message")   if credentials are invalid.
    """
    username = username.strip()
    if not username or not password:
        return False, "Please enter both username and password."

    hashed = _hash_password(password)
    conn = _get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT id FROM users WHERE username = ? AND password = ?",
        (username, hashed),
    )
    row = cursor.fetchone()
    conn.close()

    if row:
        return True, f"Welcome back, {username}!"
    return False, "Invalid username or password."


def user_exists(username: str) -> bool:
    """Return True if the username exists in the database."""
    conn = _get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM users WHERE username = ?", (username.strip(),))
    row = cursor.fetchone()
    conn.close()
    return row is not None
