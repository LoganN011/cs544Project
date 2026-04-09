import sqlite3
import os
from cryptography.fernet import Fernet

class Database:
    def __init__(self, db_path='passwords.db', key_path='secret.key'):
        self.db_path = db_path
        self.key_path = key_path
        self.key = self._load_or_generate_key()
        self.fernet = Fernet(self.key)
        self._init_db()

    def _load_or_generate_key(self):
        if os.path.exists(self.key_path):
            with open(self.key_path, 'rb') as f:
                return f.read()
        else:
            key = Fernet.generate_key()
            with open(self.key_path, 'wb') as f:
                f.write(key)
            return key

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            # New structure: One user can have multiple labeled passwords
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS vault (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT,
                    label TEXT,
                    encrypted_password BLOB,
                    UNIQUE(username, label)
                )
            ''')
            # Cleanup old table if it exists
            cursor.execute("DROP TABLE IF EXISTS users")
            conn.commit()

    def save_password(self, username, label, password):
        encrypted = self.fernet.encrypt(password.encode())
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO vault (username, label, encrypted_password)
                VALUES (?, ?, ?)
            ''', (username, label, encrypted))
            conn.commit()

    def get_labels(self, username):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT label FROM vault WHERE username = ?', (username,))
            rows = cursor.fetchall()
            return [row[0] for row in rows]

    def get_password(self, username, label):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT encrypted_password FROM vault 
                WHERE username = ? AND label = ?
            ''', (username, label))
            row = cursor.fetchone()
            if row:
                try:
                    return self.fernet.decrypt(row[0]).decode()
                except Exception as e:
                    print(f"Decryption error: {e}")
                    return None
        return None

if __name__ == "__main__":
    # Test
    db = Database()
    db.save_password("test_user", "Google", "secret123")
    db.save_password("test_user", "Bank", "money456")
    print(f"Labels for test_user: {db.get_labels('test_user')}")
    print(f"Bank Password: {db.get_password('test_user', 'Bank')}")
