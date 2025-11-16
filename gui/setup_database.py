import sqlite3
import os

# Database file path
DB_PATH = os.path.join(os.path.dirname(__file__), 'users.db')

def setup_database():
    """Initialize the database with users table and default users."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Create users table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            role TEXT NOT NULL
        )
    ''')

    # Insert default users (check if they exist first)
    users = [
        ('developer', '12121212', 'owner'),
        ('operator', '12121212', 'operator')
    ]

    for username, password, role in users:
        try:
            cursor.execute(
                'INSERT INTO users (username, password, role) VALUES (?, ?, ?)',
                (username, password, role)
            )
            print(f"Added user: {username} with role: {role}")
        except sqlite3.IntegrityError:
            print(f"User {username} already exists")

    conn.commit()
    conn.close()
    print(f"Database setup complete at: {DB_PATH}")

if __name__ == '__main__':
    setup_database()
