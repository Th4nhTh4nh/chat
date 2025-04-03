import sqlite3


def init_db():
    conn = sqlite3.connect("chat_history")
    cur = conn.cursor()
    cur.execute(
        """CREATE TABLE IF NOT EXISTS conversations (id TEXT PRIMARY KEY, title TEXT, messages TEXT, timestamp TEXT)""")
    conn.commit()
    conn.close()


def save_conversations_to_db(id, title, messages, timestamp):
    conn = sqlite3.connect("chat_history")
    cur = conn.cursor()
    cur.execute("INSERT INTO conversations VALUES (?, ?, ?, ?)", (id, title, str(messages), timestamp))
    conn.commit()
    conn.close()


def load_conversations_from_db():
    conn = sqlite3.connect("chat_history")
    cur = conn.cursor()
    cur.execute("SELECT * FROM conversations")
    rows = cur.fetchall()
    conn.close()
    return {row[0]: {"title": row[1], "messages": row[2], "timestamp": row[3]} for row in rows}

