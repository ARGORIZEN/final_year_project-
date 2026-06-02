from flask import Flask, render_template, request, redirect, session, jsonify
import sqlite3
from model import predict_stock, get_supported_stocks

app = Flask(__name__)
app.secret_key = "secret"

# Database setup
def init_db():
    conn = sqlite3.connect("users.db")
    conn.execute("CREATE TABLE IF NOT EXISTS users (username TEXT, password TEXT)")
    conn.close()

init_db()

# ─── Start background model retrainer ─────────────────────────────────
# This runs ONCE when the app starts. It spawns a daemon thread that
# periodically checks for stale models and retrains them automatically.
def start_retraining_scheduler():
    """Initialize the background retraining scheduler."""
    try:
        from retrain_scheduler import start_scheduler
        from model import ALL_500_STOCKS_TICKERS
        start_scheduler(ALL_500_STOCKS_TICKERS)
    except Exception as e:
        print(f"[WARNING] Could not start retrain scheduler: {e}")

# Only start scheduler when running the app directly (not during imports)
import os
if os.environ.get("WERKZEUG_RUN_MAIN") == "true" or not app.debug:
    start_retraining_scheduler()

# ─── Routes ───────────────────────────────────────────────────────────

@app.route('/')
def home():
    return redirect('/login')

@app.route('/login', methods=['GET','POST'])
def login():
    if request.method == 'POST':
        user = request.form['username'].strip().lower()
        pwd = request.form['password'].strip()

        conn = sqlite3.connect("users.db")
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE username=? AND password=?", (user,pwd))
        data = cursor.fetchone()

        if data:
            session['user'] = user
            return redirect('/dashboard')
        else:
            return render_template("login.html", error="Invalid username or password")

    return render_template("login.html")

@app.route('/register', methods=['GET','POST'])
def register():
    if request.method == 'POST':
        user = request.form['username'].strip().lower()
        pwd = request.form['password'].strip()

        if not user or not pwd:
            return render_template("Register.html", error="Username and password cannot be empty")

        conn = sqlite3.connect("users.db")
        cursor = conn.cursor()
        
        # Check if username already exists
        cursor.execute("SELECT * FROM users WHERE username=?", (user,))
        if cursor.fetchone():
            conn.close()
            return render_template("Register.html", error="Username already exists")

        conn.execute("INSERT INTO users VALUES (?,?)", (user,pwd))
        conn.commit()
        conn.close()
        return redirect('/login')

    return render_template("Register.html")

@app.route('/api/stocks')
def api_stocks():
    """Return the list of supported NIFTY 50 stocks as JSON."""
    return jsonify(get_supported_stocks())

@app.route('/dashboard')
def dashboard():
    if 'user' not in session:
        return redirect('/login')
    stocks = get_supported_stocks()
    return render_template("dashboard.html", stocks=stocks)

@app.route('/result', methods=['POST'])
def result():
    if 'user' not in session:
        return redirect('/login')

    stock = request.form.get('stock', '')
    stocks = get_supported_stocks()
    prediction = None

    if stock:
        res = predict_stock(stock)
        if res is None:
            prediction = "Error: Unable to fetch data for this stock. Please try again."
        elif isinstance(res, dict) and res.get("error") == "unsupported":
            prediction = f"Error: No trained model for {res['ticker']}. Please select a stock from the NIFTY 50 list."
        else:
            prediction = res

    if prediction is None:
        return redirect('/dashboard')

    return render_template("result.html", prediction=prediction, stocks=stocks)

# ─── Retrain endpoints ────────────────────────────────────────────────

@app.route('/retrain', methods=['POST'])
def retrain():
    """Manually trigger retraining for a specific stock."""
    if 'user' not in session:
        return redirect('/login')

    ticker = request.form.get('ticker', '').strip().upper()
    if not ticker:
        return jsonify({"error": "No ticker provided"}), 400

    if not ticker.endswith(".NS") and "." not in ticker:
        ticker = ticker + ".NS"

    # Run retrain in a background thread so we don't block the response
    import threading
    from retrain_scheduler import retrain_single_model

    def _do_retrain():
        retrain_single_model(ticker)

    thread = threading.Thread(target=_do_retrain, daemon=True)
    thread.start()

    return jsonify({
        "status": "started",
        "ticker": ticker,
        "message": f"Retraining {ticker} in background. Refresh results in a few minutes."
    })

@app.route('/retrain/status')
def retrain_status():
    """Check the current status of the retraining scheduler."""
    if 'user' not in session:
        return redirect('/login')

    from retrain_scheduler import get_scheduler_status
    from model_metadata import get_all_metadata_summary

    status = get_scheduler_status()
    summary = get_all_metadata_summary()

    return jsonify({
        "scheduler": status,
        "models": summary,
    })

@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect('/login')

if __name__ == "__main__":
    app.run(debug=True)