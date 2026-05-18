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

@app.route('/')
def home():
    return redirect('/login')

@app.route('/login', methods=['GET','POST'])
def login():
    if request.method == 'POST':
        user = request.form['username']
        pwd = request.form['password']

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
        user = request.form['username']
        pwd = request.form['password']

        conn = sqlite3.connect("users.db")
        conn.execute("INSERT INTO users VALUES (?,?)", (user,pwd))
        conn.commit()
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

@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect('/login')

if __name__ == "__main__":
    app.run(debug=True)