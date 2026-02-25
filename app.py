from flask import *
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

data = pd.read_csv("conv.csv")

app = Flask(__name__)
app.secret_key = "nanomind_secret_key"   # Required for session

@app.route("/", methods=["POST", "GET"])
def home():

    if "chat" not in session:
        session["chat"] = ""

    if request.method == "POST":
        qts = request.form["qts"].strip().lower()

        texts = [qts] + data["question"].str.lower().tolist()

        cv = CountVectorizer()
        vector = cv.fit_transform(texts)
        cs = cosine_similarity(vector)

        score = cs[0][1:]
        data["score"] = score * 100

        result = data.sort_values(by="score", ascending=False)
        result = result[result.score > 10]

        if len(result) == 0:
            ans = "Sorry, I don't know that yet."
        else:
            ans = result.head(1)["answer"].values[0]

        # Add to session chat history
        session["chat"] += f'<div class="message user">{qts}</div>'
        session["chat"] += f'<div class="message ai">{ans}</div>'

        return render_template("home.html", chat=session["chat"])

    return render_template("home.html", chat=session["chat"])


@app.route("/clear")
def clear():
    session["chat"] = ""
    return redirect("/")


if __name__ == "__main__":
    app.run(debug=True)