from flask import Flask, render_template, request, redirect
from nltk.tokenize import word_tokenize
from nltk import pos_tag
import sqlite3

app = Flask(__name__)
DATABASE = "analysis_history.db"


# ============================================================
# DATABASE
# ============================================================

def get_db():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db()

    conn.execute("""
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            sentence TEXT NOT NULL,
            predicate TEXT,
            agent TEXT,
            theme TEXT,
            recipient TEXT,
            location TEXT,
            time TEXT
        )
    """)

    conn.commit()
    conn.close()


# ============================================================
# PREDICATE DETECTION
# ============================================================

def find_predicates(tagged):
    predicates = []

    for word, tag in tagged:
        if tag.startswith("VB"):
            if word not in predicates:
                predicates.append(word)

    return predicates


# ============================================================
# TIME DETECTION
# ============================================================

def find_time(tokens):
    lower_tokens = [word.lower() for word in tokens]

    time_phrases = [
        ("this", "morning"),
        ("this", "afternoon"),
        ("this", "evening"),
        ("this", "night"),

        ("yesterday", "morning"),
        ("yesterday", "afternoon"),
        ("yesterday", "evening"),
        ("yesterday", "night"),

        ("today", "morning"),
        ("today", "afternoon"),
        ("today", "evening"),
        ("today", "night"),

        ("last", "night"),
        ("last", "evening"),
        ("last", "week"),
        ("last", "month"),

        ("next", "week"),
        ("next", "month")
    ]

    for i in range(len(lower_tokens) - 1):

        pair = (
            lower_tokens[i],
            lower_tokens[i + 1]
        )

        if pair in time_phrases:
            return tokens[i] + " " + tokens[i + 1]

    single_time_words = [
        "today",
        "yesterday",
        "tomorrow",
        "tonight",
        "morning",
        "afternoon",
        "evening",
        "night",
        "now"
    ]

    for i, word in enumerate(lower_tokens):

        if word in single_time_words:
            return tokens[i]

    return "Not found"


# ============================================================
# LOCATION DETECTION
# ============================================================

def find_location(tokens, tagged, start_index=0, end_index=None):

    lower_tokens = [
        word.lower()
        for word in tokens
    ]

    if end_index is None:
        end_index = len(tokens)

    location_prepositions = [
        "in",
        "at",
        "near",
        "inside",
        "outside"
    ]

    ignore_words = [
        "the",
        "a",
        "an",
        "my",
        "his",
        "her",
        "their",
        "our",
        "your"
    ]

    time_words = [
        "today",
        "yesterday",
        "tomorrow",
        "tonight",
        "morning",
        "afternoon",
        "evening",
        "night",
        "now",
        "last",
        "next"
    ]

    for i in range(start_index, end_index):

        if lower_tokens[i] in location_prepositions:

            location_start = i + 1

            while (
                location_start < end_index
                and lower_tokens[location_start] in ignore_words
            ):
                location_start += 1

            if location_start >= end_index:
                continue

            location_words = []

            j = location_start

            while j < end_index:

                word = tokens[j]
                lower_word = lower_tokens[j]
                tag = tagged[j][1]

                if lower_word in time_words:
                    break

                if lower_word in ["and", "but"]:
                    break

                if lower_word in [
                    "to",
                    "for",
                    "in",
                    "at",
                    "near",
                    "with",
                    "from",
                    "on"
                ]:
                    break

                if tag.startswith("NN"):

                    location_words.append(word)
                    j += 1

                else:

                    if not location_words:
                        location_words.append(word)
                        j += 1
                    else:
                        break

            if location_words:
                return " ".join(location_words)

    return "Not found"


# ============================================================
# RECIPIENT DETECTION
# ============================================================

def find_recipient(tokens, tagged, start_index, end_index):

    lower_tokens = [
        word.lower()
        for word in tokens
    ]

    recipient_prepositions = [
        "to",
        "for"
    ]

    ignore_words = [
        "the",
        "a",
        "an",
        "my",
        "his",
        "her",
        "their",
        "our",
        "your"
    ]

    for i in range(start_index, end_index):

        if lower_tokens[i] in recipient_prepositions:

            recipient_start = i + 1

            while (
                recipient_start < end_index
                and lower_tokens[recipient_start] in ignore_words
            ):
                recipient_start += 1

            if recipient_start >= end_index:
                continue

            recipient_words = []

            j = recipient_start

            while j < end_index:

                word, tag = tagged[j]

                if tag.startswith("NN"):
                    recipient_words.append(word)
                    j += 1
                else:
                    break

            if recipient_words:
                return " ".join(recipient_words)

    return "Not found"


# ============================================================
# THEME DETECTION
# ============================================================

def find_theme(tokens, tagged, predicate_position, end_index):

    lower_tokens = [
        word.lower()
        for word in tokens
    ]

    ignored_words = [
        "the",
        "a",
        "an",
        "my",
        "his",
        "her",
        "their",
        "our",
        "your",
        "to",
        "for",
        "in",
        "at",
        "on",
        "near",
        "inside",
        "outside",
        "with",
        "and",
        "but"
    ]

    # Words that indicate another semantic role or boundary
    boundary_words = [
        "to",
        "for",
        "in",
        "at",
        "near",
        "inside",
        "outside",
        "with",
        "from",
        "on",
        "and",
        "but"
    ]

    # --------------------------------------------------------
    # Find the beginning of the theme
    # --------------------------------------------------------

    theme_start = None

    for i in range(
        predicate_position + 1,
        end_index
    ):

        word = lower_tokens[i]

        if word in boundary_words:
            continue

        if word in ignored_words:
            continue

        tag = tagged[i][1]

        if tag.startswith("NN"):

            theme_start = i
            break

    if theme_start is None:
        return "Not found"

    # --------------------------------------------------------
    # Collect the complete noun phrase
    # --------------------------------------------------------

    theme_words = []

    j = theme_start

    while j < end_index:

        word = tokens[j]
        lower_word = lower_tokens[j]
        tag = tagged[j][1]

        # Stop at semantic boundaries
        if lower_word in boundary_words:
            break

        # Skip punctuation
        if lower_word in [".", ",", "!", "?"]:
            break

        # Collect nouns
        if tag.startswith("NN"):

            theme_words.append(word)
            j += 1
            continue

        # Collect adjectives before/inside noun phrase
        if tag.startswith("JJ"):

            theme_words.append(word)
            j += 1
            continue

        # Possessive determiner inside noun phrase
        if tag in ["PRP$", "DT"] and theme_words:

            j += 1
            continue

        break

    if theme_words:
        return " ".join(theme_words)

    return "Not found"


# ============================================================
# AGENT DETECTION
# ============================================================

def find_agent(
    tokens,
    tagged,
    predicate_position,
    previous_predicate_position
):

    # Proper noun first
    for i in range(
        predicate_position - 1,
        previous_predicate_position,
        -1
    ):

        word, tag = tagged[i]

        if tag in ["NNP", "NNPS"]:
            return word

    # Normal noun
    for i in range(
        predicate_position - 1,
        previous_predicate_position,
        -1
    ):

        word, tag = tagged[i]

        if tag.startswith("NN"):
            return word

    return "Not found"


# ============================================================
# ANALYZE EACH PREDICATE
# ============================================================

def analyze_predicates(tokens, tagged, predicates):

    predicate_data = []

    detected_time = find_time(tokens)

    predicate_positions = []

    for predicate in predicates:

        for i, word in enumerate(tokens):

            if word == predicate:

                predicate_positions.append(i)
                break

    for predicate_index, predicate in enumerate(predicates):

        data = {
            "predicate": predicates[predicate_index],
            "agent": "Not found",
            "theme": "Not found",
            "recipient": "Not found",
            "location": "Not found",
            "time": detected_time
        }

        if predicate_index >= len(predicate_positions):

            predicate_data.append(data)
            continue

        predicate_position = predicate_positions[
            predicate_index
        ]

        if predicate_index == 0:

            previous_position = -1

        else:

            previous_position = predicate_positions[
                predicate_index - 1
            ]

        if predicate_index + 1 < len(predicate_positions):

            next_position = predicate_positions[
                predicate_index + 1
            ]

        else:

            next_position = len(tokens)

        # ----------------------------------------------------
        # AGENT
        # ----------------------------------------------------

        data["agent"] = find_agent(
            tokens,
            tagged,
            predicate_position,
            previous_position
        )

        # ----------------------------------------------------
        # THEME
        # ----------------------------------------------------

        data["theme"] = find_theme(
            tokens,
            tagged,
            predicate_position,
            next_position
        )

        # ----------------------------------------------------
        # RECIPIENT
        # ----------------------------------------------------

        data["recipient"] = find_recipient(
            tokens,
            tagged,
            predicate_position + 1,
            next_position
        )

        # ----------------------------------------------------
        # LOCATION
        # ----------------------------------------------------

        data["location"] = find_location(
            tokens,
            tagged,
            predicate_position + 1,
            next_position
        )

        # ----------------------------------------------------
        # TIME
        # ----------------------------------------------------

        data["time"] = detected_time

        predicate_data.append(data)

    return predicate_data


# ============================================================
# COMPLETE SENTENCE ANALYSIS
# ============================================================

def analyze_sentence(sentence):

    tokens = word_tokenize(sentence)

    tagged = pos_tag(tokens)

    predicates = find_predicates(tagged)

    predicate_data = analyze_predicates(
        tokens,
        tagged,
        predicates
    )

    if predicate_data:

        first = predicate_data[0]

    else:

        first = {
            "predicate": "Not found",
            "agent": "Not found",
            "theme": "Not found",
            "recipient": "Not found",
            "location": "Not found",
            "time": "Not found"
        }

    return {
        "tokens": tokens,
        "tagged": tagged,
        "pos_tags": tagged,
        "predicates": predicates,
        "predicate_data": predicate_data,
        "predicate": first["predicate"],
        "agent": first["agent"],
        "theme": first["theme"],
        "recipient": first["recipient"],
        "location": first["location"],
        "time": first["time"]
    }


# ============================================================
# HOME
# ============================================================

@app.route("/")
def home():

    return render_template(
        "index.html"
    )


# ============================================================
# ANALYZE
# ============================================================

@app.route(
    "/analyze",
    methods=["POST"]
)
def analyze():

    sentence = request.form.get(
        "sentence",
        ""
    ).strip()

    if not sentence:
        return redirect("/")

    result = analyze_sentence(sentence)

    if result["predicate_data"]:

        first = result["predicate_data"][0]

    else:

        first = {
            "predicate": "Not found",
            "agent": "Not found",
            "theme": "Not found",
            "recipient": "Not found",
            "location": "Not found",
            "time": "Not found"
        }

    conn = get_db()

    conn.execute(
        """
        INSERT INTO history
        (
            sentence,
            predicate,
            agent,
            theme,
            recipient,
            location,
            time
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            sentence,
            first.get("predicate", "Not found"),
            first.get("agent", "Not found"),
            first.get("theme", "Not found"),
            first.get("recipient", "Not found"),
            first.get("location", "Not found"),
            first.get("time", "Not found")
        )
    )

    conn.commit()
    conn.close()

    graph_roles = {
        "agent": first.get("agent", "Not found"),
        "theme": first.get("theme", "Not found"),
        "recipient": first.get("recipient", "Not found"),
        "location": first.get("location", "Not found"),
        "time": first.get("time", "Not found")
    }

    return render_template(
        "index.html",
        result=result,
        graph_roles=graph_roles,
        sentence=sentence,
        show_result=True
    )


# ============================================================
# HISTORY
# ============================================================

@app.route("/history")
def history():
    conn = get_db()
    rows = conn.execute("SELECT * FROM history ORDER BY id DESC").fetchall()
    predicates = sorted(set(row["predicate"] for row in rows if row["predicate"] and row["predicate"] != "Not found"))
    agents = sorted(set(row["agent"] for row in rows if row["agent"] and row["agent"] != "Not found"))
    locations = sorted(set(row["location"] for row in rows if row["location"] and row["location"] != "Not found"))
    conn.close()
    return render_template("index.html", history=rows, history_rows=rows, predicates=predicates, agents=agents, locations=locations, show_history=True)


# DELETE HISTORY
# ============================================================

@app.route(
    "/delete_history/<int:record_id>",
    methods=["POST"]
)
def delete_history(record_id):

    conn = get_db()

    conn.execute(
        """
        DELETE FROM history
        WHERE id = ?
        """,
        (record_id,)
    )

    conn.commit()
    conn.close()

    return redirect("/history")


# ============================================================
# DASHBOARD
# ============================================================

@app.route("/dashboard")
def dashboard():
    conn = get_db()
    total = conn.execute("SELECT COUNT(*) FROM history").fetchone()[0]
    unique_predicates = conn.execute("SELECT COUNT(DISTINCT predicate) FROM history WHERE predicate IS NOT NULL AND predicate != 'Not found'").fetchone()[0]
    unique_agents = conn.execute("SELECT COUNT(DISTINCT agent) FROM history WHERE agent IS NOT NULL AND agent != 'Not found'").fetchone()[0]
    locations_detected = conn.execute("SELECT COUNT(DISTINCT location) FROM history WHERE location IS NOT NULL AND location != 'Not found'").fetchone()[0]
    predicate_frequency = conn.execute("SELECT predicate, COUNT(*) AS count FROM history WHERE predicate IS NOT NULL AND predicate != 'Not found' GROUP BY predicate ORDER BY count DESC").fetchall()
    agent_frequency = conn.execute("SELECT agent, COUNT(*) AS count FROM history WHERE agent IS NOT NULL AND agent != 'Not found' GROUP BY agent ORDER BY count DESC").fetchall()
    recent = conn.execute("SELECT * FROM history ORDER BY id DESC LIMIT 10").fetchall()
    conn.close()
    return render_template("index.html", total=total, total_analyses=total, unique_predicates=unique_predicates, unique_agents=unique_agents, locations_detected=locations_detected, total_predicates=unique_predicates, total_agents=unique_agents, total_locations=locations_detected, predicate_frequency=predicate_frequency, agent_frequency=agent_frequency, recent=recent, show_dashboard=True)


# START APPLICATION
# ============================================================

if __name__ == "__main__":

    init_db()

    app.run(debug=True)