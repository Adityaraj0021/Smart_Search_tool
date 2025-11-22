import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
CSV_PATH = os.path.join(BASE_DIR, 'cleaned_analytics_vidhya_courses.csv')

REPLACEMENTS = {
    "Introduction to SQL[Do not Delete]": {
        "Description": (
            "A practical, beginner-friendly introduction to relational databases and SQL. "
            "Covers core SQL syntax, data types, joins, subqueries, aggregate and window functions, "
            "and hands-on database operations so learners can write typical queries for reporting and analysis."
        )
    },
    "Essentials of Excel": {
        "Description": (
            "A concise, practical course covering Excel fundamentals for analysts: interface, formulas, "
            "data cleaning with Power Query, lookup and aggregation functions, charts, conditional formatting, "
            "and automation using macros and AI features."
        )
    },
    "The Complete Power BI Blueprint": {
        "Description": (
            "A comprehensive Power BI roadmap: data modeling, DAX, visuals (bar, line, area, map, funnel, etc.), "
            "date tables and time intelligence, conditional formatting, and building interactive dashboards for business storytelling."
        )
    },
    "Case Study - Data Analysis using SQL": {
        "Description": (
            "An end-to-end case study applying SQL to real business problems: database design, hands-on SQL operations, "
            "aggregate and window functions, joins, subqueries, and building reusable artifacts like stored procedures and views."
        )
    },
    "DHS 2024 Sessions": {
        "Description": (
            "A curated set of talks and workshops from DHS 2024 focused on practical AI, RAG systems, LLM efficiency, "
            "multilingual GenAI strategies, autonomous AI agents, and real-world deployment best practices."
        )
    }
}


def backup(path: str):
    bak = path + '.bak'
    if not os.path.exists(bak):
        os.rename(path, bak)
        print(f"Backup created at {bak}")
    else:
        print(f"Backup already exists at {bak}; skipping rename")


def update_csv(path: str):
    df = pd.read_csv(path)
    updated = 0

    for title, fields in REPLACEMENTS.items():
        mask = df['Title'].astype(str).str.strip() == title
        if mask.any():
            for col, val in fields.items():
                df.loc[mask, col] = val
            # also update combined_text to include the new description
            df.loc[mask, 'combined_text'] = df.loc[mask, 'Description'].astype(str).str.lower()
            updated += int(mask.sum())
        else:
            print(f"Title not found in CSV: {title}")

    if updated:
        # write out updated CSV (overwrite original)
        df.to_csv(path, index=False)
        print(f"Updated {updated} rows in {path}")
    else:
        print("No rows updated.")


if __name__ == '__main__':
    if not os.path.exists(CSV_PATH):
        print(f"CSV not found: {CSV_PATH}")
        raise SystemExit(1)

    backup(CSV_PATH)
    # restore backup copy name to work on current file (we renamed original to bak)
    bak_path = CSV_PATH + '.bak'
    if os.path.exists(bak_path):
        # copy bak back to CSV_PATH for editing
        import shutil
        shutil.copy(bak_path, CSV_PATH)

    update_csv(CSV_PATH)
