# Security and privacy

HealthLab is a **local-only** tool. It reads an Apple Health export you provide, stores it in a SQLite file on
your machine, and serves a Streamlit app bound to `127.0.0.1`. It contains no credentials and makes no network
calls with your data.

## Keep your health data out of version control

- Never commit your `export.zip` / `export.xml`, the SQLite databases, or anything derived from them. The
  provided `.gitignore` excludes `data/`, `*.zip`, `*.xml`, `*.db`, `*.sqlite*` and `*.csv`. Keep your data under
  `data/`.
- Before pushing a fork, check that nothing sensitive is tracked:
  `git ls-files | grep -E '^data/|\.(db|sqlite3?|zip|xml)$'` should print nothing.
- Screenshots of the app on your own data can reveal personal information. The ones in this repo were taken in
  demo mode on synthetic data (look for the "DEMO MODE" badge).

## Reporting a vulnerability

Please open a GitHub issue for non-sensitive problems. For anything sensitive, use GitHub's private
vulnerability reporting on this repository (Security tab, "Report a vulnerability").
