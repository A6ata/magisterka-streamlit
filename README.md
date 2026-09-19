# Aplikacja Streamlit do pracy magisterskiej

Aplikacja pełni dwie funkcje:

- **Tryb Obrona** — liniowa prezentacja najważniejszych elementów pracy.
- **Tryb Eksplorator** — interaktywne badanie funkcji asymilacji, punktów stacjonarnych, analizy wrażliwości, dynamiki przejściowej, wielopunktowości i testu odejścia od stabilnej ścieżki.

## Uruchomienie lokalne

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

## Publikacja w Streamlit Community Cloud

1. Umieść cały folder w repozytorium GitHub.
2. W Streamlit Community Cloud wybierz repozytorium i plik startowy `app.py`.
3. Aplikacja korzysta tylko z lokalnych plików projektu — nie wymaga kluczy API.

## Struktura

- `app.py` — interfejs aplikacji.
- `model.py` — model i solvery z obliczeń pracy.
- `dynamics.py` — stabilne ścieżki i integracja dynamiczna.
- `analysis_tools.py` — funkcje łączące obliczenia z interfejsem.
- `data/` — wyniki preobliczone dokładnie na podstawie skryptów użytych w pracy.

## Ważna uwaga interpretacyjna

Mapa wielopunktowości jest eksperymentem eksploracyjnym dla szerokiego zakresu parametrów i nie stanowi kalibracji empirycznej. W zakładkach bazowych zachowano parametryzację i sposób porównania modeli zastosowane w pracy.
