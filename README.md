# Wardrobe Tool

Ein Werkzeug zur Verarbeitung und Zusammenfassung von PDF-Dokumenten basierend auf Benutzeranfragen.

## Beschreibung

Wardrobe Tool ist eine Anwendung, die es Benutzern ermöglicht, mehrere PDF-Dokumente hochzuladen und eine Frage zu stellen. Das Tool analysiert dann die Dokumente, ordnet sie nach Relevanz für die Frage und erstellt Zusammenfassungen der relevantesten Dokumente.

## Funktionen

- **PDF-Textextraktion**: Extrahiert Text aus hochgeladenen PDF-Dokumenten
- **Dokumenten-Ranking**: Sortiert Dokumente basierend auf ihrer Relevanz zur gestellten Frage
- **Intelligente Zusammenfassungen**: Generiert prägnante Zusammenfassungen der relevantesten Dokumente
- **Benutzerfreundliche Oberfläche**: Einfache Bedienung über eine Gradio-Webschnittstelle

## Technische Details

Die Anwendung verwendet folgende Bibliotheken und Modelle:

- **Transformers**: Für NLP-Modelle zur Textverarbeitung
- **BART (facebook/bart-large-cnn)**: Für die Generierung von Zusammenfassungen
- **DistilBERT (msmarco-distilbert-base-v3)**: Für Dokumenten-Ranking
- **PyMuPDF (fitz)**: Für die Extraktion von Text aus PDF-Dokumenten
- **Gradio**: Für die Benutzeroberfläche

## Installation

1. Stellen Sie sicher, dass Python 3.7 oder höher installiert ist
2. Installieren Sie die erforderlichen Pakete:

```bash
pip install transformers torch pymupdf gradio
```

## Verwendung

1. Führen Sie die Anwendung aus:

```bash
python wardrobe_tool.py
```

2. Öffnen Sie den angezeigten Link in einem Webbrowser
3. Geben Sie Ihre Frage in das Textfeld ein
4. Laden Sie Ihre PDF-Dokumente hoch
5. Klicken Sie auf "Submit", um die Verarbeitung zu starten

## Funktionsweise

1. **Textextraktion**: Der Text wird aus den hochgeladenen PDF-Dokumenten extrahiert
2. **Dokumentenranking**: Die Dokumente werden mithilfe eines BERT-basierten Modells nach Relevanz für die Frage sortiert
3. **Zusammenfassung**: Die relevantesten Dokumente werden mit BART zusammengefasst
4. **Anzeige der Ergebnisse**: Die sortierten Dokumente und Zusammenfassungen werden dem Benutzer präsentiert

## Einschränkungen

- Die Verarbeitung großer PDFs kann rechenintensiv sein
- Die maximale Eingabelänge für die Modelle beträgt 1024 Token
- Die Qualität der Zusammenfassungen hängt von der Qualität des extrahierten Textes ab
