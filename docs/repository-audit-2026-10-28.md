# Repository-Audit: rag-analysis

Stand: 28.10.2026. Geprüfter Commit: `66014ad`.

## Einschätzung

Das Repository lässt sich schrittweise verbessern. Die Trennung zwischen Daten,
Generierung, Retrieval, Reader und Evaluation ist bereits brauchbar. Die größten
Risiken liegen im Lebenszyklus von Runs und Caches. Danach folgen historisch
parallele Konfigurationen, Launcher und nicht angebundene Funktionen.

Alle **284 Tests bestehen**. Alle **30 Experimentkonfigurationen** lassen sich
einschließlich ihrer Varianten hinsichtlich Retriever und Reader auflösen.
Das bestätigt weder die Verfügbarkeit sämtlicher Datenpfade noch die sachliche
Gültigkeit historischer Resultate. Es wurden keine Modellaufrufe durchgeführt.
Produktionscode, bestehende Ergebnisse und Caches wurden nicht geändert.

## 1. Hohe Priorität: reproduzierbare Korrektheitsprobleme

### 1.1 Dense-Prefix-Cache umgeht die Inhaltsprüfung

Fundstelle: `src/evlab/retrieval/dense.py:107`, `_load_cached_prefix` ab Zeile 123.

Der reguläre Embedding-Schlüssel enthält Text-Hashes, `normalize`, `doc_prefix`
und `max_seq_length`. Beim anschließenden Prefix-Reuse werden dagegen nur
Dokument-IDs verglichen. Der Scan durchsucht alle `.ids`-Dateien desselben
Modellverzeichnisses und prüft diese Parameter nicht erneut.

**Nachgestellt:** Ein Dummy-Encoder indiziert ein Dokument und danach dieselbe
ID mit verändertem Text. Beide Vektoren bleiben `[[11.0, 1.0]]`; der Encoder
wird insgesamt nur einmal aufgerufen. Die alten Vektoren werden sogar unter
dem neuen Inhaltsschlüssel gespeichert. Eine ähnlich falsche Wiederverwendung
ist beim Wechsel der genannten Embedding-Parameter möglich.

**Verbesserung:** Prefix-Metadaten müssen pro Dokument Inhalts-Hashes sowie die
vollständige Embedding-Konfiguration enthalten. Nur kompatible Prefixe nutzen;
sonst neu encodieren. Nach der Korrektur betroffene alte Cachebestände getrennt
behandeln und relevante Dense-Runs erneut prüfen. Es ist hiermit nicht bewiesen,
dass ein bestimmter gespeicherter Dense-Run tatsächlich betroffen war.

### 1.2 LLM-Cache ignoriert wirksame Anfrageparameter

Fundstelle: `src/evlab/llm/cache.py:22`.

Der Schlüssel enthält Provider, Endpoint, Modell, Prompt, Temperatur und Seed.
`max_tokens` fehlt; beim Ollama-Client fehlen auch `num_ctx` und `think`.

**Nachgestellt:** Ein Client mit `max_tokens=256` speichert `limit=256`. Ein zweiter
Client mit `max_tokens=8192`, aber demselben Cache, bekommt wieder `limit=256`.
Eine Budgeterhöhung kann somit unbemerkt dieselbe alte Antwort liefern.

**Verbesserung:** Jeder Client liefert eine kanonische, nicht geheime
Anfragekonfiguration für den Cache-Schlüssel. Cacheformat versionieren.

### 1.3 Abgebrochene Wiederholung kann alte Abschlussartefakte stehen lassen

Fundstellen: `src/evlab/experiments/runner.py:42`, `:130`, `:141`, `:235`.

Der Runner verwendet einen vorhandenen Run-Ordner wieder. Die Antwortdatei wird
zu Beginn geleert, vorhandene `summary.json` und `metrics.json` aber nicht
invalidiert. Ein späterer Fehler erzeugt einen widersprüchlichen Ordner.

**Nachgestellt:** Ein erfolgreicher Fixture-Run mit zwei Fragen wird wiederholt;
der Reader schlägt bei Frage zwei fehl. Danach stehen eine alte Summary für zwei
Fragen und alte Metriken neben einer Antwortdatei mit nur einer Zeile.

**Verbesserung:** Expliziter Runstatus (`running`, `complete`, `failed`),
Abschlussartefakte atomar veröffentlichen, vorhandene abgeschlossene Ergebnisse
unveränderlich halten. Echte Wiederaufnahme nach Query-ID statt Leeren der Datei.
Der SQLite-Cache spart momentan einige erneute Modellaufrufe, ersetzt aber keinen
Run-Checkpoint; Retrieval und Antwortschleife beginnen erneut.

### 1.4 Run-ID und Corpus-Manifest identifizieren keine Dateninhalte

Fundstellen: `src/evlab/experiments/runner.py:190`,
`src/evlab/corpus/conditions.py:7`.

Die Run-ID hängt von Konfiguration und teilweise aufgelösten Presets ab. Sie
enthält keine Fingerprints der Dataset-, Corpus- und Qrel-Inhalte und keine
Codeversion. `doc_set_hash` berücksichtigt nur IDs.

**Nachgestellt:** Zwei Dokumentlisten mit denselben IDs, aber verschiedenem Text,
erzeugen identische Corpus-Manifeste bei identischer Konfiguration.

**Verbesserung:** Eingaben, aufgelöste Konfiguration, Evaluatorversion und Git-SHA
im Runmanifest erfassen. Eingabe-Hashes müssen entweder in die Runidentität
eingehen oder als unveränderliche Dataset-/Corpus-Version referenziert werden.
Die neueren `input_audit.json`-Dateien sind ein guter Ansatz, gehören aber in den
allgemeinen Runner statt nur in einzelne Evaluationsabläufe.

### 1.5 `generation-status` scheitert bei aktuellen importierten False-Targets

Fundstellen: `src/evlab/generation/maintenance.py:196`,
`src/evlab/generation/false_answers.py:78` und `:247`.

Der Statuspfad lädt Kandidaten mit `include_text=False` und ohne Human-Corpus.
Beim Anhängen einfacher importierter `answers.jsonl`-Targets wird ein Receipt
erzeugt, das `aliases_in_doc` voraussetzt. Diese Quellenprüfung kann so nicht
funktionieren.

**Nachgestellt:** `generation_status` mit
`configs/generation/nqopen1520-deepseek-r1-false.yaml` bricht mit
`ModelOutputError: gold_in_sources: False` ab.

**Verbesserung:** Statusinventar und erneute Validierung trennen oder für diese
Prüfung tatsächlich die benötigten Quellen laden. Ein Statusbefehl sollte
fehlende/stale Daten als Befund ausgeben können.

### 1.6 Recall-Cutoff und Reader-Kontext haben verschiedene Bedeutungen

Fundstellen: `src/evlab/evaluation/retrieval.py:53` und `:88`.

Relaxed Recall entfernt doppelte Roots vor Anwendung von `k`. Der Reader nutzt
dagegen die ersten `k` rohen Treffer. Damit kann ein Dokument außerhalb des
Reader-Kontexts in `recall@k` eingehen.

**Nachgestellt:** Rank 1 und 2 sind zwei Versionen eines irrelevanten Roots;
erst Rank 3 ist relevant. Trotzdem beträgt relaxed `recall@2` genau 1.0.

Dies kann eine beabsichtigte Metrik für die ersten k unterschiedlichen Quellen
sein. Sie sollte jedoch so benannt und zusätzlich durch Kontext-Recall auf den
tatsächlichen ersten k Treffern ergänzt werden. Historische Metriken nicht ohne
Versionswechsel überschreiben.

## 2. Unvollständige Runs und gemischte Ergebnisarten

Inventar: **70 Run-Ordner mit `config.json`**, davon **66 mit Summary**.
Über die vorhandenen Retriever-Ausgaben wurden 99 strukturell abgeschlossene
Einträge, drei unvollständige Reader-Ausgaben und ein Ordner ohne Retrieval
gefunden. Bei der Prüfung wurde berücksichtigt, dass ein Reader absichtlich nur
für einen Teil der Retriever aktiviert sein kann: fehlende TF-IDF-Antworten in
mehreren alten RAG-Runs sind deshalb kein Fehler.

| Run | Lokaler Stand | Bewertung |
|---|---|---|
| `deepseek_bm25_20261027/...human_only_eca423a7ce` | 126/1,520 Antworten; keine Summary | Unvollständiger Export/Run |
| `qwen3_4b_bm25_20261027/...human_only_00e26c8850` | 750/1,520 Antworten; keine Summary | Unvollständiger Export/Run |
| `results/runs/rag_false_correct__human_only_c1bbdc7148` | Retrieval für 2,172 Fragen; keine Antworten/Summary | Früher Teilrun |
| `results/runs/rq1_dense_human_e5_5f363fd1ff` | Nur Konfiguration und Manifest | Vorbereitung oder abgebrochener Start |

Der ausgewählte Qwen-Sweep ist vollständig: **17/17 Varianten à 1,520 Antworten**.
Aus dem lokalen Dateistand lässt sich nicht erkennen, ob ein unvollständiger
Export auf einem anderen Rechner noch weiterläuft.

Zusätzlich gibt es **24 Retriever-Ausgaben aus 16 absichtlich limitierten Runs**:
17 Ausgaben für fünf Fragen, sieben für zwei Fragen. Die neueren Smoke-Tests sind
in eigenen Ordnern; ältere Zwei-/Fünf-Fragen-Runs stehen mitten in `results/runs`.
`evlab report` sammelt dort nur `metrics.json` und liefert keine Kennzeichnung
von Samplegröße, Smoke-Status und Runabschluss (`src/evlab/cli.py:185`).

Drei ältere Sweeps haben nur Corpus-Previews und keine Auswertung:
`variants_false_correct_86f187c2e6`, `sweep_correct_ratio_7566b7e796` und
`base1520_bm25_tfidf_1103fc18fa`. Beim letzten liegt daneben eine separat benannte
Summary-Datei; eine Registry sollte die Zusammengehörigkeit festhalten.
DeepSeek hat einen Preview für 32 Varianten, aber noch keine Sweep-Summary.
Previews sind Planungsergebnisse und keine gescheiterten Runs.

### Generierungsstände

Die DeepSeek-Correct-Datei enthält **960 Dokumente bei 1,670 erwarteten
Quelldokumenten**; die Registry meldet noch **952**. DeepSeek-False enthält sechs
Dokumente. Der Snapshot ist damit ebenfalls unvollständig und teilweise inkonsistent.
Die anderen sechs hier vorhandenen Registry-Corpusdateien stimmen bei der
Zeilenzahl mit ihrer Registry überein. Unterschiedliche historische Datensätze
haben unterschiedliche Zielmengen; nicht jede Differenz zur heutigen Basis ist
ein Fehler.

**Verbesserung:** Zentrales Runinventar mit `run_id`, `status`, `purpose`,
`expected_queries`, `completed_queries`, Datenfingerprints und Artefaktpfaden.
Smoke-, Preview-, Full- und Archive-Ausgaben explizit markieren. Erst danach
alte Ordner archivieren oder entfernen.

## 3. Duplikation und unnötige Parallelstrukturen

### Launcher

Die beiden Qwen-Evaluationsskripte sind zu **93.1 % zeilenweise ähnlich**.
DeepSeek und jedes Qwen-Evaluationsskript erreichen **80.3 %**. Gemeinsam sind
Server-Readiness, Prozessverwaltung, Timeout, Smoke-Aufruf, Sweep und COMPLETE.
Dies sind Sequenzähnlichkeiten, keine Prozentwerte nachgewiesener identischer
Programmlogik.

Betroffene Dateien:

- `scripts/run_qwen3_4b_evaluation.sh`
- `scripts/run_qwen3_4b_selected_evaluation.sh`
- `scripts/run_deepseek_evaluation.sh`
- Teile von `scripts/run_overnight_generation.sh`

**Verbesserung:** Ein konfigurierbarer Evaluationslauncher mit Modell,
Experimentkonfiguration, Serverlauncher, Ressourcen und Ausgabeordner als
Parametern. Dünne Kompatibilitätswrapper können vorhandene systemd-Units stabil
halten. Modellspezifische Tokenizer-Reparaturen getrennt lassen.

### Generierung und Konfiguration

- `generation/pipeline.py` unterstützt Correct-Generierung über `evlab generate`;
  daneben gibt es die spezifische Pipeline in `generation/correct_repair.py`
  über `generate-correct-documents`. Die Verfahren sind nicht identisch und dürfen
  nicht blind zusammengelegt werden. Ein gemeinsamer Jobablauf für Client,
  Checkpoint, Retries, Output und Status wäre sinnvoll; fachliche Strategien
  bleiben getrennt.
- `REPAIR_PROMPT` und `CORRECT_PROMPT` in `correct_repair.py:27` bzw. `:47`
  wiederholen große Teile der Anweisungen. Bei Zusammenführung müssen Signaturen
  versioniert werden, damit alte Generationen identifizierbar bleiben.
- Die 30 Experimentkonfigurationen verteilen sich auf **13 Legacy-, zehn Mix-
  und sieben Fixed-Konfigurationen**. Diese drei Corpuspfade sind weiterhin
  aktiv verwendbar, also keine pauschal löschbaren Codeleichen. Historische
  Konfigurationen archivieren und neue Experimente auf klar dokumentierte
  Schemata begrenzen. Fixed und Ratio-Mix erfüllen unterschiedliche Zwecke.
- Die beiden Analyseskripte für base1520 und den neuen Sweep enthalten überlappende
  Logik für Laden, Coverage, Labels und Paarvergleiche. Ein gemeinsames
  Analysemodul würde verhindern, dass Qualitätsprüfungen auseinanderlaufen.

Eine AST-Prüfung fand **keine exakt identischen Funktionskörper ab fünf Zeilen**
innerhalb der Python-Dateien von `src` und `scripts`. Die Hauptduplikation liegt
bei Abläufen, Shellskripten, Prompts und Konfigurationen.

## 4. Codeleichen und nicht angeschlossene Funktionen

Diese Definitionen haben im gescannten Python-Code einschließlich Tests keine
Aufrufer/Importverwendungen außerhalb ihrer Definition. Externe Nutzer oder
interaktive Aufrufe wurden damit nicht ausgeschlossen.

| Fundstelle | Befund | Sinnvoller Umgang |
|---|---|---|
| `mitigation/mmr.py:6` | Expliziter Platzhalter; schneidet nur `hits[:top_k]` ab | Entfernen oder erst mit echter Implementierung aktivieren |
| `mitigation/dedup.py:6` | Text-Dedup implementiert, aber nicht angebunden | Als optionale Kontextstufe anbinden und testen |
| `mitigation/provenance.py:6` | Ein Treffer pro Root implementiert, aber nicht angebunden | Für Mitigationsexperimente anbinden |
| `data/prepare.py:4` | Nicht aufgerufener `NotImplementedError`-Stub | Entfernen; echte Datenaufbereitung existiert in `nqopen_plus.py` |
| `evaluation/citations.py:7` | Unverwendete Prüfung auf Antwortzeichenfolge | Entfernen oder klar als lexikalische Prüfung einbauen; kein Entailment-Nachweis |
| `schema.py:75` | `ReaderAnswer` ungenutzt; echte Antworten sind Dictionaries | Typmodell vereinheitlichen; `invalid`-Label und Integer-Citations berücksichtigen |
| `generation/minimal_edit.py:183` | Unbenutzter Kompatibilitätswrapper; Client wird ignoriert | Nach Prüfung externer Aufrufe entfernen |
| `io.py:12` | `resolve_path` ungenutzt | Entfernen oder Pfadauflösung zentral verwenden |
| `corpus/mix.py:371` | `clear_source_cache` wird intern nie aufgerufen | Cache-Lebenszyklus klären; nicht vorschnell löschen |

Der Source-Cache hält geladene Dateien nach Pfad ohne Inhalts-/mtime-Prüfung.
Bei Änderungen während desselben Python-Prozesses können alte Daten weiterleben.
Dieser Helfer ist deshalb eher eine fehlende Lifecycle-Anbindung als nur Ballast.

Außerdem sind `Sequence` und `Doc` in `experiments/runner.py` ungenutzte Imports.
Das ist ein kleiner, sicher behebbarer Stilbefund.

## 5. Artefakte, Abhängigkeiten und Dokumentation

- `results`: ca. **1.1 GB**, davon 646 Dateien von Git erfasst. `.git`: ca. 343 MB.
  Für den Forschungsnachweis kleine versionierte Manifeste, Konfigurationen und
  Tabellen behalten; große Rankings/Antwortexports bei Bedarf in einem separaten
  Artefaktspeicher versionieren. Alte Ergebnisse nicht ohne Provenienznachweis löschen.
- Es gibt sieben Gruppen byte-identischer `retrieval.jsonl`-Dateien mit insgesamt
  **14 zusätzlichen Kopien**. Das ist teils sinnvoll, weil verschiedene Reader
  dieselben Rankings nutzen. Gemeinsame unveränderliche Retrieval-Artefakte mit
  Verweisen würden Wiederholungen sparen. Gleiche Rankings beweisen nicht, dass
  zwei vollständige Experimente redundant sind.
- `vllm` ist die einzige verpflichtende Runtime-Abhängigkeit (`pyproject.toml:7`).
  Damit zieht auch ein reiner Analyse-/CPU-Workflow die Serving-Abhängigkeit nach.
  Ein optionales Extra `serve` und getrennte CPU-/Dense-/Server-Installationen
  würden die lokale Einrichtung vereinfachen. `pytest` ist sowohl als Extra als
  auch als Dependency-Group definiert; einen dokumentierten Dev-Weg wählen.
- Keine CI-Konfiguration und keine Lint-/Typcheck-Konfiguration im Repo gefunden.
  Eine kleine CPU-CI mit Tests, Linting und Konfigurationsprüfung deckt viel ab,
  ohne GPU-Server oder externe API-Zugriffe zu benötigen.
- `docs/blueprint.md` beschreibt Semantic Validation noch als nicht angebunden und
  RAG-Grids als zukünftige Arbeit. Beides ist vom aktuellen Code überholt.
  README und `runit.md` enthalten teils parallele Betriebsanweisungen. Eine
  aktuelle Übersicht und separate historische Notizen reduzieren Verwechslungen.
- `main.py` und der Wrapper `run_deepseek_overnight.sh` sind kleine alternative
  Einstiegspunkte. Sie sind kein nennenswerter Wartungsaufwand und nicht die
  erste Aufräumpriorität.

## Empfohlene Reihenfolge

1. **Ergebnisintegrität sichern:** Dense-Prefix-Cache, LLM-Cache,
   atomare Runabschlüsse, Checkpoint-Wiederaufnahme und Input-Fingerprints.
2. **Status zuverlässig machen:** Generation-Status reparieren, Runinventar
   einführen und Teilruns/Smoke/Previews kennzeichnen.
3. **Forschungsmetriken versionieren:** Kontext-Recall ergänzen und zwischen
   syntaktisch gültigen und faktisch gestützten Zitaten unterscheiden.
4. **Wartung vereinfachen:** Launcher zusammenführen, Legacy-Konfigurationen
   beschriften, unbenutzte Stubs entfernen und geplante Mitigations anbinden.
5. **Entwicklungsbasis festigen:** optionale Serving-Abhängigkeit, CPU-CI,
   aktuelle Dokumentation und klare Artefaktablage.

Ein kompletter Rewrite ist dafür nicht erforderlich. Kleine getrennte Änderungen
mit gezielten Regressionstests sind für die Reproduzierbarkeit besser prüfbar.

## Prüfumfang und Artefakte

Geprüft wurden `src`, `scripts`, `tests`, Konfigurationen, Runartefakte,
Generierungsregistries und die zentrale Dokumentation. Im Notebookordner wurden
keine Dateien gefunden. Die Runprüfung prüft vorhandene Dateien, JSONL-Parsebarkeit,
Abschlussdateien, Queryanzahlen und konfigurierte Reader; sie zertifiziert nicht
die faktische Qualität sämtlicher historischer Modellantworten.

Das maschinenlesbare Inventar liegt unter
`results/analysis/repository_audit_20261028/run_inventory.json`.
Reproduktionsproben wurden mit Fixtures und Dummy-Clients in temporären Ordnern
ausgeführt. Es wurden keine bestehenden Artefakte gelöscht oder repariert.

## Umsetzungsstand (Branch `refactor/run-integrity`)

| Befund | Stand |
|---|---|
| 1.1 Dense-Prefix-Cache | Behoben: Prefix-Reuse nur mit Metadaten (Einstellungen + Inhalts-Hash je Dokument); alte Cacheeinträge ohne Metadaten werden nie als Prefix genutzt. |
| 1.2 LLM-Cache | Behoben: `request_config()` je Client (`max_tokens`, bei Ollama `num_ctx`, `think`) im Schlüssel; Clients ohne Parameter behalten ihre Einträge. Bestehende Reader-Caches mit gesetztem `max_tokens` treffen nicht mehr. |
| 1.3 Abbruch bei Wiederholung | Behoben: `run_status.json`, `summary.json`/`metrics.json` werden beim Start entfernt, atomares `write_json`, Wiederaufnahme nach Query-ID. |
| 1.4 Run-ID ohne Dateninhalt | Eingabe-Fingerprints im Manifest; Run-IDs bleiben config-basiert, geänderte Daten im selben Ordner lösen `RunConflictError` aus. |
| 1.5 `generation-status` | Behoben (`with_receipts=False`). |
| 1.6 Recall-Cutoff | `context_recall@k` ergänzt, `recall@k` unverändert; `EVALUATOR_VERSION = 2`. |
| 2 Runinventar | `evlab inventory`, `evlab report` blendet Smoke-Runs aus. |
| 3 Launcher | `scripts/run_evaluation.sh` + Profile; alte Skripte sind Wrapper. |
| 4 Codeleichen | Entfernt: `mitigation/` (mmr, dedup, provenance), `data/prepare.py`, `evaluation/citations.py`, `ReaderAnswer`, `resolve_path`, `minimal_edit_with_fallback`, ungenutzte Imports. `clear_source_cache` bleibt. |
| 5 Basis | `vllm` nur noch Extra `serve`, PyYAML Pflicht, ruff + CPU-CI, Test über alle Experimentkonfigurationen, Blueprint/README/runit aktualisiert. |

Nicht umgesetzt: gemeinsamer Job-Ablauf der beiden Correct-Generatoren, Zusammenlegung der
Analyseskripte, Referenzierung identischer Retrieval-Dateien, Archivierung alter Konfigurationen.
