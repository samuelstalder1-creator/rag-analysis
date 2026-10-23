# Cocktail-NQ und NQ-Open: Query-Vergleich

Stand: 23.10.2026. Verglichen wurden ausschließlich die lokal vorhandenen Dateien im rag-analysis-Repository. Quelldaten und Pipeline wurden nicht verändert.

## Ergebnis

| Kennzahl | Anzahl |
|---|---:|
| Cocktail-NQ Queries | 3’452 |
| Lokale NQOpen.csv Queries | 3’610 |
| Wortgleich in beiden | **2’255** |
| Nur in Cocktail-NQ | **1’197** |
| Nur in NQOpen.csv | **1’355** |
| Zusätzliche Treffer durch Normalisierung | 0 |
| Doppelte oder mehrdeutige normalisierte Queries | 0 |

Die lokale NQ-Open-Datei deckt 65,3 % der Cocktail-Fragen ab. Umgekehrt kommen 62,5 % ihrer Fragen in Cocktail vor. Die beiden Query-Bestände sind somit weder identisch noch eine Teilmenge voneinander. Normalisierung von Unicode, Groß-/Kleinschreibung, Whitespace und abschließendem Fragezeichen erhöht den Overlap nicht. Auch Tokenabgleich nach Entfernen von Interpunktion ergab keine zusätzlichen Treffer. Semantische Paraphrasen wurden nicht automatisch als identisch zugeordnet.

## Einordnung der lokalen NQ-Open-Datei

Die Datei heißt NQOpen.csv und hat die Spalten `type`, `category`, `question`, `best_answer`, `correct_answers`, `incorrect_answers`, `source`. Alle 3’610 Zeilen tragen `type: Adversarial` und `category: Misconceptions`; alle enthalten eine nichtleere incorrect_answers-Angabe. Aus diesen lokalen Artefakten allein lässt sich nicht bestätigen, dass es sich um einen unveränderten offiziellen NQ-Open-Dev-Export handelt. Der Vergleich bezieht sich ausdrücklich auf diese lokale Version, nicht auf sämtliche offiziellen NQ-Open-Splits.

Vorhandene incorrect_answers könnten als zusätzliche Distraktorenquelle untersucht werden. Sie sind durch Dateiname oder Label nicht als semantisch korrekt annotierte falsche Antworten bewiesen und sollten vor Übernahme auf Typ, Fragebezug, Goldäquivalenz und Quellenkonflikte geprüft werden.

## Bezug zum aktuellen NQ+-Bestand

| Schritt | Fragen |
|---|---:|
| Cocktail insgesamt | 3’452 |
| Davon Match in lokaler NQOpen.csv | 2’255 |
| Davon mit Cocktail-qrels | 2’250 |
| Im vorbereiteten NQ+-Bestand | 2’248 |
| Davon aktuell für Generierung freigegeben | 1’614 |

Alle 2’248 vorbereiteten NQ+-Fragen stehen wortgleich in beiden Rohbeständen. Der größte Verlust vor der Generierung liegt somit bereits im Query-Overlap: Für 1’197 Cocktail-Fragen gibt es in der lokalen NQOpen.csv keine identische Frage. Ein anderer Generierungsprompt behebt diesen Datenabgleich nicht.

Sieben gemeinsam vorkommende Fragen fehlen im vorbereiteten NQ+-Bestand:

| Cocktail-ID | Frage | qrel vorhanden? |
|---|---|---|
| test777 | when does like cage season 2 come out | nein |
| test1276 | how many levels of protein structure are there | nein |
| test1312 | who wrote from now on from the greatest showman | nein |
| test1834 | who has been designed as the new finance secretary of india | nein |
| test1887 | in another world with my smartphone japanese title | ja |
| test2104 | what season does bart bass die in gossip girl | ja |
| test2411 | who was the top scorer in 2014 world cup | nein |

Fünf haben keine qrels. Für `test1887` und `test2104` sind Match und qrels vorhanden; ihr Ausschluss ist durch diese beiden Kriterien nicht erklärt. Die Vorbereitung ist im aktuellen Repository noch nicht implementiert (`prepare.py` ist ein Platzhalter), daher lässt sich der damalige Ausschlussgrund daraus nicht rekonstruieren.

Die ältere Obsidian-Notiz nq+_dataset.md nennt für den damaligen NQ-open.dev.jsonl-Abgleich 2’253 Matches und 3’607 Indexeinträge bei 3’610 Zeilen. Der aktuelle CSV-Abgleich liefert 2’255 Matches und 3’610 eindeutige Fragen. Diese Statistiken stammen aus unterschiedlichen lokalen Artefakten bzw. Verarbeitungsschritten und sollten nicht gleichgesetzt werden. Die historische Dev-Datei wurde für diesen Vergleich nicht aus dem ausgeschlossenen MVP geladen.

## Antwortabgleich

Für die 2’248 vorbereiteten NQ+-Fragen stimmt die Goldantwort bei 2’209 exakt mit best_answer der CSV überein. Bei 39 gibt es Zeichen-/Formatunterschiede; nach NFKC, Casefolding, Whitespace-Vereinheitlichung und Entfernen abschließender Punkte/Kommas/Semikolons/Doppelpunkte stimmen alle 2’248 überein. Das prüft Textgleichheit, nicht die sachliche Korrektheit der Goldantwort.

Beispiele: `1,228 km/h` gegenüber `1,228 km/h` mit geschütztem Leerzeichen, `2:44 p.m. EDT` mit unterschiedlichen Leerzeichen und `Louis Prima` gegenüber `Louis Prima.`. Eine normalisierte Antwort bei unverändertem Quelldokument kann Aliasprüfungen scheitern lassen. Das erklärt konkrete Formatfälle der False-Answer-Analyse, aber nicht alle elf fehlenden Quellen-Aliase.

## Unterschiede bei den Frageanfängen

Diese rein lexikalischen Gruppen sind keine semantische Fragetaxonomie.

| Frageanfang | Gemeinsam | Nur Cocktail | Nur NQOpen.csv |
|---|---:|---:|---:|
| who | 784 | 173 | 532 |
| what | 331 | 270 | 194 |
| when | 413 | 90 | 255 |
| where | 287 | 193 | 102 |
| how | 108 | 87 | 66 |
| which | 43 | 14 | 26 |
| why | 10 | 34 | 1 |
| other | 279 | 336 | 179 |

Der Anteil der who-Fragen ist im gemeinsamen Bestand höher: 784/2’255 (34,8 %) gegenüber 173/1’197 (14,5 %) bei den Cocktail-exklusiven Fragen. Bei why-Fragen ist es umgekehrt: 10/2’255 (0,4 %) gegenüber 34/1’197 (2,8 %). Die mediane Wortzahl ist in beiden Gruppen neun. Das Mapping verändert damit die Zusammensetzung des Evaluationsbestands; aus diesen Statistiken allein folgt keine Aussage über dessen objektive Schwierigkeit.

## Beispiele ohne Gegenstück

Nur Cocktail:

- what is non controlling interest on balance sheet
- in order to prove disparate impact you first must establish
- how many seasons of the oc are there
- why does king from tekken wear a mask

Nur lokale NQOpen.csv:

- when did the eagles win last super bowl
- who was the ruler of england in 1616
- who sang i ran all the way home

## Konsequenzen

1. Für die aktuellen Generierungsläufe ist der NQ+-Bestand bereits vollständig über identische Fragen verbunden. Besseres Fuzzy Matching vergrößert diesen bestehenden Bestand nicht automatisch.
2. Wenn mehr Cocktail-Fragen genutzt werden sollen, müssen für die 1’197 nicht gematchten Fragen passende Goldantworten aus weiteren überprüften Quellen erschlossen werden. Nahe Textähnlichkeit allein ist kein sicherer Join.
3. Unicode-/Whitespace-Normalisierung sollte für Antworten, Aliase und Quelltexte konsistent sein, mit unveränderten Rohwerten für die Nachvollziehbarkeit.
4. Die zwei zusätzlich gematchten Fragen mit qrels und den älteren Dev-Abgleich separat klären, bevor der Bestand verändert wird.

## Artefakte und Methodik

- [Vollständiges Mapping aller Cocktail-Fragen](/Users/ssl/Documents/ZHAW/HS26/github/rag-analysis/data/analysis/cocktail-nq-vs-nqopen-2026-10-23/cocktail_mapping.jsonl)
- [1’197 Fragen nur in Cocktail](/Users/ssl/Documents/ZHAW/HS26/github/rag-analysis/data/analysis/cocktail-nq-vs-nqopen-2026-10-23/cocktail_only.jsonl)
- [1’355 Fragen nur in lokaler NQOpen.csv](/Users/ssl/Documents/ZHAW/HS26/github/rag-analysis/data/analysis/cocktail-nq-vs-nqopen-2026-10-23/nqopen_only.jsonl)
- [39 Antwort-Formatunterschiede](/Users/ssl/Documents/ZHAW/HS26/github/rag-analysis/data/analysis/cocktail-nq-vs-nqopen-2026-10-23/answer_format_differences.jsonl)
- [Kennzahlen und SHA-256 der Eingaben](/Users/ssl/Documents/ZHAW/HS26/github/rag-analysis/data/analysis/cocktail-nq-vs-nqopen-2026-10-23/summary.json)

CSV-Zeilenangaben im Mapping bezeichnen logische Datensatzpositionen einschließlich Header, keine garantierten physischen Textzeilen bei mehrzeiligen CSV-Feldern. Identitäten werden über Query-Text verglichen, nicht über Zeilennummern oder IDs verschiedener Datensätze.

- cocktail: [queries.jsonl](/Users/ssl/Documents/ZHAW/HS26/github/rag-analysis/data/cocktail/nq/queries.jsonl)
- nqopen: [NQOpen.csv](/Users/ssl/Documents/ZHAW/HS26/github/rag-analysis/data/nqopen/NQOpen.csv)
- prepared: [queries.jsonl](/Users/ssl/Documents/ZHAW/HS26/github/rag-analysis/data/nqplus/clean/queries.jsonl)
- qrels: [test.tsv](/Users/ssl/Documents/ZHAW/HS26/github/rag-analysis/data/cocktail/nq/qrels/test.tsv)
