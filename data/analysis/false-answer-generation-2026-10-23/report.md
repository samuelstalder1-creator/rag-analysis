# Analyse der False-Answer-Generierung

Stand: 23.10.2026. Auswertung der lokal vorhandenen Artefakte; keine neuen Modellaufrufe. Die Fragen werden nicht neu generiert: Zu bestehenden NQ+-Fragen werden falsche Zielantworten erzeugt.

## Bestand und Abdeckung

| Bestand | Anzahl |
|---|---:|
| Vorbereitete NQ+-Fragen | 2,248 |
| Für Reader freigegeben | 2,172 |
| Für Generierung freigegeben | 1,614 |
| Von Generierung ausgeschlossen | 634 |
| Gespeicherte falsche Antworten | 1,572 / 1,614 (97.4 %) |
| Ohne gespeicherte falsche Antwort | 42 |

Gespeichert bedeutet: vom aktuellen regelbasierten Ablauf akzeptiert. Alle 1,572 gespeicherten Antworten tragen `semantic_verified: false`. Das ist keine gemessene semantische Erfolgsquote. Der Frage-/Gold-/Alias-Hash passt für alle gespeicherten Antworten zum aktuellen Fragenbestand.

Die 634 ausgeschlossenen Fragen umfassen nach effektivem `answer_kind`: 505 unresolved, 72 list, 3 conflict, 1 list_with_conflict sowie 42 single und 11 variants. Die letzten Gruppen können z. B. wegen Antwort bereits in Frage oder fehlender Antwortstelle ausgeschlossen sein. Flags überlappen; sie dürfen nicht addiert werden. Für die Freigabe sind die mit `generated_labels.jsonl` zusammengeführten Metadaten maßgeblich, nicht allein die ursprüngliche queries.jsonl.

## Welche Fragetypen sind schwierig?

Die Typen stammen aus `answer_type()` und sind heuristische Antwortkategorien. Insbesondere enthält Freitext auch schlecht erkannte Zeit- und Mengenangaben. Sie sind keine manuell geprüfte Taxonomie.

| Antworttyp | Fragen | Gespeichert | Erst beim 2./3. Aufruf | Offen |
|---|---:|---:|---:|---:|
| Zahl | 47 | 47 | 2 | 0 |
| Person / Organisation | 558 | 555 | 117 | 3 |
| Entität / Freitext | 597 | 563 | 74 | 34 |
| Ort | 194 | 194 | 9 | 0 |
| Datum | 109 | 105 | 9 | 4 |
| Jahr | 107 | 107 | 0 | 0 |
| Messgröße | 2 | 1 | 0 | 1 |

1,361 Fragen wurden beim ersten Aufruf akzeptiert, 157 beim zweiten und 54 beim dritten. Damit brauchten 211 erfolgreiche Fälle zusätzliche Aufrufe; 42 blieben nach drei Versuchen offen. Das misst Schwierigkeit für diese Pipeline und diesen Lauf, nicht objektive Schwierigkeit einer Frage.

34 der 42 offenen Fälle liegen im heuristischen Freitexttyp. Personenfragen verursachen ebenfalls viele Retries: 117 von 558 wurden erst später akzeptiert. Reine Jahre und Zahlen funktionieren strukturell am zuverlässigsten; aus zwei Messgrößenfällen lässt sich keine belastbare Rate ableiten.

Alle gespeicherten Antworten wurden laut Receipt mit fünf angeforderten Kandidaten pro Aufruf erzeugt und nach `first_valid_in_model_order` ausgewählt. Ein erster erfolgreicher Aufruf kann also mehrere verworfene Vorschläge enthalten. Es ist keine unabhängige Qualitätsrangfolge.

## Vier Problemgruppen der 42 offenen Fälle

Für diese Übersicht hat ein fehlender Gold-Alias in der Quelle Vorrang; die übrigen Fälle werden nach häufigstem protokolliertem Vorschlagsstatus eingeordnet. Die Gruppen sind dadurch disjunkt, aber keine vollständige semantische Ursachenanalyse.

| Gruppe | Fragen | Bedeutung |
|---|---:|---|
| Frage kopiert | 4 | Modell verfehlt die Ausgabeaufgabe. |
| Typ / Format / Granularität | 24 | Mischung aus zu groben Prüfregeln und tatsächlich unpassenden Vorschlägen. |
| Gold-Alias fehlt in Quelle | 11 | Quellenabgleich statt Distraktorenqualität blockiert. |
| Gold-/Namensvarianten | 3 | Modell bleibt bei der richtigen Antwort. |

Zwei konkrete Quellenprobleme sind reine Zeichenformatierung: `test1317` enthält ein geschütztes Leerzeichen vor km/h, `test3261` einen Soft-Hyphen im Goldwert Hydro-gen. Bei den anderen Quellenfällen muss geprüft werden, ob Aliasnormalisierung reicht oder die Annotation sachlich überarbeitet werden muss.

Bei den offenen Fällen wurden neun Generatorantworten als Format-/Parsefehler erkannt. Die 28 protokollierten Validierungsablehnungen wegen fehlender Gold-Aliase sind davon getrennt; sie dürfen nicht als JSON-Fehler gezählt werden. Vorschlagszahlen sind keine Fragezahlen.

## Gespeichert ist nicht automatisch ein brauchbarer Distraktor

Gezielte Beispiele; keine repräsentative Qualitätsstichprobe und keine Schätzung einer Fehlerrate:

| ID | Frage | Gold | Gespeicherte falsche Antwort | Einschätzung |
|---|---|---|---|---|
| test1 | how many episodes are in chicago fire season 4 | 23 | 24 | Passender Zahlentyp; 23 → 24 als beabsichtigtes Gegenfaktum. |
| test2 | who sings love will keep us alive by the eagles | Timothy B. Schmit | Seth MacFarlane | Andere Person als Sänger; als Zielantwort strukturell passend. |
| test85 | when was the debating club established in almora | 1871 A.D | Carlotta Alberich | Personenname statt Gründungsjahr: offensichtlich falscher semantischer Antworttyp. |
| test163 | calpurnia son name in to kill a mockingbird | Zeebo | Calpurnia had a brother named Jem | Satz über Bruder statt gesuchtem Sohn: Relation verfehlt. |
| test311 | where do the sharks play in san jose | the SAP Center | packed | packed ist kein alternativer Spielort: unbrauchbarer Ort-Distraktor. |

Das Hauptproblem ist daher nicht nur, die 42 Lücken zu schließen: Auch bereits akzeptierte Ziele können die Frage semantisch verfehlen. Die aktuelle Typprüfung arbeitet überwiegend mit Form und Frageanfang; sie erkennt z. B. Freitext und Orte nicht zuverlässig als echte semantische Kategorien.

## Empfohlene Reihenfolge

1. Gold-/Quelltext für die elf Aliasfälle prüfen und Unicode/Leerzeichen vereinheitlichen. Solche Fälle vor Modellaufrufen abfangen; drei neue Distraktoren beheben die Quelle nicht.
2. Antworttyp aus Frage, Gold und Aliases ableiten: Jahr mit Zusätzen, Zeitspanne, Anzahl mit Einheit, Datum mit Granularität, Titel, Person, Ort und relationale Freitextantwort. Qualifikatoren wie more than/by/after gezielt erhalten.
3. Im Prompt den semantischen Antwortslot verdeutlichen, etwa „anderer Albumtitel“ oder „anderes Gründungsjahr“, statt nur entity_or_text. Die Frage soll mit dem Ziel grammatisch und semantisch beantwortbar bleiben.
4. Bereits gespeicherte Ziele auf Typ- und Relationspassung prüfen; zuerst 211 Retryfälle und Freitextfälle, danach eine zufällige Stichprobe der Erstaufruf-Erfolge. Plausibilität und tatsächliche Falschheit zur konkreten Frage separat bewerten.
5. Kleine Vergleichsläufe mit denselben Fragen durchführen: aktuelle Basis gegen bessere Typisierung/Prompt, optional stärkeres Modell. Akzeptanz, semantische Brauchbarkeit, Wiederholungen und Aufrufkosten getrennt messen. Regeln nicht pauschal deaktivieren.

## Alle 42 offenen Fragen

| ID | Frage | Gold | Gruppe / Diagnose |
|---|---|---|---|
| test4 | nitty gritty dirt band fishin in the dark album | Hold On | Frage kopiert: Modell übernimmt Frage oder deren Suchbegriffe statt einer passenden alternativen Antwort. |
| test39 | royal society for the protection of birds number of members | more than a million | Typ / Format / Granularität: Gold more than a million gilt als Freitext; numerische Alternativen werden abgelehnt. |
| test98 | what year is the deer hunter set in | late 1967 | Typ / Format / Granularität: Gold late 1967 gilt als Freitext; vorgeschlagene Jahreszahlen werden abgelehnt. |
| test125 | when did the prr first begin its passenger service between pittsburgh and philadelphia | by 1854 | Typ / Format / Granularität: Gold by 1854 gilt als Freitext; numerische Jahresalternativen werden abgelehnt. |
| test272 | how many terms can a mayor serve in texas | two four-year terms | Typ / Format / Granularität: Vorschläge werden überwiegend durch Typ-/Formatregeln verworfen; konkrete Versuche im Inventar prüfen. |
| test591 | what is the account number of airtel payment bank | Your Airtel mobile number | Typ / Format / Granularität: Gold beschreibt eine Art Kontonummer; Modell liefert konkrete Zahlen statt einer alternativen Regel. |
| test651 | when did ireland gain its independence from england | In 1922 | Typ / Format / Granularität: Vorschläge werden überwiegend durch Typ-/Formatregeln verworfen; konkrete Versuche im Inventar prüfen. |
| test660 | when was the day that the pubs ran dry in the poem 9-3 | 31 October 1972 | Typ / Format / Granularität: Vollständiges Datum erwartet; Modell liefert u. a. Jahre oder Monat/Tag. |
| test686 | when did they stop saying mass in latin | from 1967 onwards | Typ / Format / Granularität: Vorschläge werden überwiegend durch Typ-/Formatregeln verworfen; konkrete Versuche im Inventar prüfen. |
| test688 | when does the eclipse end in the us | 2:44 p.m. EDT | Gold-Alias fehlt in Quelle: Quellen-/Alias-Abgleich blockiert auch strukturell gültige Vorschläge; kein Beweis fehlender semantischer Evidenz. |
| test718 | when was the first election held in india | 1951–52 | Typ / Format / Granularität: Vorschläge werden überwiegend durch Typ-/Formatregeln verworfen; konkrete Versuche im Inventar prüfen. |
| test742 | how many writes does a flash drive have | up to 100,000 | Typ / Format / Granularität: Vorschläge werden überwiegend durch Typ-/Formatregeln verworfen; konkrete Versuche im Inventar prüfen. |
| test763 | time of the state of the nation address | after 9pm ET | Gold-Alias fehlt in Quelle: Quellen-/Alias-Abgleich blockiert auch strukturell gültige Vorschläge; kein Beweis fehlender semantischer Evidenz. |
| test823 | who's the model on let's make a deal | Tiffany Adams Coyne | Gold-/Namensvarianten: Modell wiederholt Gold oder erzeugt gekürzte/erweiterte Varianten statt eines anderen Ziels. |
| test960 | when is the opening ceremonies of the olympics 2018 | 9 February 2018 | Typ / Format / Granularität: Vollständiges Datum erwartet; Modell liefert Monat/Jahr. |
| test983 | who won the most medals at the 2014 winter olympics | Russia ( | Gold-Alias fehlt in Quelle: Goldwert Russia ( wirkt unvollständig; Annotation prüfen. |
| test1175 | when did the black death end in england | December 1349 | Typ / Format / Granularität: Monat/Jahr im Gold wird als Freitext behandelt; Modell liefert Jahreszahlen. |
| test1222 | who won season 5 of hell's kitchen | Executive Chef Danny Veltri | Gold-/Namensvarianten: Modell wiederholt Gold oder erzeugt gekürzte/erweiterte Varianten statt eines anderen Ziels. |
| test1286 | 1 the bureau of land management manages over acres of land in the united states | 247.3 million | Gold-Alias fehlt in Quelle: Quellen-/Alias-Abgleich blockiert auch strukturell gültige Vorschläge; kein Beweis fehlender semantischer Evidenz. |
| test1317 | what is the top speed of thrust ssc | 1,228 km/h | Gold-Alias fehlt in Quelle: Geschütztes Leerzeichen in Quelle: 1,228 NBSP km/h statt gewöhnlichem Leerzeichen. |
| test1671 | when were 2 dollar bills stopped being made | current denomination of U.S. currency | Typ / Format / Granularität: Gold verneint die vorausgesetzte Einstellung der Banknote; Modell liefert Jahreszahlen. |
| test1711 | how many strong verbs are there in german | more than 200 | Typ / Format / Granularität: Vorschläge werden überwiegend durch Typ-/Formatregeln verworfen; konkrete Versuche im Inventar prüfen. |
| test1764 | tujhe dekha toh yeh jana sanam movie name | Dilwale Dulhania Le Jayenge | Gold-/Namensvarianten: Modell wiederholt Gold oder erzeugt gekürzte/erweiterte Varianten statt eines anderen Ziels. |
| test1806 | who became king of erebor after thorin dies | Dáin | Gold-Alias fehlt in Quelle: Quellen-/Alias-Abgleich blockiert auch strukturell gültige Vorschläge; kein Beweis fehlender semantischer Evidenz. |
| test1941 | the chinese dragons are protectors of how many seas diggy | Four Seas | Frage kopiert: Modell übernimmt Frage oder deren Suchbegriffe statt einer passenden alternativen Antwort. |
| test1969 | when was harry potter and the philosophers stone published | in 1997 | Typ / Format / Granularität: Gold in 1997 gilt als Freitext; vorgeschlagene Jahreszahlen werden als anderer Typ abgelehnt. |
| test2108 | when does buffy tell her friends she was in heaven | Something to Sing About | Typ / Format / Granularität: Gesucht ist ein Song-/Szenentitel; Modell liefert Jahreszahlen. |
| test2154 | when did the uk and us become allies | Since 1940 | Typ / Format / Granularität: Vorschläge werden überwiegend durch Typ-/Formatregeln verworfen; konkrete Versuche im Inventar prüfen. |
| test2192 | what is the maximum data rate for the 802.11a standard select one | 54 Mbit/s | Gold-Alias fehlt in Quelle: Quellen-/Alias-Abgleich blockiert auch strukturell gültige Vorschläge; kein Beweis fehlender semantischer Evidenz. |
| test2303 | when was the last time mount ruapehu erupted | 25 September 2007 | Typ / Format / Granularität: Vollständiges Datum erwartet; Modell liefert nur Jahre. |
| test2565 | when did lord howe island became a world heritage site | in 1982 | Typ / Format / Granularität: Vorschläge werden überwiegend durch Typ-/Formatregeln verworfen; konkrete Versuche im Inventar prüfen. |
| test2637 | gestational sac and yolk sac but no baby at 7 weeks | an anembryonic gestation | Gold-Alias fehlt in Quelle: Quellen-/Alias-Abgleich blockiert auch strukturell gültige Vorschläge; kein Beweis fehlender semantischer Evidenz. |
| test2648 | what was the lowest barometric pressure for hurricane irma | 914 mbar (hPa) | Gold-Alias fehlt in Quelle: Quellen-/Alias-Abgleich blockiert auch strukturell gültige Vorschläge; kein Beweis fehlender semantischer Evidenz. |
| test2658 | when was the canadian pacific railway started and finished | between 1881 and 1885 | Typ / Format / Granularität: Vorschläge werden überwiegend durch Typ-/Formatregeln verworfen; konkrete Versuche im Inventar prüfen. |
| test2721 | how many episodes are there in ordeal by innocence | three part | Typ / Format / Granularität: Gold three part gilt als Freitext; four/five usw. werden als Zahlen abgelehnt. |
| test2855 | surface area in many kinds of organs is important because the amount of surface determines | effective gas exchange | Frage kopiert: Modell übernimmt Frage oder deren Suchbegriffe statt einer passenden alternativen Antwort. |
| test2988 | the boiling point of water is 100 degrees celsius express this in si units | 100 °C | Gold-Alias fehlt in Quelle: Quellen-/Alias-Abgleich blockiert auch strukturell gültige Vorschläge; kein Beweis fehlender semantischer Evidenz. |
| test3029 | university of michigan school of public health ranking | #4 | Frage kopiert: Modell übernimmt Frage oder deren Suchbegriffe statt einer passenden alternativen Antwort. |
| test3114 | when did response to state of the union start | in 1966 | Typ / Format / Granularität: Vorschläge werden überwiegend durch Typ-/Formatregeln verworfen; konkrete Versuche im Inventar prüfen. |
| test3115 | how many hospitals are there in the united states | 5,534 registered hospitals | Typ / Format / Granularität: Gold enthält Anzahl und hospitals; nackte Zahlen werden als anderer Typ abgelehnt. |
| test3261 | the first element on the periodic table is | Hydro­gen | Gold-Alias fehlt in Quelle: Gold enthält unsichtbaren Soft-Hyphen in Hydro-gen; Quelle schreibt hydrogen. |
| test3352 | when did vat change from 17.5 to 20 | 4 January 2011 | Typ / Format / Granularität: Vollständiges Datum erwartet; Modell liefert Monat/Jahr; zusätzlich ein JSON-Formatfehler. |

## Vollständiges Inventar und Quellen

- [Alle 2,248 Fragen mit Status, Gold, Zielantwort und Versuchen](/Users/ssl/Documents/ZHAW/HS26/github/rag-analysis/data/analysis/false-answer-generation-2026-10-23/questions.jsonl)
- [Aggregierte Kennzahlen](/Users/ssl/Documents/ZHAW/HS26/github/rag-analysis/data/analysis/false-answer-generation-2026-10-23/summary.json)
- [Gespeicherte Zielantworten](/Users/ssl/Documents/ZHAW/HS26/github/rag-analysis/data/generated/false-deterministic/generated_false_answers.jsonl)
- [Fragen-Loader inklusive Label-Join](/Users/ssl/Documents/ZHAW/HS26/github/rag-analysis/src/evlab/data/nqplus.py)
- [Generator und Typregeln](/Users/ssl/Documents/ZHAW/HS26/github/rag-analysis/src/evlab/generation/false_answer.py)
- [Versuchssteuerung und Quellenprüfung](/Users/ssl/Documents/ZHAW/HS26/github/rag-analysis/src/evlab/generation/false_answers.py)

Zusätzlich ausgewertet: `data/nqplus/clean/queries.jsonl`, `generated_labels.jsonl`, Originalkorpus `data/cocktail/nq/corpus/human.jsonl` und das False-Answer-Rejectlog im false-deterministic-Verzeichnis. Die False-Document-Rejects sind nicht Teil dieser Erfolgsstatistik. Änderungen am False-Document-Prompt erzeugen keine neuen False Answers und verbessern diese gespeicherten Ziele nicht rückwirkend.
