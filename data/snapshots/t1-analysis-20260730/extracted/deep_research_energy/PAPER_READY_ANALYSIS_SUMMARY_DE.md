# Paperreife Abschlussanalyse für T1

## Status

**Empirisches Full-Paper-GO.** Die zentrale Placement-Story ist reproduziert, die zuvor fehlenden Robustheitsanalysen sind abgeschlossen und das Analysepaket ist vollständig ausführbar.

Ein Clean-Room-Lauf mit zuvor gelöschten Outputs und Abbildungen erzeugt:

- 37 CSV-Tabellen,
- fünf Abbildungen,
- 42/42 byte-identische generierte Dateien,
- 77/77 erfolgreiche SHA-256-Prüfungen.

Die vollständige Sessionbasis umfasst 918 Konfigurationen mit jeweils fünf Sessionmedianen, insgesamt 4590 Sessionzeilen.

## Eingefrorene Hauptbefunde

### 1. Objective-dependent Placement Map

- 51 Workload-Größen-/Shape-Zellen.
- 24/51 kanonisch robuste Laufzeit-/Energiekonflikte.
- Konflikte in allen sechs Workloadfamilien.
- 23/24 Konflikte sind RTX 3090 gegen RTX 5060 Ti und liegen damit in derselben NVML-Boardenergiedomäne.

### 2. Vollständig konsistentes großes GPU-Regime

Für AXPY, STREAM und REDUCTION bei 16M, 32M, 64M, 128M und 256M Elementen gilt in 15/15 Zellen:

- RTX 3090 laufzeitoptimal,
- RTX 5060 Ti energieoptimal.

Sessionbasierte Bootstrapwerte:

- medianer RTX-3090-Speedup: **2,121874×**, 95-%-CI **[2,121340; 2,123785]**;
- mediane Boardenergieersparnis der RTX 5060 Ti: **42,4739 %**, 95-%-CI **[42,2244 %; 42,9774 %]**;
- medianes Leistungsratio RTX 3090/RTX 5060 Ti: **3,6994×**;
- medianes Energieratio RTX 3090/RTX 5060 Ti: **1,7383×**.

Die kürzere Laufzeit der RTX 3090 kompensiert den höheren Boardleistungsbedarf nicht vollständig.

### 3. Die Konflikte bleiben auch bei großzügiger Toleranz hart

Die nichtzirkuläre gemeinsame-Näherungsoptimum-Sensitivität ergibt:

- bei 10 % Toleranz besitzen **22/24** Konfliktzellen kein gemessenes gemeinsames Näherungsoptimum;
- bei 20 % Toleranz besitzen noch **19/24** kein gemeinsames Näherungsoptimum;
- alle 27 Nicht-Konfliktzellen besitzen bei 5 %, 10 % und 20 % mindestens eine gemeinsame Näherungsoption.

### 4. Verschachtelte Sessionvalidierung

Auswahl jeweils auf vier Sessions, Evaluation auf der fünften:

#### GPU-only, 255 Folds je Metrik

- Laufzeitplattform: **100,00 %** exakte Treffer, **100,00 %** innerhalb 5 %.
- Energieplattform: **99,61 %** exakte Treffer und innerhalb 5 %.
- EDP-Plattform: **99,61 %** exakte Treffer, **100,00 %** innerhalb 5 %.
- Einzige GPU-Energieabweichung: die bereits unsichere STRIDED_GEMM-Zelle bei N=512; Holdout-Regret 6,20 %.

#### Kanonischer symmetrischer GPU-Konfliktkern

Alle **23/23** kanonischen RTX-3090-/RTX-5060-Ti-Konfliktzellen:

- bleiben in allen fünf Holdouts Konfliktzellen;
- wählen in jedem Trainingsfold unterschiedliche Laufzeit- und Energieplattformen;
- liegen mit beiden ausgewählten Optionen in jedem Fold innerhalb 5 % des jeweiligen GPU-Session-Oracles.

#### All-platform

- Laufzeitauswahl: **95,69 %** der 255 Folds innerhalb 5 %.
- Energieauswahl: **95,29 %** innerhalb 5 %.
- EDP-Auswahl: **95,29 %** innerhalb 5 %.
- 22/24 Konfliktzellen bleiben in 5/5 Holdouts all-platform Konflikte; 24/24 in mindestens 4/5.

Die zwei Grenzfälle sind:

1. REDUCTION 16M, AMD 64T: eine Session liegt bei etwa 0,26× der üblichen Laufzeit und 0,35× der üblichen Energie;
2. STREAM 4M, AMD 32T: eine Session liegt bei etwa 0,52× der üblichen Laufzeit und 0,54× der üblichen Energie.

Die Median-von-fünf-Klassifikation bleibt stabil. Für das Paper ist jedoch zwischen perfekter GPU-only-Holdout-Stabilität und 22/24 vollständiger all-platform Stabilität zu unterscheiden.

### 5. Statische GPU-Policies besitzen extreme Tail-Kosten

Gleichgewichtete GPU-only Workload-Größen-Zellen:

- Immer RTX 3090: Energie-CVaR10 **486,87 %**, 95-%-CI **[482,59 %; 490,22 %]**; nur **29,41 %** der Zellen innerhalb 5 % des Energieoracles.
- Immer RTX 5060 Ti: Laufzeit-CVaR10 **427,26 %**, 95-%-CI **[426,64 %; 427,76 %]**; nur **15,69 %** der Zellen innerhalb 5 % des Laufzeitoracles.

Die bekannten kanonischen Median-Regrets bleiben:

- immer RTX 3090: rund 72,38 % Energie-Regret;
- immer RTX 5060 Ti: rund 110,56 % Laufzeit-Regret.

### 6. EDP löst den Zielkonflikt überwiegend nicht

- In 48/51 Zellen stimmt die EDP-Punktplattform mit der Laufzeitplattform überein.
- Median der Plattform-Rangkorrelationen:
  - Spearman EDP/Laufzeit: 1,0;
  - Kendall EDP/Laufzeit: 1,0;
  - Spearman EDP/Energie: 0,8;
  - Kendall EDP/Energie: 0,667.

Zulässiger Claim: In dieser Kampagne verhält sich EDP überwiegend geschwindigkeitsorientiert und beseitigt den beobachteten Energie-Laufzeit-Placement-Konflikt nicht.

### 7. CPU-Threadzahl bildet eine zweite Entscheidungsebene

- Laufzeit- und Energie-Threadoptimum unterscheiden sich in 65/102 CPU-Zellen.
- Maximale Threadzahl verursacht median:
  - 7,47 % Laufzeit-Regret,
  - 38,52 % Energie-Regret,
  - 63,06 % EDP-Regret.

Intel REDUCTION 256M, 4T statt laufzeitoptimaler 8T:

- Laufzeitaufschlag: **0,244 %**, 95-%-CI **[-0,014 %; 0,443 %]**;
- Energieersparnis: **28,29 %**, 95-%-CI **[26,31 %; 35,59 %]**.

Die verschachtelte CPU-Threadauswahl ist erwartungsgemäß weniger stabil als die GPU-Plattformwahl:

- Energie: 91,96 % der Folds innerhalb 5 %;
- Laufzeit: 89,02 %;
- EDP: 86,86 %.

Die hohen Tail-Regrets konzentrieren sich vor allem auf sessionvariable AMD-REDUCTION-/STREAM-Konfigurationen bei kleineren Größen.

### 8. Die Entscheidungsfront bleibt sehr klein

Nach der projektspezifischen toleranzbewussten Dominanzregel:

- strikt: mittlere Frontgröße 1,549/18, 91,39 % dominiert;
- 2 %: 1,510/18, 91,61 % dominiert;
- 5 %: 1,510/18, 91,61 % dominiert;
- CPU auf der Front in 6/51 Zellen bei allen drei Einstellungen.

Im Paper muss die verwendete praktische Dominanzdefinition explizit angegeben werden, da sie nicht einfach eine Erweiterung der strikten Front ist.

### 9. Intel-DRAM-Sensitivität ändert die Threadentscheidungen nicht

Für AXPY, STREAM und REDUCTION auf Intel:

- 0/27 Threadoptima ändern sich beim Wechsel von Package Energy auf Package+DRAM;
- medianer DRAM-Aufschlag am Package-Optimum:
  - AXPY 2,65 %,
  - STREAM 2,32 %,
  - REDUCTION 1,31 %.

Der GPU-only-Hauptbefund ist davon unabhängig.

## Verbleibende Limitation

Die numerische konfigurierte beziehungsweise standardmäßige RTX-3090-Power-Limit-Einstellung ist in den gelieferten Session-Snapshots nicht enthalten. Vorhanden und reproduziert sind:

- Boardleistung,
- Clockbereiche,
- Temperaturen,
- für AXPY `throttle_masks=0x4`.

Für einen numerischen Stock-Power-Limit-Satz werden die ursprünglichen Run-Logs oder Systemmetadaten benötigt. Ein neuer Power-Cap-Sweep ist für die T1-Hauptstory nicht erforderlich.

## Empfohlene Paperclaims

1. **Harte objective-dependent Placement-Konflikte:** 24/51 Zellen, davon 23 symmetrische GPU-GPU-Fälle.
2. **Vollständiges großes Regime:** 15/15 mit 2,12× Speedup gegen 42,47 % Energieersparnis.
3. **Kontext ist erforderlich:** statische Policies besitzen geringe Coverage und sehr hohe Tail-Regrets.
4. **GPU-Konflikte sind out-of-session stabil:** 23/23 in allen verschachtelten Holdouts.
5. **Threadzahl ist eine zweite Zielentscheidung:** maximale Threads und Laufzeitoptima können energetisch stark suboptimal sein.
6. **EDP ist hier kein neutraler Vermittler:** 48/51-mal folgt es der Laufzeitplattform.

## Formulierungsgrenzen

Nicht behaupten:

- 24/24 all-platform Konflikte seien in jeder einzelnen Session vollständig stabil;
- die 51 Zellen seien unabhängige Experimente;
- EDP sei allgemein ungeeignet;
- nominelle Operational Intensity widerlege physischen Speicherverkehr;
- CPU- und GPU-Energiedomänen seien identisch;
- vollständige Systemenergie oder Transferkosten seien gemessen;
- der RTX-3090-Zustand sei ohne die fehlende numerische Power-Limit-Metadatei sicher als Stock-Limit belegt.

## Analysestatus

Die für T1 erforderliche Analyse ist abgeschlossen. Zusätzliche explorative Modelle, Conv2D-Shape-Regressionen, Changepoint-Fits oder ML-Prädiktoren sind keine Voraussetzung für das Paper.
