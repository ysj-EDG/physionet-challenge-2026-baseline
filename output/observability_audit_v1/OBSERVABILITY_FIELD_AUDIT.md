# Observability field audit

- DATA2 records audited: 6600; sites: {'S0001': 5139, 'I0006': 1142, 'I0002': 319}.
- Sources: demographics.csv, ICD_codes_CI.csv, EDF headers; no model training and no NPZ changes.

## Candidate field classification

| Field | Source | Overall coverage | PSG-known | Inference-known | Future info | Outcome-derived | Safe |
|---|---|---:|---|---|---|---|---|
| CreationTime | demographics.csv:CreationTime | 6600/6600 (100.000%) | unclear | yes | no | no | requires_review |
| Age | demographics.csv:Age | 6600/6600 (100.000%) | yes | yes | no | no | yes |
| BMI | demographics.csv:BMI | 1538/6600 (23.303%) | yes | yes | no | no | yes |
| Time_to_Event | demographics.csv:Time_to_Event | 498/6600 (7.545%) | no | no | yes | yes | no |
| Last_Known_Visit_Date | demographics.csv:Last_Known_Visit_Date | 6600/6600 (100.000%) | no | no | yes | yes | no |
| Time_to_Last_Visit | demographics.csv:Time_to_Last_Visit | 6600/6600 (100.000%) | no | no | yes | yes | no |
| Cognitive_Impairment | demographics.csv:Cognitive_Impairment | 6600/6600 (100.000%) | no | no | no | yes | no |
| ICDDate_min | ICD_codes_CI.csv:ICDDate | 498/6600 (7.545%) | unclear | no | yes | yes | no |
| EDF_start_datetime | EDF header:startdate/starttime | 0/6600 (0.000%) | unknown | unknown | unknown | unknown | requires_review |

## Findings

- `EDF_start_datetime` is present in all 6600 scanned EDF headers and is the only direct acquisition-time candidate; it is safe only as an acquisition-time field, not as a proxy for diagnosis or follow-up.
- `CreationTime` is complete administrative metadata but is not established as PSG acquisition time; treat as requires review.
- `Time_to_Event`, `Last_Known_Visit_Date`, `Time_to_Last_Visit`, and `ICDDate` are future/outcome-derived and must not enter a predictor.
- Recording date availability should be checked by site in `observability_site_coverage.csv`; no follow-up variable is safe for inference.
- A 6-year eligibility proxy is conceptually computable from a validated acquisition/recording date plus a fixed cohort cutoff, but is **not approved here** until the date semantics and cutoff are specified.
