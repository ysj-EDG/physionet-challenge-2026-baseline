# G1-A Decision Requests

The following are evidence-backed requests for GPT and user review. No choice is made here.

## 1. Minimum valid EEG epochs
The audit counts are available in `audit/g1a_stage_support.csv`; descriptive distributions are:
- Wake: min=0, P05=66.00, median=199.50, P95=473.05, max=902.
- REM: min=0, P05=4.00, median=102.00, P95=213.00, max=637.
- N1: min=0, P05=18.00, median=59.00, P95=143.00, max=368.
- N2: min=0, P05=211.00, median=413.00, P95=580.00, max=1738.
- N3: min=0, P05=0.00, median=82.00, P95=174.05, max=282.
- NREM: min=0, P05=330.00, median=574.00, P95=727.00, max=1738.
- All-stage: min=0, P05=745.95, median=899.00, P95=1036.00, max=2514.
Candidates to review: 1, 2, 3, 5, 10, 20, 30, or 60 effective epochs. No threshold was selected and no patient was excluded.

## 2. IQR with very few valid epochs
Choose whether one/two observations use an explicitly defined deterministic IQR convention, are represented as missing, or are withheld until a larger support threshold. This audit does not choose among them.
## 3. Spatial support threshold
The 36 candidates use same-epoch intersections: three left/right pairs for N2 and REM, and four-lead F3/F4/O1/O2 intersections for N2/N3/REM. Review whether their support threshold must equal EEG54's threshold.
## 4. Demo5 sex encoding
Existing helper mapping merges Other and missing into Other/Unknown; 0 of 6600 demographic rows map to Unknown. Candidates: scalar Female/Male with an explicit Unknown code; scalar Female/Male with Unknown missing; or a pre-approved one-hot allocation that still preserves five columns.
## 5. Demo5 race three columns
Existing helper categories are Asian/Black/Others/Unavailable/White; 185 of 6600 rows map to Unavailable. Candidates: [White, Black, Asian] with Others/Unavailable all-zero; [White, Black, Other-or-Unavailable]; or [Asian, Black, White] with an explicit missing policy. The three columns and order are not frozen.
## 6. Demo5 unknown and BMI representation
Existing BMI helper maps invalid/nonfinite values to zero; 5062 rows are nonfinite/missing under the audit parser. Candidates include preserving zero as the existing sentinel, using a fold-fitted imputation plus a missing indicator only if the dimensionality is changed by approval, or retaining a deterministic missing state outside the five columns. No choice is made.
## 7. Other unresolved semantics
Review CAISR20 missing semantics, lawful zero versus unavailable fills in EMG/Resp, HRV window-level validity and alignment, and any source unit that remains `source-defined`. These are marked `PENDING_G1B_REVIEW` in the CSV manifest.

Labels, model performance, and LOSO metrics were not used.
