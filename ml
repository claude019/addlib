Following up with the complete list of criteria that can result in an alert not producing a recommendation row. There are three distinct stages — worth separating because each has a different root cause and a different fix if something looks wrong.

Stage 1 — Extraction (alert never even enters the pipeline)

Source: raw_data_extraction.py, query against aa_hsbc_rapid2_hzn_alert_conformed.

An alert is excluded at this stage if any of these fail:

batch <> excluded_batch — batch value must not equal the configured excluded batch
ingest_chanl_id IN (2, 8) — must be one of these two channels
is_manual_duplicate IS NULL — must not be flagged as a manual duplicate
Batch must match one of: a configured prefix (currently M, W, D — case-sensitive), be purely numeric, or be in an explicit allow-list
alert_cat_cde must be in the configured category list

Stage 2 — Text preprocessing (alert extracted, but discarded before scoring)

Source: text_preprocessing.py.

An alert is discarded here if its text (title/summ_contnt/summ_updt) is blank after cleaning, matches a configured junk-text pattern, or fails the language/length filter. Discarded alerts are not silently dropped — they’re re-appended to the output with LOB_Recommended = "Filtered in Text Preprocessing", so they should still appear in the table with that label.

Stage 3 — Location/LOB mapping (alert scored, but no valid business entity to map to)

Source: aa_hsbc_helios_entity_mapper, filtered to entity_mapper_status = "Active".

An alert’s Location is checked against this mapping. If there’s no active (location, business_entity_l2) pair for that location in Helios, the alert is excluded — no valid business entity exists for the model to score against. This is a Helios data-availability gap, not a pipeline defect.
