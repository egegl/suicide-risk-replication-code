"""Synthetic training data.

ICHI 2027 paper (E2): pkgc.py builds the fold-local cases and writer prompts,
screens the writer outputs, assembles the training files, and draws and scores
the human audit. realism.py, realism_headroom.py, and diagnose.py compute the
pilot's synthetic-vs-real classifier AUC. bundles.load_rows, filters, and
state are shared helpers.

Competition-era pipeline (GPT-5.4-mini via the OpenAI Batch API, driven by
legacy/generate_synthetic.py), not the paper's method: assemble, batch,
bundles, emit_synth, exemplars, noise, parse, pipeline, prompts,
validate_synth.
"""
