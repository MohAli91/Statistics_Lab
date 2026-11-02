# Offline sampling frames

The five historical CSVs (`housing`, `steel`, `bikes`, `air`, `oats`) contain actual recorded values from the linked datasets. Five additional CSVs (`shafts`, `production_lines`, `pallets`, `asphalt`, `concrete`) are default generated teaching populations, clearly marked in `sources.json`. `sources.json` defines the target finite frame, transformations of units, subset selection, derived strata/clusters, schematic-layout limits, and original/derived SHA256 hashes.

- Housing: Pace & Barry California Housing archive, also used by scikit-learn. Retain the original research attribution. Geographic grouping is added for teaching.
- Steel: Sathishkumar V E, Shin & Cho, UCI Steel Industry Energy Consumption. CC BY 4.0.
- Bikes: Fanaee-T, UCI Bike Sharing. CC BY 4.0.
- Air: De Vito et al., UCI Air Quality. UCI lists CC BY 4.0 and retains a research-only author note; this frame is provided for academic teaching/research. Missing −200 readings are excluded.
- Oats: Yates split-plot trial, via Pinheiro & Bates' `nlme::Oats` and the Rdatasets mirror. Source nlme package license GPL (>=2). The frame retains all 72 factual yield observations with attribution. Display positions are schematic.

The sources contain historical measurements; the app makes no live-data, present-market, regulatory-compliance, or causal-treatment claim. Historical measurement values are not synthetically filled, imputed, or perturbed. Generated engineering cases are created separately by `engineering_sampling.py`; their process parameters and recipe hashes are documented explicitly.

Optional refresh:

```bash
python sampling_data/build_sources.py --raw-dir sampling_raw
```

Refresh requires internet access. Normal app operation reads only these bundled files and does not run this script. Keep all ten CSVs and `sources.json` when deploying.

Generated defaults can be rebuilt offline with `python engineering_sampling.py` from the project root. Their recipe hashes describe code-generation settings, rather than a source download.
