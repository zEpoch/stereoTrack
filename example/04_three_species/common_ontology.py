"""Conservative shared ontology for cortex comparisons across species."""

from __future__ import annotations

import numpy as np
import pandas as pd


UNKNOWN_VALUES = {"", "na", "nan", "none", "unknown", "unassigned", "discard", "lunknown"}


def _clean(values, index=None) -> pd.Series:
    if isinstance(values, pd.Series):
        series = values.astype("string")
    else:
        series = pd.Series(values, index=index, dtype="string")
    return series.fillna("Unknown").str.strip()


def _contains(text: pd.Series, pattern: str) -> pd.Series:
    return text.str.contains(pattern, case=False, regex=True, na=False)


def harmonize_cell_family(
    species,
    cell_class_broad,
    cell_subclass,
    cell_type,
) -> pd.Series:
    """Map source annotations to shared, biologically conservative cell families."""
    broad = _clean(cell_class_broad)
    subclass = _clean(cell_subclass, index=broad.index)
    cell_type = _clean(cell_type, index=broad.index)
    species = _clean(species, index=broad.index)
    text = (subclass + " " + cell_type).str.upper()
    result = pd.Series("Unknown", index=broad.index, dtype="string")

    # Non-neuronal families. Detailed source labels take precedence over broad classes.
    result[_contains(text, r"(?:^|[^A-Z])(?:ASC|AST|ASTROCY)(?:[^A-Z]|$)")] = "Astrocyte"
    result[
        _contains(text, r"(?:^|[^A-Z])(?:OLG|OL)(?:[_./-]|$)|OLIGODENDROCYTE")
        & ~_contains(text, r"OPC")
    ] = "Oligodendrocyte"
    result[_contains(text, r"(?:^|[^A-Z])OPC(?:[_./-]|$)|PRECURSOR CELLS")] = "OPC"
    result[_contains(text, r"(?:^|[^A-Z])(?:MG|MGL)(?:[_./-]|$)|MICROGLIA")] = "Microglia"
    result[_contains(text, r"(?:^|[^A-Z])(?:EC|EDC)(?:[_./-]|$)|ENDOTHELIAL")] = "Endothelial"
    result[_contains(text, r"VLMC|VASCULAR|LEPTOMENINGEAL|PERICYTE")] = "Vascular/VLMC"

    # Conserved inhibitory neuron families.
    gaba = broad.str.lower().eq("gabaergic") | _contains(text, r"GABA")
    pvalb = _contains(text, r"PVALB|(?:^|[^A-Z])PV(?:[_./-]|$)|CHODL|CHC")
    sst = _contains(text, r"(?:^|[^A-Z])SST(?:[_./-]|$)")
    vip = _contains(text, r"(?:^|[^A-Z])VIP(?:[_./-]|$)") & ~_contains(text, r"VIP[_-]?RELN")
    lamp5_reln = _contains(text, r"LAMP5|RELN|SNCG")
    result[gaba & pvalb] = "PVALB interneuron"
    result[gaba & sst] = "SST interneuron"
    result[gaba & vip] = "VIP interneuron"
    result[gaba & lamp5_reln] = "LAMP5/RELN interneuron"
    result[gaba & result.eq("Unknown")] = "Other GABAergic neuron"

    glutamatergic = broad.str.lower().eq("glutamatergic") | _contains(text, r"GLU")
    result[glutamatergic & result.eq("Unknown")] = "Glutamatergic neuron"

    non_neuronal = broad.str.lower().eq("non-neuronal")
    result[non_neuronal & result.eq("Unknown")] = "Other non-neuronal"

    source_unknown = (
        broad.str.lower().isin(UNKNOWN_VALUES)
        & subclass.str.lower().isin(UNKNOWN_VALUES)
        & cell_type.str.lower().isin(UNKNOWN_VALUES)
    )
    result[source_unknown] = "Unknown"
    result.name = "cell_family_common"
    return result.astype(str)


def harmonize_layer(layer) -> pd.Series:
    """Map source layer labels to a shared laminar ontology without guessing mixed labels."""
    source = _clean(layer)
    normalized = source.str.upper().str.replace("LAYER", "", regex=False).str.replace(" ", "", regex=False)
    normalized = np.where(pd.Series(normalized).str.startswith("L"), normalized, "L" + normalized)
    normalized = pd.Series(normalized, index=source.index, dtype="string")

    result = pd.Series("Unknown", index=source.index, dtype="string")
    result[normalized.eq("L1")] = "L1"
    result[normalized.isin(["L2", "L3", "L2/3"])] = "L2/3"
    result[normalized.eq("L4")] = "L4"
    result[normalized.eq("L5")] = "L5"
    result[normalized.eq("L6")] = "L6"
    result[_contains(source, r"WHITE.?MATTER|(?:^|[^A-Z])WM(?:[^A-Z]|$)")] = "WM"
    result.name = "layer_common"
    return result.astype(str)
