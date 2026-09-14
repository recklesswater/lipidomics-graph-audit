# -*- coding: utf-8 -*-
"""
Centralised path configuration.

The input file is not shipped with the repository (see the Data section of the
README). By default the scripts look in data/. Every path can be overridden with
an environment variable:

    LIPIDMAPS_XLSX    path to the LipidMaps xlsx export
    LIPIDMAPS_SHEET   worksheet name, default "5大类脂质-get"
    AUDIT_OUT         output directory, default ./outputs
"""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DATA_XLSX = Path(os.environ.get("LIPIDMAPS_XLSX", str(ROOT / "data" / "LipidMaps.xlsx")))
# The sheet name is part of the upstream LipidMaps export and is kept verbatim.
SHEET = os.environ.get("LIPIDMAPS_SHEET", "5大类脂质-get")
OUT = Path(os.environ.get("AUDIT_OUT", str(ROOT / "outputs")))
