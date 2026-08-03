#!/usr/bin/env python3
"""RED/GREEN contract tests for the Operation-tab legacy-marking pass.

Matt's directive: promote the newer (v2) standards without deleting older
methods. v2-standard options come first in each dropdown, older options stay
fully functional but carry a " (legacy)" suffix, optgroup labels
("v2 standard" / "legacy") are used where the widget allows, and defaults
stay the v2 standards. The markup itself is not unit-testable in a JS
runner (none exists in this repo), so this pins the HTML text contract;
the rendered behavior is verified live in the browser (see
GUI_LAUNCH_FIX_NOTES.md).

Run: ./.venv312/bin/python -m pytest test_gui_legacy_marking.py -q
"""
from __future__ import annotations

import re
from pathlib import Path

HTML = (Path(__file__).parent / "webui" / "static" / "index.html").read_text(
    encoding="utf-8"
)


def _select_block(select_id: str) -> str:
    m = re.search(
        rf'<select id="{select_id}".*?</select>', HTML, flags=re.DOTALL
    )
    assert m, f"select #{select_id} not found"
    return m.group(0)


def test_mode_select_v2_group_first_and_iterate_marked_legacy() -> None:
    block = _select_block("mode")
    assert 'label="v2 standard"' in block
    assert 'label="legacy"' in block
    # fgm_solve listed before every legacy option; fgm_iterate marked legacy
    assert block.index('value="fgm_solve"') < block.index('value="fgm_iterate"')
    iterate_opt = re.search(r'<option value="fgm_iterate">[^<]*', block).group(0)
    assert "(legacy)" in iterate_opt
    # nothing removed
    for mode in (
        "single", "sweep", "optimizer", "turntable", "orientation_optimizer",
        "placement_optimizer", "shell_sweep", "fgm_solve", "fgm_iterate",
        "fgm_import", "prewarp",
    ):
        assert f'value="{mode}"' in block, f"mode {mode} removed"
    # default stays the standard single run (explicit, since fgm_solve is now first)
    single_opt = re.search(r'<option value="single"[^>]*>', block).group(0)
    assert "selected" in single_opt


def test_proxy_select_t_phi90_first_qrf_marked_legacy() -> None:
    block = _select_block("fgmIterProxy")
    assert 'label="v2 standard"' in block
    assert 'label="legacy"' in block
    assert block.index('value="T_phi90"') < block.index('value="Qrf"')
    qrf_opt = re.search(r'<option value="Qrf">[^<]*', block).group(0)
    assert "(legacy)" in qrf_opt
    for value in ("T_phi90", "T", "Qrf", "rho_rel", "__thorough__",
                  "__regime_adaptive__"):
        assert f'value="{value}"' in block, f"proxy {value} removed"
    t_opt = re.search(r'<option value="T_phi90"[^>]*>', block).group(0)
    assert "selected" in t_opt  # default stays the v2 standard


def test_corr_mode_proportional_listed_first_integral_marked_legacy() -> None:
    block = _select_block("fgmIterCorrMode")
    assert 'label="v2 standard"' in block
    assert 'label="legacy"' in block
    assert block.index('value="proportional"') < block.index('value="integral"')
    integral_opt = re.search(r'<option value="integral"[^>]*>[^<]*', block).group(0)
    assert "(legacy)" in integral_opt
    # the legacy-comparison mode keeps its own historical default
    assert "selected" in re.search(
        r'<option value="integral"[^>]*>', block
    ).group(0)
    for value in ("integral", "proportional", "hybrid"):
        assert f'value="{value}"' in block


def test_drive_mode_voltage_before_power_scaling_legacy() -> None:
    block = _select_block("advEnforceGen")
    assert block.index('value="false"') < block.index('value="true"')
    true_opt = re.search(r'<option value="true">[^<]*', block).group(0)
    assert "(legacy)" in true_opt
    false_opt = re.search(r'<option value="false">[^<]*', block).group(0)
    assert "v2 standard" in false_opt


def test_turntable_fixed_step_marked_legacy() -> None:
    block = _select_block("turntableProgramSelect")
    none_opt = re.search(r'<option value=""[^>]*>[^<]*', block).group(0)
    assert "(legacy)" in none_opt
    assert "selected" in none_opt  # default unchanged (no program auto-picked)


def test_import_bpp_default_is_v2_standard_4bpp() -> None:
    block = _select_block("fgmImportBpp")
    assert block.index('value="4"') < block.index('value="2"')
    four_opt = re.search(r'<option value="4"[^>]*>', block).group(0)
    assert "selected" in four_opt
    two_opt = re.search(r'<option value="2"[^>]*>', block).group(0)
    assert "selected" not in two_opt
