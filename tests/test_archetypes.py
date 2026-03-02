"""
Archetype regression tests — pin top-N mechanism rankings for known character profiles.

Fixtures live in tests/fixtures/archetypes.json. These are deterministic (no LLM calls).
When scoring changes intentionally, re-run the capture script and update the fixture.
"""

import json
from pathlib import Path

import pytest

from mcp_server import _score_mechanisms, get_conn

FIXTURES = Path(__file__).parent / "fixtures" / "archetypes.json"
DB_PATH = Path(__file__).parent.parent / "db" / "mechanisms.sqlite"

pytestmark = pytest.mark.skipif(
    not DB_PATH.exists(),
    reason="mechanisms.sqlite not present — run db_load.py first",
)


@pytest.fixture(scope="module")
def conn():
    c = get_conn()
    yield c
    c.close()


def _load_archetypes():
    return json.loads(FIXTURES.read_text())


@pytest.fixture(params=_load_archetypes(), ids=lambda a: a["name"])
def archetype(request):
    return request.param


def test_archetype_top3(conn, archetype):
    """Top-3 mechanisms match the pinned expectation for this archetype."""
    results = _score_mechanisms(conn, archetype["profile"], archetype["situation"], top_n=5)
    actual_top3 = [r["id"] for r in results[:3]]
    expected = archetype["expected_top3"]
    assert actual_top3 == expected, (
        f"Archetype '{archetype['name']}': expected {expected}, got {actual_top3}"
    )
