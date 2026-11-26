import csv
import io
import json
from pathlib import Path

import pytest

from smartphone_shopping_assistant.prepare import COLUMNS, prepare_catalog, read_catalog


def row(name="Example Phone 8/256GB", **updates):
    values = {
        "Smartphone": name,
        "Brand": "Example",
        "Model": "Phone",
        "RAM": "8",
        "Storage": "256",
        "Color": "Blue",
        "Free": "Yes",
        "Final Price": "231.60",
    }
    return values | updates


def csv_bytes(rows):
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=COLUMNS)
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode()


def test_raw_values_missing_memory_and_usd_are_preserved():
    original = row(RAM="", Storage="", Free="No")
    cards = read_catalog(csv_bytes([original]))
    card = cards[0]
    assert card["raw_fields"] == original
    assert card["ram_gb"] is None and card["storage_gb"] is None
    assert card["carrier_unlocked"] is False
    assert card["price"] == 231.6
    assert card["currency"] == "USD" and card["currency_status"] == "user_selected"
    assert card["source_currency"] is None and card["conversion_applied"] is False
    assert not {"source_ref", "source_url", "source_version"} & card.keys()
    assert "неизвестно" in card["text"] and "пересчёт не выполнялся" in card["text"]


def test_id_survives_reordering_and_price_updates_but_distinguishes_variants():
    first = read_catalog(csv_bytes([row(), row("Example Phone+ 8/256GB")]))
    changed = read_catalog(
        csv_bytes(
            [
                row("Example Phone+ 8/256GB"),
                row(**{"Final Price": "199.99"}),
            ]
        )
    )
    assert {c["name"]: c["id"] for c in first} == {c["name"]: c["id"] for c in changed}
    assert len({c["id"] for c in first}) == 2


@pytest.mark.parametrize(
    "field,value",
    [
        ("RAM", "8.5"),
        ("RAM", "-1"),
        ("Storage", "0"),
        ("RAM", "NaN"),
        ("Final Price", "NaN"),
        ("Final Price", "Infinity"),
        ("Final Price", "-1"),
        ("Final Price", "0"),
        ("Final Price", "1e-999"),
        ("Final Price", "1e999"),
        ("Final Price", "not a price"),
        ("Free", "Maybe"),
        ("Brand", " "),
        ("Model", ""),
    ],
)
def test_invalid_values_are_rejected(field, value):
    with pytest.raises(ValueError):
        read_catalog(csv_bytes([row(**{field: value})]))


def test_duplicate_identity_is_not_silently_dropped():
    with pytest.raises(ValueError, match="duplicate"):
        read_catalog(csv_bytes([row(), row("EXAMPLE   Phone 8/256GB")]))


@pytest.mark.parametrize(
    "data",
    [
        b"Smartphone,Brand\nPhone,Example\n",
        ",".join(COLUMNS).encode() + b"\n",
        csv_bytes([row()]).replace(b",231.60", b",231.60,unexpected"),
    ],
)
def test_schema_empty_catalog_and_extra_values_are_rejected(data):
    with pytest.raises(ValueError):
        read_catalog(data)


def test_repeated_preparation_replaces_one_json_and_invalid_input_keeps_it(tmp_path):
    source = tmp_path / "input.csv"
    original = csv_bytes([row()])
    source.write_bytes(original)
    output = tmp_path / "prepared/catalog.json"
    cards = prepare_catalog(source, output)
    content = output.read_bytes()
    assert json.loads(content) == cards
    assert len(list(output.parent.iterdir())) == 1
    prepare_catalog(source, output)
    assert output.read_bytes() == content
    assert source.read_bytes() == original
    source.write_bytes(csv_bytes([row(**{"Final Price": "199.99"})]))
    updated = prepare_catalog(source, output)
    assert len(json.loads(output.read_text())) == 1
    assert updated[0]["price"] == 199.99 and updated[0]["id"] == cards[0]["id"]
    content = output.read_bytes()
    source.write_bytes(csv_bytes([row(RAM="broken")]))
    with pytest.raises(ValueError):
        prepare_catalog(source, output)
    assert output.read_bytes() == content
    assert len(list(output.parent.iterdir())) == 1


def test_input_cannot_be_overwritten(tmp_path: Path):
    source = tmp_path / "input.csv"
    original = csv_bytes([row()])
    source.write_bytes(original)
    with pytest.raises(ValueError, match="paths must differ"):
        prepare_catalog(source, source)
    assert source.read_bytes() == original
