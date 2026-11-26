"""Prepare a local smartphone CSV as a JSON array of validated text cards."""

import argparse
import csv
import io
import json
import math
import tempfile
import unicodedata
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

COLUMNS = ("Smartphone", "Brand", "Model", "RAM", "Storage", "Color", "Free", "Final Price")


def memory_gb(value: str) -> int | None:
    if not value.strip():
        return None
    if not value.strip().isascii() or not value.strip().isdigit() or int(value) <= 0:
        raise ValueError("memory must be a positive integer or empty")
    return int(value)


def make_card(raw: dict[str, str]) -> dict:
    if set(raw) != set(COLUMNS) or any(value is None for value in raw.values()):
        raise ValueError("row must contain exactly eight fields")
    for field in ("Smartphone", "Brand", "Model", "Color"):
        if not raw[field].strip():
            raise ValueError(f"{field} is required")
    ram, storage = memory_gb(raw["RAM"]), memory_gb(raw["Storage"])
    price = float(raw["Final Price"])
    if not math.isfinite(price) or price <= 0:
        raise ValueError("price must be finite and positive")
    if raw["Free"].strip() not in {"Yes", "No"}:
        raise ValueError("Free must be Yes or No")
    name = " ".join(raw["Smartphone"].split())
    identity = unicodedata.normalize("NFC", name).casefold()
    unlocked = raw["Free"].strip() == "Yes"
    # Keep the original UUID namespace so project renames do not change catalog IDs.
    return {
        "id": str(uuid5(NAMESPACE_URL, f"rag-phones:smartphones#{identity}")),
        "name": name,
        "brand": raw["Brand"].strip(),
        "model": raw["Model"].strip(),
        "ram_gb": ram,
        "storage_gb": storage,
        "color": raw["Color"].strip(),
        "carrier_unlocked": unlocked,
        "price": price,
        "currency": "USD",
        "currency_status": "user_selected",
        "source_currency": None,
        "conversion_applied": False,
        "raw_fields": dict(raw),
        "text": (
            f"Название: {name}. Бренд: {raw['Brand'].strip()}. "
            f"Модель: {raw['Model'].strip()}. "
            f"RAM, ГБ: {ram if ram is not None else 'неизвестно'}. "
            f"Накопитель, ГБ: {storage if storage is not None else 'неизвестно'}. "
            f"Цвет: {raw['Color'].strip()}. "
            f"Без привязки к оператору: {'да' if unlocked else 'нет'}. "
            f"Исходная цена: {raw['Final Price']}. Валюта проекта: USD. "
            "Валюта источника не подтверждена; пересчёт не выполнялся."
        ),
    }


def read_catalog(raw_bytes: bytes) -> list[dict]:
    reader = csv.DictReader(io.StringIO(raw_bytes.decode("utf-8-sig")), strict=True)
    if not reader.fieldnames or sorted(reader.fieldnames) != sorted(COLUMNS):
        raise ValueError(f"CSV columns must be: {', '.join(COLUMNS)}")
    cards, ids = [], set()
    for number, raw in enumerate(reader, start=2):
        try:
            card = make_card(raw)
            if card["id"] in ids:
                raise ValueError("duplicate smartphone name")
        except ValueError as error:
            raise ValueError(f"record {number}: {error}") from error
        cards.append(card)
        ids.add(card["id"])
    if not cards:
        raise ValueError("CSV contains no records")
    return cards


def prepare_catalog(input_path: Path, output_path: Path) -> list[dict]:
    if input_path.resolve() == output_path.resolve():
        raise ValueError("input and output paths must differ")
    cards = read_catalog(input_path.read_bytes())
    content = json.dumps(cards, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=output_path.parent, delete=False) as temporary:
        temporary_path = Path(temporary.name)
    try:
        temporary_path.write_text(content, encoding="utf-8")
        temporary_path.replace(output_path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return cards


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("var/raw/smartphones.csv"))
    parser.add_argument("--output", type=Path, default=Path("var/prepared/catalog.json"))
    args = parser.parse_args()
    try:
        cards = prepare_catalog(args.input, args.output)
    except (ValueError, OSError, UnicodeError, csv.Error) as error:
        parser.exit(1, f"Catalog preparation failed: {error}\n")
    print(f"Prepared {len(cards)} cards: {args.output}")
    print(f"Missing RAM: {sum(c['ram_gb'] is None for c in cards)}")
    print(f"Missing storage: {sum(c['storage_gb'] is None for c in cards)}")


if __name__ == "__main__":
    main()
