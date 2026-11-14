import pytest

from app import tag_reader
from scripts import cms_photo_batches as batches


def test_parse_answer_reads_json_inside_text():
    text = (
        'Here it is:\n```json\n{"brand": "NORTHWEST TERRITORY", "material": "Cotton 100%",'
        ' "origin_country": "China", "origin_code": "cn", "size": "XL"}\n```'
    )
    got = tag_reader.parse_answer(text)
    assert got == {
        "brand": "NORTHWEST TERRITORY",
        "material": "Cotton 100%",
        "origin_country": "China",
        "origin_code": "CN",
        "size": "XL",
    }


def test_parse_answer_turns_unreadable_values_into_none():
    got = tag_reader.parse_answer(
        '{"brand": "", "material": "null", "origin_country": "unknown",'
        ' "origin_code": "USA", "size": null}'
    )
    assert got == {
        "brand": None,
        "material": None,
        "origin_country": None,
        "origin_code": None,
        "size": None,
    }


def test_parse_answer_without_json_is_an_error():
    with pytest.raises(tag_reader.TagReaderError):
        tag_reader.parse_answer("I cannot read the tag.")


def test_tag_images_are_the_tag_closeups():
    photos = [
        batches.Photo(1, "a", None, image=b"front", label="正面"),
        batches.Photo(2, "b", None, image=b"tag1", label="タグのアップ"),
        batches.Photo(3, "c", None, image=b"tag2", label="タグのアップ"),
    ]
    assert batches.tag_images(photos) == [b"tag1", b"tag2"]


def test_reading_fields_map_to_cms_names():
    reading = {"brand": "Levi's", "material": "100% COTTON/COTON", "origin_code": "US", "size": "L"}
    assert batches.reading_fields(reading) == {
        "brandName": "Levi's",
        "material": "Cotton 100%",
        "originCountry": "US",
        "tagSize": "L",
    }


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("100% COTTON", "Cotton 100%"),
        ("100% COTTON/COTON", "Cotton 100%"),
        ("65% POLYESTER/35% COTTON", "Polyester 65% / Cotton 35%"),
        ("55% RAMIE 45% COTTON", "Ramie 55% / Cotton 45%"),
        ("SHELL: 100% SILK LINING: 100% POLYESTER", "表地: Silk 100%、裏地: Polyester 100%"),
        (
            "SHELL/EXTERIOR: 100% ACRYLIC / ACRÍLICA LINING/EDORRO: 100% NYLON / NYLON",
            "表地: Acrylic 100%、裏地: Nylon 100%",
        ),
        (
            "SHELL: 50% WOOL 39% ACRYLIC 10% NYLON 1% OTHERS"
            " LINING: 100% NYLON FILLER: 100% POLYESTER",
            "表地: Wool 50% / Acrylic 39% / Nylon 10% / その他 1%"
            "、裏地: Nylon 100%、中綿: Polyester 100%",
        ),
        ("PARKA IMPROVED RAINSU", None),
        ("SUEDED FINISH", None),
        (None, None),
    ],
)
def test_normalize_material(text, expected):
    assert tag_reader.normalize_material(text) == expected
