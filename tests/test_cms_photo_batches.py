from datetime import UTC, datetime, timedelta

import httpx2 as httpx
import pytest

from scripts import cms_photo_batches as batches

START = datetime(2026, 11, 14, 2, 0, tzinfo=UTC)


def photo(i: int, seconds: float, kind="whole", view="other", conf=0.0, label="全体", garment=0):
    p = batches.Photo(
        i,
        f"IMG_{i}.DNG",
        START + timedelta(seconds=seconds),
        kind=kind,
        view=view,
        view_confidence=conf,
        label=label,
    )
    # テストでは、どの服かを image に入れておき、偽物の same が比べる
    p.image = str(garment).encode()
    return p


def front(i, seconds, conf=0.9, garment=0):
    return photo(i, seconds, view="front", conf=conf, label="正面", garment=garment)


def closeup(i, seconds, garment=0, label="タグのアップ"):
    return photo(i, seconds, kind="closeup", label=label, garment=garment)


def same(a, b):
    return 0.95 if a.image == b.image else 0.05


def split(photos):
    return batches.split_groups(photos, same)


def ids(groups):
    return [[p.id for p in g.photos] for g in groups]


def test_new_product_starts_at_the_next_whole_front():
    photos = [
        front(1, 0, garment=1),
        photo(2, 20, view="back", conf=0.9, label="背面", garment=1),
        closeup(3, 40, garment=1),
        front(4, 80, garment=2),
        photo(5, 100, view="back", conf=0.9, label="背面", garment=2),
    ]
    assert ids(split(photos)) == [[1, 2, 3], [4, 5]]


def test_retaken_front_of_the_same_garment_stays():
    photos = [front(1, 0, garment=1), closeup(2, 20, garment=1), front(3, 40, garment=1)]
    assert ids(split(photos)) == [[1, 2, 3]]


def test_closeups_of_the_next_garment_taken_before_its_front_move_forward():
    photos = [
        front(1, 0, garment=1),
        closeup(2, 20, garment=1),
        closeup(3, 40, garment=2),
        closeup(4, 50, garment=2),
        front(5, 60, garment=2),
    ]
    assert ids(split(photos)) == [[1, 2], [3, 4, 5]]


def test_worn_and_unsure_fronts_do_not_split():
    photos = [
        front(1, 0),
        photo(2, 10, kind="worn", view="front", conf=0.95, label="着用"),
        front(3, 20, conf=0.4),  # 向きに自信が無い
    ]
    assert ids(split(photos)) == [[1, 2, 3]]


def test_long_gap_starts_a_new_product_even_without_a_front():
    photos = [
        front(1, 0),
        photo(2, 30, kind="closeup", label="タグのアップ"),
        photo(3, 30 + 6 * 60, kind="closeup", label="ディテール"),
    ]
    assert ids(split(photos)) == [[1, 2], [3]]


def test_photos_without_time_stay_with_the_current_product():
    untimed = batches.Photo(2, "IMG_2.DNG", None, image=b"1", kind="closeup")
    photos = [front(1, 0, garment=1), untimed, front(3, 60, garment=2)]
    assert ids(split(photos)) == [[1, 2], [3]]


def test_group_confidence():
    assert batches.group_confidence(batches.Group([front(1, 0, conf=0.9)])) == 0.9
    assert batches.group_confidence(batches.Group([front(1, 0, conf=0.7)])) == 0.7
    assert batches.group_confidence(batches.Group([photo(1, 0, kind="closeup")])) == 0.4


class FakeModel:
    def __init__(self, choice, confidence):
        self.answer = {"choice": choice, "confidence": confidence}
        self.questions = None

    def ask(self, state, questions, images):
        self.questions = questions
        return {"choice": self.answer}


def test_ask_choice_maps_option_keys_back_to_names():
    model = FakeModel("o1", 0.8)
    name, confidence = batches.ask_choice(model, "s", "i", ["Shirts", "Coats & Jackets"], [b"x"])
    assert (name, confidence) == ("Coats & Jackets", 0.8)
    # 「&」や空白を含む名前はキーにしない
    assert model.questions["choice"]["criteria"] == {"o0": "Shirts", "o1": "Coats & Jackets"}
    assert batches.ask_choice(FakeModel("zz", 0.9), "s", "i", ["A"], [b"x"]) == (None, 0.9)


def test_parse_time():
    assert batches.parse_time("2026-11-14T02:53:41.000Z") == datetime(
        2026, 11, 14, 2, 53, 41, tzinfo=UTC
    )
    assert batches.parse_time(None) is None
    assert batches.parse_time("bad") is None


class FakeCms:
    """縮小版: bad を含む写真は 422、ほかは写真のバイト列。"""

    def __init__(self, status: int = 422):
        self.status = status

    def preview(self, path: str) -> bytes:
        if "bad" in path:
            request = httpx.Request("GET", f"https://cms.test{path}")
            raise httpx.HTTPStatusError(
                "x", request=request, response=httpx.Response(self.status, request=request)
            )
        return b"jpeg"


def _photos(*names: str):
    photos = [batches.Photo(i, name, None) for i, name in enumerate(names, start=1)]
    sources = [{"preview": f"/api/photoSources/{name}/preview?max=1280"} for name in names]
    return photos, sources


def test_load_previews_skips_photos_without_a_preview():
    photos, sources = _photos("a", "bad1", "b")
    readable = batches.load_previews(photos, sources, FakeCms())
    assert [p.filename for p in readable] == ["a", "b"]
    assert all(p.image == b"jpeg" for p in readable)


def test_load_previews_fails_when_nothing_is_readable():
    photos, sources = _photos("bad1", "bad2")
    with pytest.raises(RuntimeError, match="読める写真"):
        batches.load_previews(photos, sources, FakeCms())


def test_load_previews_does_not_hide_other_errors():
    photos, sources = _photos("a", "bad1")
    with pytest.raises(httpx.HTTPStatusError):
        batches.load_previews(photos, sources, FakeCms(status=500))


def test_try_read_tags_keeps_the_groups_when_the_reader_fails(monkeypatch, capsys):
    def boom():
        raise MemoryError("CUDA out of memory")

    monkeypatch.setattr(batches, "open_tag_reader", boom)
    items = [{"key": "g1"}]
    batches.try_read_tags(items, [batches.Group()])
    assert items == [{"key": "g1"}]
    assert "タグは読めませんでした" in capsys.readouterr().out
