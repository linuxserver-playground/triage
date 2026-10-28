import pytest

from triage.sources import Sources


def test_paths_stay_inside_the_source(sources: Sources) -> None:
    for bad in ("../../app.pem", "/../triage.db", ".git/config", "images/../.git/config"):
        with pytest.raises(ValueError, match="leaves the source"):
            sources.read_file("hub", bad)
    with pytest.raises(ValueError, match="unknown source"):
        sources.read_file("config", "app.pem")
    assert ".git" not in sources.list_dir("hub", "")


def test_list_read_and_search(sources: Sources) -> None:
    assert "manifest.main.json" in sources.list_dir("hub", "images/sonarr")
    assert "4.0.20.3014-ls4" in sources.read_file("hub", "images/sonarr/latest.json")
    assert sources.grep("docs", "puid") == "general/docker-compose.md:3: Set PUID and PGID to your user."
    assert "sqlite" in sources.grep("hub", "sqlite", "images/sonarr")
    assert sources.grep("hub", "secret-ish") == "no matches"
    assert sources.grep("docs", "[unclosed") == "no matches"  # a bad regex is searched as text


def test_long_files_are_read_in_chunks(sources: Sources) -> None:
    (sources.hub / "big.txt").write_text("a" * 30_000 + "b" * 10)
    first = sources.read_file("hub", "big.txt")
    assert first.startswith("a") and "offset 30000" in first
    assert sources.read_file("hub", "big.txt", 30_000) == "b" * 10


def test_image_index(sources: Sources) -> None:
    assert sources.image_index() == "sonarr | app | main | Sonarr is a PVR for usenet users."
