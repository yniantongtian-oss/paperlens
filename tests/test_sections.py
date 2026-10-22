"""Section detection and the per-call character budget."""

from paperlens.sections import chunk_text, plan_chunks, split_sections


def test_known_and_numbered_headings():
    text = (
        "Widget nets\n\n"
        "Abstract\n"
        "Widgets help.\n\n"
        "1. Introduction\n"
        "We study widgets.\n\n"
        "3.1 Architecture\n"
        "A small encoder.\n\n"
        "# Results and Discussion\n"
        "Gain is 12 percent.\n"
    )
    sections = dict(split_sections(text))
    assert sections["Preamble"] == "Widget nets"
    assert sections["Abstract"] == "Widgets help."
    assert sections["1. Introduction"] == "We study widgets."
    assert sections["3.1 Architecture"] == "A small encoder."
    assert "12 percent" in sections["Results and Discussion"]


def test_numbered_sentences_stay_in_the_body():
    text = "1. We trained the model on ImageNet.\n\nIt worked.\n"
    sections = split_sections(text)
    assert sections == [("Document", text.strip())]


def test_no_headings_is_one_document():
    assert split_sections("just a paragraph") == [("Document", "just a paragraph")]


def test_chunks_respect_max_chars_and_keep_the_tail():
    text = ("alpha " * 30) + "TAIL_TOKEN"
    chunks = chunk_text(text, 40)
    assert len(chunks) > 1
    assert all(len(chunk) <= 40 for chunk in chunks)
    assert chunks[-1].endswith("TAIL_TOKEN") or "TAIL_TOKEN" in chunks[-1]
    assert "TAIL_TOKEN" not in chunks[0]


def test_plan_chunks_labels_section_parts():
    body = "word " * 50
    text = f"1. Introduction\n{body}INTRO\n\n2. Results\n{body}RESULT\n"
    chunks = plan_chunks(text, 80)
    titles = [chunk.title for chunk in chunks]
    assert any(title.startswith("1. Introduction (part ") for title in titles)
    assert any(title.startswith("2. Results (part ") for title in titles)
    assert all(len(chunk.text) <= 80 for chunk in chunks)
    assert any("RESULT" in chunk.text for chunk in chunks)
    assert any("INTRO" in chunk.text for chunk in chunks)
