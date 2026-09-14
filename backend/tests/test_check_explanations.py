from ml import check_explanations as check


def test_numbers_not_in_prompt_are_flagged_and_formatting_is_ignored():
    prompt = "score 0.37 ... files changed: 50 (average 5.7); lines added: 1,980 (average 347); true for 18% of pull requests"

    assert check.numbers_not_in("It changes 50 files and 1980 lines, above 5.7 and 347; 18% of PRs.", prompt) == []
    assert check.numbers_not_in("That is about 9 times the average of 5.7 files.", prompt) == ["9"]
    assert check.numbers_not_in("Score 0.37 over 2,000 lines.", prompt) == ["2,000"]


def test_alarming_words_depend_on_label_and_ignore_negation():
    assert check.alarming_words("This change is dangerous.", "High") == ["dangerous"]
    assert check.alarming_words("The risk is moderately elevated.", "Medium") == []
    assert check.alarming_words("The risk is moderately elevated.", "Low") == ["elevated"]
    assert check.alarming_words("This is a high risk change.", "Medium") == ["high risk"]
    assert check.alarming_words("This is a high risk change.", "High") == []
    assert check.alarming_words("Nothing here is dangerous or critical.", "Low") == []
    assert check.alarming_words("The short message was seen as less risky.", "Low") == []
    assert check.alarming_words("This change looks risky.", "Low") == ["risky"]


def test_sentence_count_ignores_decimals_and_markdown_is_detected():
    assert check.count_sentences("It changes 5.7 files. It adds 347 lines! Risk is low?") == 3
    assert check.has_markdown("- files changed: 50") is True
    assert check.has_markdown("It has **50** files.") is True
    assert check.has_markdown("It changes 50 files - more than usual.") is False
