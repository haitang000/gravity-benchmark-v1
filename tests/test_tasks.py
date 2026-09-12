from collections import Counter

from backend.benchmarks.tasks import dataset_hash, load_tasks


def test_full_is_layered_and_uses_independent_tasks():
    tasks = load_tasks("full", ["zh", "en"], 42)
    assert len(tasks) >= 60
    assert {task.difficulty for task in tasks} == {"easy", "medium", "hard"}
    assert {task.category for task in tasks} == {"instruction", "math", "coding", "performance"}
    assert len({task.id for task in tasks}) == len(tasks)
    assert Counter(task.difficulty for task in tasks)["hard"] >= 18


def test_quick_and_standard_do_not_include_hard():
    for suite in ("quick", "standard"):
        assert all(task.difficulty != "hard" for task in load_tasks(suite, ["zh", "en"], 42))


def test_dataset_hash_includes_extension():
    assert len(dataset_hash()) == 64
