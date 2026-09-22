import numpy as np

from hatchmatch.augment import augment_training_sample


def _sample() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    image = np.full((48, 48), 255, np.uint8)
    image[8:40:4, 6:42] = 0
    query = image[8:32, 8:32].copy()
    target = np.zeros((48, 48), bool)
    target[12:36, 12:36] = True
    known = np.zeros((48, 48), bool)
    known[4:44, 4:44] = True
    return image, query, target, known


def test_augmentation_is_exactly_deterministic_for_a_seed() -> None:
    image, query, target, known = _sample()

    first = augment_training_sample(image, query, target, known, seed=1234)
    second = augment_training_sample(image, query, target, known, seed=1234)

    assert set(first) == {"image", "query", "target", "known"}
    for key in first:
        np.testing.assert_array_equal(first[key], second[key])


def test_augmentation_changes_with_seed_and_preserves_mask_contract() -> None:
    image, query, target, known = _sample()

    first = augment_training_sample(image, query, target, known, seed=1)
    second = augment_training_sample(image, query, target, known, seed=2)

    assert not np.array_equal(first["image"], second["image"])
    assert not np.array_equal(first["query"], second["query"])
    assert first["image"].shape == image.shape
    assert first["query"].shape == query.shape
    assert first["target"].shape == target.shape
    assert first["known"].shape == known.shape
    assert first["image"].dtype == first["query"].dtype == np.uint8
    assert first["target"].dtype == first["known"].dtype == np.bool_
    assert np.all(first["target"] <= first["known"])


def test_augmentation_does_not_mutate_inputs() -> None:
    arrays = _sample()
    originals = tuple(array.copy() for array in arrays)

    augment_training_sample(*arrays, seed=9)

    for actual, expected in zip(arrays, originals, strict=True):
        np.testing.assert_array_equal(actual, expected)
