import numpy as np

from eogrow.utils.zipmap import map_values


def test_map_values():
    x = np.array([1, 5, -2, 3, 1, 1, 5, 2, 3, 1])
    mapping = {1: 11, 5: 15, -2: -3, 3: 3, 2: 3}
    result = np.array([11, 15, -3, 3, 11, 11, 15, 3, 3, 11])

    assert np.array_equal(map_values(x, mapping=mapping), result)

    new_shape = (2, 5, 1)
    reshaped_x, reshaped_result = x.reshape(new_shape), result.reshape(new_shape)
    assert np.array_equal(
        map_values(reshaped_x, mapping=mapping), reshaped_result
    ), "Does not map over differently shaped arrays correctly."

    assert map_values(x, mapping=mapping, dtype=np.uint8).dtype == np.uint8, "Dtype not set correctly."

    x[3] = 27
    result[3] = 137
    assert np.array_equal(map_values(x, mapping=mapping, default=137), result), "Default value not working correctly."
