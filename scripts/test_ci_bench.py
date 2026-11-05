import unittest

from ci_bench import check_ran_to_length, rate, verdict


class RateTest(unittest.TestCase):
    def test_removes_the_time_of_the_short_run(self):
        self.assertEqual(rate(129, 5.0, 1.0), 32.0)

    def test_a_long_run_that_is_not_slower_is_an_error(self):
        with self.assertRaises(ValueError):
            rate(129, 1.0, 1.0)


class RanToLengthTest(unittest.TestCase):
    def test_a_full_output_passes(self):
        check_ran_to_length("word " * 123)

    def test_the_99_words_of_a_full_linux_run_pass(self):
        # qwen2.5-0.5b-instruct, greedy, 129 tokens on linux-x64 at b60b812.
        check_ran_to_length("word " * 99)

    def test_the_39_words_of_an_early_stop_fail(self):
        with self.assertRaises(ValueError):
            check_ran_to_length("word " * 39)


class VerdictTest(unittest.TestCase):
    def test_a_drop_within_the_noise_does_not_warn(self):
        self.assertEqual(verdict(91.0, 100.0), (0.91, False))
        self.assertEqual(verdict(85.0, 100.0), (0.85, False))

    def test_a_larger_drop_warns(self):
        self.assertEqual(verdict(84.0, 100.0), (0.84, True))

    def test_a_gain_does_not_warn(self):
        self.assertEqual(verdict(114.0, 100.0), (1.14, False))


if __name__ == "__main__":
    unittest.main()
