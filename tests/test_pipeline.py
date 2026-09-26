"""데이터 누출 방지와 임곗값 경계 처리를 작은 합성 데이터로 검증한다."""
import unittest
import numpy as np
import pandas as pd
from press_pipeline import FEATURES, prepare, select_threshold, windows, evaluate


class PipelineTests(unittest.TestCase):
    def test_window_future_label(self):
        values = np.arange(130 * 3).reshape(130, 3)
        x, y = windows(values, np.arange(130))
        self.assertEqual(x.shape, (10, 20, 3))
        np.testing.assert_array_equal(x[0], values[:20])
        np.testing.assert_array_equal(y, np.arange(120, 130))

    def test_split_and_scaler_fit_only_training(self):
        def frame(n, label):
            df = pd.DataFrame({col: np.arange(n, dtype=float) for col in FEATURES})
            df["Equipment_state"] = label
            return df
        frames = {"normal": frame(20000, 0), "anomaly": frame(600, 1)}
        for mode, valid_count, test_count in [("guide", 1180, 4180), ("strict", 1060, 4060)]:
            data, scaler = prepare(frames, mode)
            self.assertEqual(data["train"].shape, (14880, 20, 3))
            self.assertEqual(len(data["valid"]), valid_count)
            self.assertEqual(len(data["test"]), test_count)
            np.testing.assert_array_equal(scaler.data_max_, [14999] * 3)
            if mode == "strict":
                # 정상 검증의 마지막 입력과 미래 라벨(100+1 시점 뒤)도 테스트 시작 이전이다.
                last_input = data["valid_normal"][-1, -1, 0] * 14999
                test_start = data["test"][0, 0, 0] * 14999
                self.assertLess(last_input + 101, test_start)

    def test_threshold_ties_match_reported_precision(self):
        labels = np.array([0, 0, 1, 1])
        scores = np.array([0.1, 0.2, 0.2, 0.9])
        threshold, detail = select_threshold(labels, scores, "f1")
        result = evaluate(labels, scores, threshold)
        self.assertAlmostEqual(result["precision"], detail["precision"])
        self.assertAlmostEqual(result["recall"], detail["recall"])


if __name__ == "__main__":
    unittest.main()
