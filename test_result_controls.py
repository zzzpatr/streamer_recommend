import unittest

import numpy as np
import pandas as pd

from preference_state import empty_preferences
from recommender import load_tag_index, rank_streamers
from result_controls import dismiss_current_batch, streamer_similarity_scores


def _streamers(count: int = 6) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "pfid": str(index),
                "gender": "女",
                "personality": "活潑開朗型",
                "appearance": "長髮",
                "talents": "歌唱",
                "featured_topics": "日常輕鬆閒聊",
                "live_streaming_style": "親切友善",
                "overall_vibe": f"主播 {index}",
                "reasons": "{}",
                "self_description": f"自我介紹 {index}",
            }
            for index in range(1, count + 1)
        ]
    )


class ResultControlTests(unittest.TestCase):
    def test_similarity_reuses_normalized_streamer_vectors(self):
        index = (
            np.array(["a", "b", "c"]),
            np.array(
                [
                    [1.0, 0.0],
                    [0.8, 0.6],
                    [0.0, 1.0],
                ],
                dtype=np.float32,
            ),
            {"model": "test"},
        )

        scores = streamer_similarity_scores(index, "a").set_index("pfid")

        self.assertAlmostEqual(float(scores.loc["a", "vector_score"]), 1.0)
        self.assertAlmostEqual(float(scores.loc["b", "vector_score"]), 0.8)
        self.assertAlmostEqual(float(scores.loc["c", "vector_score"]), 0.0)

    def test_replace_one_fills_from_the_next_ranked_candidate(self):
        streamers = _streamers()
        preferences = empty_preferences()
        preferences["talents"] = ["歌唱"]
        vector_scores = pd.DataFrame(
            {
                "pfid": [str(index) for index in range(1, 7)],
                "vector_score": [0.9, 0.8, 0.7, 0.6, 0.5, 0.4],
            }
        )
        kwargs = {
            "streamers": streamers,
            "tag_index": load_tag_index(streamers),
            "preferences": preferences,
            "excluded_preferences": empty_preferences(),
            "vector_scores": vector_scores,
            "top_n": 5,
        }

        original = rank_streamers(**kwargs)
        replaced = rank_streamers(**kwargs, hidden_pfids={"3"})

        self.assertEqual(original["pfid"].tolist(), ["1", "2", "3", "4", "5"])
        self.assertEqual(replaced["pfid"].tolist(), ["1", "2", "4", "5", "6"])

    def test_change_batch_does_not_mutate_existing_set(self):
        dismissed = {"1"}
        updated = dismiss_current_batch(dismissed, ["2", "3"])
        self.assertEqual(dismissed, {"1"})
        self.assertEqual(updated, {"1", "2", "3"})


if __name__ == "__main__":
    unittest.main()
