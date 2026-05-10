import io
import os
import random
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

import super_tictactoe_rl as st


class BoardAndRulesTests(unittest.TestCase):
    def test_rect_board_has_expected_coordinates(self):
        board = st.make_rect_board(4, 4)

        self.assertEqual(len(board), 16)
        self.assertEqual(board.id_at(1, 1), 0)
        self.assertEqual(board.id_at(4, 4), 15)
        self.assertIsNone(board.id_at(5, 5))

    def test_super_board_has_six_4_by_4_blocks_and_three_levels(self):
        board = st.make_super_board(4)

        self.assertEqual(len(board), 96)
        self.assertEqual({cell.level for cell in board.cells}, {1, 2, 3})
        self.assertEqual(sum(cell.level == 1 for cell in board.cells), 16)
        self.assertEqual(sum(cell.level == 2 for cell in board.cells), 32)
        self.assertEqual(sum(cell.level == 3 for cell in board.cells), 48)

    def test_generate_lines_includes_row_column_and_diagonal_windows(self):
        board = st.make_rect_board(4, 4)
        lines = st.generate_lines(board, row_len=4, diag_len=4)
        line_sets = {(line.kind, line.ids) for line in lines}

        self.assertIn(("row", (0, 1, 2, 3)), line_sets)
        self.assertIn(("col", (0, 4, 8, 12)), line_sets)
        self.assertIn(("diag_down", (0, 5, 10, 15)), line_sets)
        self.assertIn(("diag_up", (12, 9, 6, 3)), line_sets)

    def test_deterministic_placement_puts_mark_on_chosen_square(self):
        board = st.make_rect_board(4, 4)
        state = st.new_state(board)

        new_state, placed, target = st.place_mark(state, board, 5, 1, noisy=False)

        self.assertTrue(placed)
        self.assertEqual(target, 5)
        self.assertEqual(new_state[5], 1)
        self.assertEqual(sum(value != 0 for value in new_state), 1)

    def test_noisy_placement_can_forfeit_when_sampled_neighbour_is_outside_board(self):
        board = st.make_rect_board(4, 4)
        state = st.new_state(board)
        original_random = random.random
        original_choice = random.choice

        try:
            random.random = lambda: 0.75
            random.choice = lambda seq: (-1, -1)
            new_state, placed, target = st.place_mark(state, board, 0, 1, noisy=True)
        finally:
            random.random = original_random
            random.choice = original_choice

        self.assertFalse(placed)
        self.assertIsNone(target)
        self.assertEqual(new_state, state)

    def test_check_win_finds_rows_columns_and_diagonals(self):
        board = st.make_rect_board(4, 4)
        lines = st.generate_lines(board, row_len=4, diag_len=4)

        row_state = tuple(1 if i in (0, 1, 2, 3) else 0 for i in range(16))
        col_state = tuple(1 if i in (0, 4, 8, 12) else 0 for i in range(16))
        diag_state = tuple(1 if i in (0, 5, 10, 15) else 0 for i in range(16))

        self.assertTrue(st.check_win(row_state, board, lines, 1))
        self.assertTrue(st.check_win(col_state, board, lines, 1))
        self.assertTrue(st.check_win(diag_state, board, lines, 1))

    def test_column_win_can_require_crossing_levels(self):
        board = st.make_super_board(4)
        lines = st.generate_lines(board, row_len=4, diag_len=5)

        same_level_ids = tuple(board.id_at(9, y) for y in range(1, 5))
        cross_level_ids = tuple(board.id_at(9, y) for y in range(4, 8))
        self.assertTrue(all(i is not None for i in same_level_ids))
        self.assertTrue(all(i is not None for i in cross_level_ids))

        same_level_state = tuple(1 if i in same_level_ids else 0 for i in range(len(board)))
        cross_level_state = tuple(1 if i in cross_level_ids else 0 for i in range(len(board)))

        self.assertTrue(st.check_win(same_level_state, board, lines, 1,
                                     column_must_cross_level=False))
        self.assertFalse(st.check_win(same_level_state, board, lines, 1,
                                      column_must_cross_level=True))
        self.assertTrue(st.check_win(cross_level_state, board, lines, 1,
                                     column_must_cross_level=True))


class LearningAndReportingTests(unittest.TestCase):
    def test_tabular_training_returns_q_table_policy_and_curve(self):
        random.seed(123)
        board = st.make_rect_board(3, 3)
        lines = st.generate_lines(board, row_len=3, diag_len=3)

        result = st.train_tabular_q(board, lines, episodes=20, eval_every=10,
                                    eval_games=1000)

        self.assertIn("Q", result)
        self.assertIn("policy", result)
        self.assertEqual([row["episode"] for row in result["curve"]], [10, 20])
        for row in result["curve"]:
            total = row["win_rate"] + row["draw_rate"] + row["loss_rate"]
            self.assertAlmostEqual(total, 1.0)

    def test_assignment_uses_1000_games_for_each_evaluation(self):
        calls = []

        def fake_train(name):
            def _fake(*args, **kwargs):
                calls.append((name, kwargs["eval_games"]))
                return {
                    "policy": lambda state, board, player, lines: st.empty_actions(state)[0],
                    "curve": [
                        {"episode": 1, "win_rate": 0.0, "draw_rate": 1.0, "loss_rate": 0.0}
                    ],
                }
            return _fake

        with patch.object(st, "train_tabular_q", side_effect=fake_train("tabular")), \
             patch.object(st, "train_linear_q", side_effect=fake_train("linear")), \
             patch.object(st, "save_curves_csv"), \
             patch.object(st, "save_curves_svg"), \
             redirect_stdout(io.StringIO()):
            st.run_assignment()

        self.assertEqual(calls, [
            ("tabular", 1000),
            ("tabular", 1000),
            ("linear", 1000),
        ])

    def test_linear_features_detect_immediate_win_and_block(self):
        board = st.make_rect_board(4, 4)
        lines = st.generate_lines(board, row_len=4, diag_len=4)
        cache = st.make_line_cache(board, lines)

        winning_state = tuple(1 if i in (0, 1, 2) else 0 for i in range(16))
        winning_features = st.features_for_action(winning_state, board, lines,
                                                  cache, 3, player=1)
        self.assertEqual(winning_features[9], 1.0)

        blocking_state = tuple(-1 if i in (0, 1, 2) else 0 for i in range(16))
        blocking_features = st.features_for_action(blocking_state, board, lines,
                                                   cache, 3, player=1)
        self.assertEqual(blocking_features[10], 1.0)

    def test_summary_threshold_uses_first_episode_to_reach_target(self):
        results = {
            "example": {
                "curve": [
                    {"episode": 10, "win_rate": 0.2, "draw_rate": 0.5, "loss_rate": 0.3},
                    {"episode": 20, "win_rate": 0.7, "draw_rate": 0.2, "loss_rate": 0.1},
                ]
            }
        }

        summary = st.summarize_convergence(results, threshold=0.6)

        self.assertEqual(summary[0]["episodes_to_60pct"], 20)
        self.assertEqual(summary[0]["best_win_rate"], 0.7)

    def test_csv_and_svg_outputs_are_created(self):
        results = {
            "example": {
                "curve": [
                    {"episode": 10, "win_rate": 0.25, "draw_rate": 0.5, "loss_rate": 0.25},
                    {"episode": 20, "win_rate": 0.50, "draw_rate": 0.25, "loss_rate": 0.25},
                ]
            }
        }

        with tempfile.TemporaryDirectory() as tmp:
            csv_path = os.path.join(tmp, "curves.csv")
            svg_path = os.path.join(tmp, "curves.svg")
            st.save_curves_csv(results, csv_path)
            st.save_curves_svg(results, svg_path)

            self.assertTrue(os.path.exists(csv_path))
            self.assertTrue(os.path.exists(svg_path))
            with open(csv_path, encoding="utf-8") as f:
                self.assertIn("experiment,episode,win_rate", f.read())
            with open(svg_path, encoding="utf-8") as f:
                self.assertIn("<svg", f.read())


if __name__ == "__main__":
    unittest.main()
