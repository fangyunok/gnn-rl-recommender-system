from pathlib import Path

from gnn_rl_recommender.movielens import load_movielens_1m


def test_temporal_leave_two_out(tmp_path: Path) -> None:
    (tmp_path / "movies.dat").write_text(
        "10::A::Drama\n11::B::Comedy\n12::C::Drama\n13::D::Action\n", encoding="latin-1"
    )
    (tmp_path / "ratings.dat").write_text(
        "1::10::5::100\n1::11::4::200\n1::12::3::300\n1::13::5::400\n",
        encoding="latin-1",
    )
    split = load_movielens_1m(tmp_path)
    assert split.train.item_ids.tolist() == [0, 1]
    assert split.validation_item[0] == 2
    assert split.test_item[0] == 3

