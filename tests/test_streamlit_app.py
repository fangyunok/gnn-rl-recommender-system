from streamlit.testing.v1 import AppTest


def test_streamlit_demo_renders_real_model() -> None:
    app = AppTest.from_file("streamlit_app.py").run(timeout=30)
    assert not app.exception
    assert len(app.metric) == 4
    assert len(app.dataframe) == 1
    assert app.metric[0].value == "0.0462 ± 0.0024"

