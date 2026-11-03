import pytest
import elodin.ui as ui


def test_place_names_survive_kdl_and_python_conversion():
    scene = ui.schematic(
        place_names=ui.place_names(
            [
                ("LC-39A", 28.60839, -80.60433, 3.0),
                ("Landing Zone 1", 28.48580, -80.54440, 5.0),
            ],
            visible=True,
        )
    )
    text = scene.emit_kdl()
    assert "place_names" in text and "Landing Zone 1" in text
    assert ui.from_kdl(text) == scene
    namespace = {}
    exec(ui.to_python(text), namespace)
    assert namespace["build"]() == scene


@pytest.mark.parametrize(
    "entry",
    [
        ("", 0, 0, 0),
        ("bad", 91, 0, 0),
        ("bad", 0, 181, 0),
        ("bad", 0, 0, float("nan")),
        ("bad", float("inf"), 0, 0),
    ],
)
def test_invalid_names_fail_before_loading_the_editor(entry):
    with pytest.raises(ValueError):
        ui.place_names([entry])
