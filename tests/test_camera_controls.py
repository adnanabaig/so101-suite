import copy
import json

import pytest

from so101.hardware.camera_controls import CameraProfile


class Driver:
    identity = "/dev/v4l/by-id/test-camera-serial-123"

    def __init__(self):
        self.values = {
            "exposure_auto": 3,
            "exposure_absolute": 313,
            "white_balance_temperature": 4000,
        }
        self.writes = []
        self.ignore = None

    def snapshot(self):
        return {
            name: {
                "value": value,
                "minimum": 1,
                "maximum": 10000,
                "step": 1,
                "writable": name != "exposure_absolute" or self.values["exposure_auto"] == 1,
            }
            for name, value in self.values.items()
        }

    def write(self, name, value):
        self.writes.append((name, value))
        if self.ignore != name:
            self.values[name] = value


def test_profile_unlocks_manual_control_verifies_and_restores_per_identity(tmp_path):
    driver = Driver()
    profile = CameraProfile(driver, tmp_path)
    result = profile.apply({"exposure_absolute": 100, "exposure_auto": 1})
    assert driver.writes == [("exposure_auto", 1), ("exposure_absolute", 100)]
    assert result["controls"]["exposure_absolute"]["value"] == 100
    assert json.loads(profile.path.read_text())["identity"] == driver.identity
    restored = Driver()
    loaded = CameraProfile(restored, tmp_path)
    assert restored.values["exposure_absolute"] == 100
    assert loaded.saved == profile.saved


def test_failed_readback_rolls_back_and_does_not_save(tmp_path):
    driver = Driver()
    before = copy.deepcopy(driver.values)
    driver.ignore = "exposure_absolute"
    profile = CameraProfile(driver, tmp_path)
    with pytest.raises(ValueError, match="retain"):
        profile.apply({"exposure_auto": 1, "exposure_absolute": 100})
    assert driver.values == before
    assert not profile.path.exists()


def test_manual_to_auto_profile_removes_inactive_override_and_survives_reconnect(tmp_path):
    driver = Driver()
    profile = CameraProfile(driver, tmp_path)
    profile.apply({"exposure_auto": 1, "exposure_absolute": 100})
    profile.apply({"exposure_auto": 3})
    assert profile.saved == {"exposure_auto": 3}
    restored = CameraProfile(Driver(), tmp_path)
    assert restored.error is None
    assert restored.saved == {"exposure_auto": 3}


def test_white_balance_manual_to_auto_profile_survives_reconnect(tmp_path):
    class WhiteBalanceDriver(Driver):
        def __init__(self):
            super().__init__()
            self.values["white_balance_auto"] = 1

        def snapshot(self):
            controls = super().snapshot()
            controls["white_balance_auto"]["minimum"] = 0
            controls["white_balance_temperature"]["writable"] = (
                self.values["white_balance_auto"] == 0
            )
            return controls

    profile = CameraProfile(WhiteBalanceDriver(), tmp_path)
    profile.apply({"white_balance_auto": 0, "white_balance_temperature": 4500})
    profile.apply({"white_balance_auto": 1})
    assert profile.saved == {"white_balance_auto": 1}
    restored = CameraProfile(WhiteBalanceDriver(), tmp_path)
    assert restored.error is None
    assert restored.saved == {"white_balance_auto": 1}


@pytest.mark.parametrize(
    "changes",
    [
        {"unknown": 2},
        {"exposure_absolute": 0},
        {"exposure_absolute": True},
        {"exposure_absolute": float("nan")},
        {"exposure_absolute": 100},
    ],
)
def test_invalid_or_inactive_controls_cannot_change_camera(tmp_path, changes):
    driver = Driver()
    with pytest.raises(ValueError):
        CameraProfile(driver, tmp_path).apply(changes)
    assert not driver.writes


def test_disk_failure_restores_settings(monkeypatch, tmp_path):
    import so101.hardware.camera_controls as controls

    driver = Driver()
    original = dict(driver.values)
    profile = CameraProfile(driver, tmp_path)

    def fail(*args):
        raise OSError("full")

    monkeypatch.setattr(controls, "atomic_write_json", fail)
    with pytest.raises(OSError):
        profile.apply({"exposure_auto": 1, "exposure_absolute": 100})
    assert driver.values == original


def test_profile_from_another_device_is_not_applied(tmp_path):
    driver = Driver()
    profile = CameraProfile(driver, tmp_path)
    profile.path.write_text(json.dumps({"identity": "other", "values": {"exposure_auto": 1}}))
    loaded = CameraProfile(driver, tmp_path)
    assert "identity mismatch" in loaded.error
    assert not driver.writes


def test_realsense_manual_fields_require_explicit_auto_off_and_gain_does_not_break_reconnect(
    tmp_path,
):
    from types import SimpleNamespace

    from so101.hardware.camera_controls import RealSenseControls

    class Sensor:
        def __init__(self):
            self.values = {
                "exposure_auto": 1,
                "exposure_absolute": 166,
                "gain": 64,
                "white_balance_auto": 1,
                "white_balance_temperature": 4600,
            }

        def supports(self, option):
            return True

        def get_option_range(self, option):
            return SimpleNamespace(min=0, max=10000, step=1, default=0)

        def get_option(self, option):
            return self.values[option]

        def is_option_read_only(self, option):
            return False

        def set_option(self, option, value):
            self.values[option] = value

    driver = object.__new__(RealSenseControls)
    driver.sensor = Sensor()
    driver.options = {name: name for name in driver.sensor.values}
    driver.identity = "realsense:test"
    profile = CameraProfile(driver, tmp_path)
    assert profile.snapshot()["controls"]["exposure_absolute"]["inactive"]
    with pytest.raises(ValueError, match="inactive"):
        profile.apply({"exposure_absolute": 100})
    profile.apply({"exposure_auto": 0, "gain": 80, "exposure_absolute": 100})
    profile.apply({"exposure_auto": 1})
    assert profile.saved == {"exposure_auto": 1}
    loaded = CameraProfile(driver, tmp_path)
    assert loaded.error is None


def test_independent_v4l2_gain_is_preserved_under_automatic_exposure(tmp_path):
    class GainDriver(Driver):
        def __init__(self):
            super().__init__()
            self.values["gain"] = 64

    driver = GainDriver()
    profile = CameraProfile(driver, tmp_path)
    profile.apply({"exposure_auto": 3})
    profile.apply({"gain": 80})
    assert profile.saved == {"exposure_auto": 3, "gain": 80}
    replacement = GainDriver()
    loaded = CameraProfile(replacement, tmp_path)
    assert replacement.values["gain"] == 80
    assert loaded.error is None
